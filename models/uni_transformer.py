import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum

from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product


class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):

        # h              all_v[N*E_all, 128]
        # r_feat     dist_feat[N*E_all*k, 80]
        # edge_feat  edge_feat[N*E_all*k, 4]
        # edge_index edge_index[2, N*E_all*k]
        # e_w             e_w[N*E_all*k, 1]

        N = h.size(0) # N*E_all
        src, dst = edge_index
        hi, hj = h[dst], h[src] # hi[N*E_all*k, 128] destination hj[N*E_all*k, 128] neighboring

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1) # [N*E_all*k, 80+128+128]
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1) # # [N*E_all*k, 4+80+128+128]

        # compute k
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads) # [N*E_all*k, 128] # [N*E_all*k, 16, 8]
        # compute v
        v = self.hv_func(kv_input) # # [N*E_all*k, 128]

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1) # [N*E_all*k, 1]
        else:
            e_w = 1.
        v = v * e_w # [N*E_all*k, 128]
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads) # [N*E_all*k, 16, 8]

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads) # [N*E_all, 16, 8]

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads] # [N*E_all*k, 16]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head) # [N*E_all*k, 16, 8]
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)  # [N*E_all, 16, 8]
        output = output.view(-1, self.output_dim) # [N*E_all, 128]
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h # [N*E_all, 128]
        return output


class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):

        # new_h h[N*E_all, 128] h_out include p
        # rel_x [N*E_all*k, 3] neighbor destina
        # dist_feat [N*E_all*k, 80]
        # edge_feat [N*E_all*k, 4]
        # edge_index  [2, N*E_all*k]  dst src
        # e_w [N*E_all*k, 1]  q k v v_weight

        N = h.size(0) # N*E_all
        src, dst = edge_index # [N*E_all*k, ] # [N*E_all*k, ]
        hi, hj = h[dst], h[src] # [N*E_all*k, 128] # [N*E_all*k, 128]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1) # [N*E_all*k, 80+128+128]
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1) # [N*E_all*k, 4+80+128+128]

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads) # [N*E_all*k, 128] [N*E_all*k, 16, 8]
        v = self.xv_func(kv_input) # [N*E_all*k, 16]
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None: # True
            e_w = e_w.view(-1, 1) # [N*E_all*k, 1]
        else:
            e_w = 1.
        v = v * e_w  # [N*E_all*k, 16]

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3] # [N*E_all*k, 16, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads) # [N*E_all, 128] # [N*E_all*k, 16, 8]

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads) # [N*E_all*k, 16]

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3) # [N*E_all*k, 16, 3]
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3) # [N*E_all, 16, 3]
        return output.mean(1)  # [num_nodes, 3] # [N*E_all, 3]


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    # h: pv0+vt [N*E_protein+N*E_ligand, 128]
    # x: px0+xt [N*E_protein+N*E_ligand, 3]
    # edge_attr [N*E_all*k, 4] 指示edge的neighbor destination分别来自于protein or ligand
    # # neighbor=l destination=l 0
    # # neighbor=l destination=p 1
    # # neighbor=p destination=l 2
    # # neighbor=p destination=p 3
    # edge_index [2, N*E_all*k] from knn_graph
    # e_w [N*E_all*k, 1] distance_feature from x and edge_index

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):
        src, dst = edge_index # [N*E_all*k, ] [N*E_all*k, ]
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types] [N*E_all*k, 4]
        else:
            edge_feat = None

        rel_x = x[dst] - x[src] # neighbor destination 的相对位置 [N*E_all*k, 3]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True) # neighbor destination 的L2距离 [N*E_all*k, ]

        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist) # [N*E_all*k, 20]
            dist_feat = outer_product(edge_attr, dist_feat) # [N*E_all*k, 80]
            # h_in h[N*E_all, 128] pv0+vt
            # dist_feat [N*E_all*k, 80]
            # edge_feat [N*E_all*k, 4]
            # edge_index  [2, N*E_all*k]  dst src
            # e_w [N*E_all*k, 1]  q k v v_weight
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out  # [N*E_all, 128]
        x2h_out = h_in # pre_v0 # [N*E_all, 128]

        new_h = h if self.sync_twoup else x2h_out # h_out [N*E_all, 128]
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            # new_h h[N*E_all, 128] h_out include pv+pre_v0
            # rel_x [N*E_all*k, 3] neighbor destination 的相对位置
            # dist_feat [N*E_all*k, 80]
            # edge_feat [N*E_all*k, 4]
            # edge_index  [2, N*E_all*k]  dst src
            # e_w [N*E_all*k, 1]  q k v v_weight
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w) # [N*E_all, 3]
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated  # [N*E_all, 3]
            rel_x = x[dst] - x[src] # [N*E_all*k, 3]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True) # [N*E_all*k, ]

        return x2h_out, x # all_v all_x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        # Build the network
        # h_all,  # protein_v+vt[N*E_protein+N*E_ligand, 128]
        # pos_all # protein_x+xt[N*E_protein+N*E_ligand, 3]
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.k = k
        self.ew_net_type = ew_net_type  # [r, m, none]

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch):
        # # px0+xt[all, 3]
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn': # True
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index) # all = 0
        n_src = mask_ligand[src] == 1  # True for ligand False for protein
        n_dst = mask_ligand[dst] == 1  # True for ligand False for protein
        edge_type[n_src & n_dst] = 0   # neighbor=l destination=l 0
        edge_type[n_src & ~n_dst] = 1  # neighbor=l destination=p 1
        edge_type[~n_src & n_dst] = 2  # neighbor=p destination=l 2
        edge_type[~n_src & ~n_dst] = 3 # neighbor=p destination=p 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type

    def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):

        all_x = [x]
        all_h = [h]

        # h_all,  # protein_v+vt[N*E_protein+N*E_ligand, 128]
        # x_all   # protein_x+xt[N*E_protein+N*E_ligand, 3]

        for b_idx in range(self.num_blocks):
            edge_index = self._connect_edge(x, mask_ligand, batch) # px0+xt[all, 3] ->
            src, dst = edge_index # dst:index of center destination src:index of neighboring k position

            # edge type (dim: 4)
            edge_type = self._build_edge_type(edge_index, mask_ligand)
            # neighbor=l destination=l 0
            # neighbor=l destination=p 1
            # neighbor=p destination=l 2
            # neighbor=p destination=p 3
            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            for l_idx, layer in enumerate(self.base_block):
                # print(f'layer: {l_idx}')
                h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)
            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs
