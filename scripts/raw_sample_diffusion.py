import argparse
import os
import shutil
import sys
import time

import numpy as np
import pickle
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

path = os.path.abspath(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)
sys.path.append(path)

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior'):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
        # batch = 一个protein * 100， 一个ligand * 100
        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior': # True
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy()) # protein pocket size 用于预测ligand的atom num
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)] # [100, 1] ligand atom num
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device) # [100*E, 1]
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0) # [100, 3] protein center pos
            batch_center_pos = center_pos[batch_ligand] # [100*E, 3] center pos for ligand pos
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos) # [100*E, 3]  ligand pos xt

            # init ligand v
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device) # [100*E, 13] ligand atom feature vt
                init_ligand_v = log_sample_categorical(uniform_logits)  # [100*E, 1] ligand atom feature vt

            # protein_pos            px0[100*E, 3]   raw no rel
            # protein_atom_feature   pv0[100*E, 27]
            # init_ligand_pos         xt[100*E, 3]   raw no rel
            # init_ligand_v           vt[100*E, 1]

            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode
            )
            # abs_pos pre_xt-1_1000[100*E, 3]   pre_vt-1_1000[100*E, 1]   pre_xt-1_ls:pre_xt-1[step, 100*E, 3]    pre_vt-1_ls:pre_vt-1[step, 100*E, 1]
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj'] # ls:pre_v0[step, 100*E, 13] ls:pre_vt-1[step, 100*E, 13]
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            # 得到每个ligand的起始index和结束index
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]
            # num_samples * [num_atoms_i, 3] all_pre_xt-1_1000[100, E, 3]
            # 切分pre_xt-1_1000[100*E, 3]  100个ligand 每个ligand的E不同  但都是由同一个protein得到的，切分为100个ligand

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]
            # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]
            # all_pre_vt-1_1000[100, E, 1]
            # 切分pre_vt-1_1000[100*E, 1]  100个ligand 每个ligand的E不同  但都是由同一个protein得到的，切分为100个ligand

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only: # True
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list
    # 将batch=100的数据进行切分
    # abs_pos pre_xt-1_1000[100, E, 3] 最后一轮的xt-1 100个ligand，E不同，对应同一个protein的结果
    #         pre_vt-1_1000[100, E, 1] 最后一轮的vt-1 100个ligand，E不同，对应同一个protein的结果
    # abs_pos ls:pre_xt-1[100, step, E, 3]  100个ligand，E不同，对应同一个protein的结果  每个ligand最后step的结果作为pre_xt-1_1000
    #         ls:pre_vt-1[100, step, E, 1]  100个ligand，E不同，对应同一个protein的结果  每个ligand最后step的结果作为pre_vt-1_1000
    #         ls:pre_v0  [100, step, E, 13] 100个ligand，E不同，对应同一个protein的结果
    #         ls:pre_vt-1[100, step, E, 13] 100个ligand，E不同，对应同一个protein的结果  每个ligand最后step的结果作为pre_vt-1_1000，编码不同


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/raw_sampling.yml')
    parser.add_argument('-i', '--data_id', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--sample_logdir', type=str, default='../logs/')
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--diffusion_path', type=str, default='../logs_diffusion/raw_training_20240415_090924/checkpoints/256000.pt')
    args = parser.parse_args()

    begin_time = time.time()

    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    log_dir = misc.get_new_log_dir(args.sample_logdir, prefix=config_name, tag=args.tag)
    logger = misc.get_logger('sampling', log_dir)

    # Load config
    config = misc.load_config(args.config)
    # logger.info(config)
    for item in list(config):
        logger.info(f'\n Sampling Config: {item}')
        for item2 in str(config[item]).replace('\'', '').split(','):
            logger.info(f'Sampling Config: {item2}')
    misc.seed_all(config.sample.seed)

    # modify checkpoint of diffusion path
    config.model.checkpoint = args.diffusion_path

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    # logger.info(f"Training Config: {ckpt['config']}")
    for item in list(ckpt['config']):
        logger.info(f'\n Training Config: {item}')
        for item2 in str(ckpt['config'][item]).replace('\'', '').split(','):
            logger.info(f'Training Config: {item2}')

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # modify data_path
    ckpt['config'].data.path = args.data_path + 'crossdocked_v1.1_rmsd1.0_pocket10'
    ckpt['config'].data.split = args.data_path + 'crossdocked_pocket10_pose_split.pt'

    # Load dataset
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    test_data_ls = []
    if '-' in args.data_id:
        begin_data_index = int(args.data_id.split('-')[0])
        end_data_index = int(args.data_id.split('-')[1])
        test_data_ls = [i for i in range(begin_data_index, end_data_index+1)]
    else: # just one data
        test_data_ls = [int(args.data_id)]
    logger.info(f'Sample test data index: {test_data_ls}')

    for test_data_index in test_data_ls:
        logger.info('\n')
        logger.info(f'Current test_data_index(0-99): {test_data_index}')
        # begin one test data
        data = test_set[test_data_index]
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model, data, config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms
        )
        # 将batch=100的数据进行切分
        # pred_pos      abs_pos pre_xt-1_1000[100, E, 3] 最后一轮的xt-1 100个ligand，E不同，对应同一个protein的结果
        # pred_v                pre_vt-1_1000[100, E, 1] 最后一轮的vt-1 100个ligand，E不同，对应同一个protein的结果
        # pred_pos_traj abs_pos ls:pre_xt-1[100, step, E, 3]  100个ligand，E不同，对应同一个protein的结果  每个ligand最后step的结果作为pre_xt-1_1000
        # pred_v_traj           ls:pre_vt-1[100, step, E, 1]  100个ligand，E不同，对应同一个protein的结果  每个ligand最后step的结果作为pre_vt-1_1000
        # pred_v0_traj          ls:pre_v0  [100, step, E, 13] 100个ligand，E不同，对应同一个protein的结果
        # pred_vt_traj          ls:pre_vt-1[100, step, E, 13] 100个ligand，E不同，对应同一个protein的结果  每个ligand最后step的结果作为pre_vt-1_1000，编码不同
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'time': time_list
        }
        logger.info('Sample done!')
        # Sample done! save result by test_data_index
        result_path = log_dir
        os.makedirs(result_path, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
        torch.save(result, os.path.join(result_path, f'result_{test_data_index}.pt'))
        # finish one test data

    end_time = time.time()
    running_time = end_time - begin_time
    logger.info(f'running_time: {running_time}')
