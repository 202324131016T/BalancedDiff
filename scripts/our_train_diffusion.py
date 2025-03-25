import argparse
import os, sys
import shutil
import time

# print(sys.path) # PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path) # PATH
# sys.exit()

import numpy as np
import torch
import torch.utils.tensorboard
# import tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH


# from models.our_model_50 import ScorePosNet3D as my_model
# 动态导入模型
import importlib
def import_model(model_path, model_type="ScorePosNet3D"):
    module = importlib.import_module(model_path)
    model_class = getattr(module, model_type)
    return model_class


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


def validate(it):
    # fix time steps
    sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0
    sum_loss_bond, sum_loss_non_bond = 0, 0
    all_pred_v, all_true_v = [], []
    all_pred_bond_type, all_gt_bond_type = [], []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_loader, desc='Validate'):
            batch = batch.to(args.device)
            batch_size = batch.num_graphs
            t_loss, t_loss_pos, t_loss_v = [], [], []
            for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                time_step = torch.tensor([t] * batch_size).to(args.device)
                results = model.get_diffusion_loss(
                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch.protein_element_batch,

                    ligand_pos=batch.ligand_pos,
                    ligand_v=batch.ligand_atom_feature_full,
                    batch_ligand=batch.ligand_element_batch,
                    time_step=time_step
                )
                loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']

                sum_loss += float(loss) * batch_size
                sum_loss_pos += float(loss_pos) * batch_size
                sum_loss_v += float(loss_v) * batch_size
                sum_n += batch_size
                all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

    avg_loss = sum_loss / sum_n
    avg_loss_pos = sum_loss_pos / sum_n
    avg_loss_v = sum_loss_v / sum_n
    atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                           feat_mode=config.data.transform.ligand_atom_mode)

    if config.train.scheduler.type == 'plateau':
        scheduler.step(avg_loss)
    elif config.train.scheduler.type == 'warmup_plateau':
        scheduler.step_ReduceLROnPlateau(avg_loss)
    else:
        scheduler.step()

    logger.info(
        '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Avg atom auroc %.6f' % (
            it, avg_loss, avg_loss_pos, avg_loss_v * 1000, atom_auroc
        )
    )
    writer.add_scalar('val/loss', avg_loss, it)
    writer.add_scalar('val/loss_pos', avg_loss_pos, it)
    writer.add_scalar('val/loss_v', avg_loss_v, it)
    writer.flush()
    return avg_loss


def train(it, args_model_type):
    model.train()
    optimizer.zero_grad() # 清空梯度
    for _ in range(config.train.n_acc_batch):

        batch = next(train_iterator).to(args.device)

        protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
        # 生成满足0, 1正态分布的随机数 * pos_noise_std = 0 0.01
        gt_protein_pos = batch.protein_pos + protein_noise
        # gt pos + noise

        if args_model_type:  # 1:use loss balance
            results = model.get_diffusion_loss(
                info_path=args.info_path,
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,

                pid=batch.protein_filename,
                lid=batch.ligand_filename
            )
        else:
            results = model.get_diffusion_loss(
                # info_path=args.info_path,
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,

                pid=batch.protein_filename,
                lid=batch.ligand_filename
            )

        loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
        loss = loss / config.train.n_acc_batch
        loss.backward() # 计算梯度
        # n_acc_batch 个梯度累计
    orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
    # 梯度裁剪 max_grad_norm为梯度的最大值，避免梯度爆炸
    optimizer.step() # 梯度下降

    if it % args.train_report_iter == 0:
        logger.info(
            '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                it, loss, loss_pos, loss_v, optimizer.param_groups[0]['lr'], orig_grad_norm
            )
        )
        for k, v in results.items():
            if torch.is_tensor(v) and v.squeeze().ndim == 0:
                writer.add_scalar(f'train/{k}', v, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/our_training.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--diffusion_logdir', type=str, default='../logs/')
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--info_path', type=str, default='../data/info.pkl')
    parser.add_argument('--model_idx', type=int, default=510)
    parser.add_argument('--model_type', type=int, default=1)
    args = parser.parse_args()

    # 动态导入模型
    model_path = "models.our_model_" + str(args.model_idx)
    my_model = import_model(model_path)

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # modify data_path
    config.data.path = args.data_path + 'crossdocked_v1.1_rmsd1.0_pocket10'
    config.data.split = args.data_path + 'crossdocked_pocket10_pose_split.pt'

    # Logging
    log_dir = misc.get_new_log_dir(args.diffusion_logdir, prefix=config_name, tag=args.tag)
    config.log_dir = log_dir
    config.model.log_dir = log_dir
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    # logger.info(config)
    for item in list(config):
        logger.info(f'\n config: {item}')
        for item2 in str(config[item]).replace('\'', '').split(','):
            logger.info(f'config: {item2}')

    print(sys.path) # PATH

    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    print(os.path.join(log_dir, 'models'))
    # shutil.copytree('../models', os.path.join(log_dir, 'models'))

    # copy config files
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    # copy model files
    path = os.path.abspath(sys.modules[my_model.__module__].__file__)
    model_path = path.split('/')[-1]
    shutil.copyfile(path, os.path.join(log_dir, model_path))
    # copy train_diffusion files
    train_diffusion_path = os.path.abspath(__file__).split('/')[-1]
    shutil.copyfile(os.path.abspath(__file__), os.path.join(log_dir, train_diffusion_path))

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform
    )
    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ) # 99990 / 4 = 24998 batch_size


    train_iterator = utils_train.inf_iterator(train_loader)

    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    model = my_model(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)

    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)


    try:
        best_loss, best_iter = None, None
        begin_time = time.time()
        for it in range(1, config.train.max_iters + 1): # 200,000
            # with torch.autograd.detect_anomaly():

            train(it, args_model_type=args.model_type)

            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                    # save as best.pt
                    ckpt_path = os.path.join(ckpt_dir, 'best.pt')
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
        end_time = time.time()
        running_time = end_time - begin_time
        logger.info(f'running_time: {running_time}')
    except KeyboardInterrupt:
        logger.info('Terminating...')
