import argparse
import os
import time
import sys

path = os.path.abspath(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)
sys.path.append(path)
# $PATH + /../targetdiff-main

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate_logdir', type=str, default='../logs/')
    parser.add_argument('--data_path', type=str, default='../data/test_set/') # not use
    parser.add_argument('--sample_path', type=str, default='../logs_sample/raw_sampling_20240622_092629/')
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='protein_path_for_testset')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, default='vina_dock', choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)  # vina 的参数，越大对接的结果越精细，效果可能越好，时间需要的更多
    args = parser.parse_args()

    begin_time = time.time()

    args.evaluate_logdir = misc.get_new_log_dir(args.evaluate_logdir, prefix='our_evaluate')
    os.makedirs(args.evaluate_logdir, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=args.evaluate_logdir)

    # modify protein_path_for_testset
    args.protein_root = args.data_path

    # modify sample_path to get all result_{test_data_index}.pt
    args.sample_path = args.sample_path + 'result_*.pt'

    for k, v in vars(args).items():
        logger.info(f'{k:<20}: {v}')

    logger.info('evaluate')

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    results_fn_list = glob(args.sample_path)
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    # sys.exit()
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list) # 100 ligand*protein_num
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3] [ligand_num=100, step=1000, E, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj'] # [ligand_num=100, step=1000, E, 1]
        num_samples += len(all_pred_ligand_pos)

        ligand_filename = r['data']['ligand_filename']
        protein_filename = r['data']['protein_filename']
        # for item in 100-ligand
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step] # 取最后一轮 第1000轮的结果
            # pred_pos, [E, 3]
            # pred_v    [E, 1]
            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode) # [E, 1]
            all_atom_types += Counter(pred_atom_type) # atom_type:atom_num
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)  # 检查稳定性 (bool, E, E)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            # 每两个atom之间的dist
            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            # reconstruction 根据 v x atom_type 重构ligand
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                if args.verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue
            except Exception as e:
                logger.warning('error: not reconstruct error, may be python error (e.g. %s)' % str(e))
                continue
            n_recon_success += 1

            if '.' in smiles:
                continue
            n_complete += 1

            # chemical and docking check
            try:
                chem_results = scoring_func.get_chem(mol)
                if args.docking_mode == 'qvina':
                    vina_task = QVinaDockingTask.from_generated_mol(
                        mol, r['data'].ligand_filename, protein_root=args.protein_root)
                    vina_results = vina_task.run_sync()
                elif args.docking_mode in ['vina_score', 'vina_dock']: # True
                    vina_task = VinaDockingTask.from_generated_mol(
                        mol, r['data'].ligand_filename, protein_root=args.protein_root)  # Mol:ligand
                    score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                    minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                    vina_results = {
                        'score_only': score_only_results,
                        'minimize': minimize_results
                    }
                    if args.docking_mode == 'vina_dock':
                        docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                        vina_results['dock'] = docking_results
                else:
                    vina_results = None

                n_eval_success += 1
            except Exception as e:
                print("\n except:", e)
                logger.info(f'\n except:{e}, protein_filename:{protein_filename}, ligand_filename:{ligand_filename}')
                if args.verbose:
                    logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')
                continue

            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)

            results.append({
                'mol': mol,
                'smiles': smiles,
                'ligand_filename': r['data'].ligand_filename,
                'protein_filename': r['data'].protein_filename,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
                'chem_results': chem_results,
                'vina': vina_results
            })
    logger.info(f'Evaluate done! {num_samples} samples in total.')

    fraction_mol_stable = all_mol_stable / num_samples
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / num_samples
    fraction_eval = n_eval_success / num_samples
    fraction_complete = n_complete / num_samples
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete
    }
    print_dict(validity_dict, logger)

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    # logger.info('JS bond distances of complete mols: ')
    # print_dict(c_bond_length_dict, logger)

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    print_dict(success_js_metrics, logger)

    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    logger.info('Atom type JS: %.4f' % atom_type_js)

    if args.save:
        eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                            metrics=success_js_metrics,
                                            save_path=os.path.join(args.evaluate_logdir, f'pair_dist_hist_{args.eval_step}.png'))

    logger.info('Number of reconstructed mols: %d' % n_recon_success)
    logger.info('complete mols: %d' % n_complete)
    logger.info('evaluated mols: %d' % len(results))

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    vina_score_only = None
    vina_min = None
    vina_dock = None
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    if args.docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    logger.info('%.2f%.2f%.2f%.2f%.2f%.2f%.2f%.2f%.2f%.2f' % (np.mean(vina_score_only), np.median(vina_score_only), np.mean(vina_min), np.median(vina_min), np.mean(vina_dock), np.median(vina_dock), np.mean(qed), np.median(qed), np.mean(sa), np.median(sa)))
    # check ring distribution
    # print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    if args.save:
        torch.save({
            'stability': validity_dict,
            'bond_length': all_bond_dist,
            'all_results': results
        }, os.path.join(args.evaluate_logdir, f'metrics_{args.eval_step}.pt'))

    end_time = time.time()
    running_time = end_time - begin_time
    logger.info(f'running_time: {running_time}')
