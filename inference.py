import copy
import collections
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import PeriodicTable
from rdkit.Chem.Lipinski import RotatableBondSmarts
import scipy
from scipy import spatial as sci_spatial
import torch
from tqdm.auto import tqdm
import seaborn as sns
from utils.evaluation import eval_bond_length, scoring_func, similarity

ptable = Chem.GetPeriodicTable()

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import sys, os
sys.path.insert(0, os.path.abspath('./'))


MODEL_NAME = 'Ours'


# Load Data
class Globals:
    reference_path = './result/results_sampling/crossdocked_test_vina_docked.pt'
    cvae_path = './result/results_sampling/CVAE_test_docked_sf1.5.pt'
    ar_path = './result/results_sampling/ar_vina_docked.pt'
    pocket2mol_path = './result/results_sampling/pocket2mol_vina_docked.pt'
    targetDiff_path = './result/results_sampling/targetdiff_vina_docked.pt'
    targetDiff_checkpoint_path = './result/results_sampling/targetDiff_checkpoint.pt'
    ours_path = './result/filter_mols/all_results.pt'
    # ours_path = './result/filter_mols/fliter_stars_5.pt'
    ours_5 = './result/0.5/all_results.pt'
    ours_6 = './result/0.6/all_results.pt'
    ours_7 = './result/0.7/all_results.pt'
    ours_8 = './result/0.8/all_results.pt'
    ours_9 = './result/0.9/all_results.pt'
    ours_7_32 = './result/0.7-all-32/all_results.pt'
    ours_7_64 = './result/0.7-all-64/all_results.pt'
    ours_7_128 = './result/0.7-all-128/all_results.pt'
    ours_7_fuse = './result/0.7-fuse/all_results.pt'
    ours_7_KAN = './result/0.7-KAN/all_results.pt'


Globals.reference_results = torch.load(Globals.reference_path)
Globals.reference_results = [[v] for v in Globals.reference_results]


# deal ours
def deal_ours(results):
    ligand_filename_ls = []
    for item in Globals.reference_results:
        lf = item[0]["ligand_filename"]
        ligand_filename_ls.append(lf)

    if type(results) is dict:
        results = results['all_results']

    ls = [[] for i in range(100)]
    for item in results:
        for idx in range(len(ligand_filename_ls)):
            ligand_filename = ligand_filename_ls[idx]
            if item["ligand_filename"] == ligand_filename:
                ls[idx].append(item)
    return ls


Globals.ar_results = torch.load(Globals.ar_path)
Globals.pocket2mol_results = torch.load(Globals.pocket2mol_path)
Globals.cvae_results = torch.load(Globals.cvae_path)
Globals.targetDiff_results = torch.load(Globals.targetDiff_path)
Globals.targetDiff_checkpoint_path = torch.load(Globals.targetDiff_checkpoint_path)
Globals.targetDiff_checkpoint_path = deal_ours(Globals.targetDiff_checkpoint_path)
Globals.ours_results = torch.load(Globals.ours_path)
Globals.ours_results = deal_ours(Globals.ours_results)


cnt = 0
for item in Globals.ours_results:
    cnt += len(item)
print(f'cnt={cnt}')


def get_new_metrics_value(vina_score_only, vina_minimize, vina_dock, qed, sa, diversity):
    # mapping to 0-1
    vina_score_only = min(vina_score_only, 0)  # find > 0; unreasonable mol
    vina_minimize = min(vina_minimize, 0)  # find > 0; unreasonable mol
    vina_dock = min(vina_dock, 0)  # find > 0; unreasonable mol
    VS = -(vina_score_only + vina_minimize + vina_dock) / 3
    WT = math.exp((qed + sa + diversity)/3)
    k = 0.25
    x = VS * WT
    temp = (math.exp(k * x) - 1) / (math.exp(k * x) + 1)
    return temp


def get_new_metrics(mols):
    new_metrics = []
    for idx in range(100):
        ref_mol = Globals.reference_results[idx][0]
        cur_mols = mols[idx]
        if len(cur_mols) == 0:
            continue
        if ref_mol['ligand_filename'] != cur_mols[0]['ligand_filename']:
            print("ERROR: ref_mol not match cur_mol")
            return
        for mol in cur_mols:
            vina_score_only = mol['vina']['score_only'][0]['affinity']
            vina_minimize = mol['vina']['minimize'][0]['affinity']
            vina_dock = mol['vina']['dock'][0]['affinity']
            qed = mol['chem_results']['qed']
            sa = mol['chem_results']['sa']
            diversity = similarity.tanimoto_distance(mol['mol'], ref_mol['mol'])
            # if qed <= 0 or sa <= 0 or diversity <= 0:
            #     print("Warning: find unreasonable mol")
            temp = get_new_metrics_value(vina_score_only=vina_score_only, vina_minimize=vina_minimize, vina_dock=vina_dock, qed=qed, sa=sa, diversity=diversity)
            new_metrics.append(temp)

    print(f'new Evaluation Metrics: Mean={np.mean(np.array(new_metrics)):.3f}, Median={np.median(np.array(new_metrics)):.3f}')


def get_new_metrics_best(mols):
    best_metrics = [0.0 for i in range(100)]
    for idx in range(100):
        new_metrics = []
        ref_mol = Globals.reference_results[idx][0]
        cur_mols = mols[idx]
        if len(cur_mols) == 0:
            continue
        if ref_mol['ligand_filename'] != cur_mols[0]['ligand_filename']:
            print("ERROR: ref_mol not match cur_mol")
            return
        for mol in cur_mols:
            vina_score_only = mol['vina']['score_only'][0]['affinity']
            vina_minimize = mol['vina']['minimize'][0]['affinity']
            vina_dock = mol['vina']['dock'][0]['affinity']
            qed = mol['chem_results']['qed']
            sa = mol['chem_results']['sa']
            diversity = similarity.tanimoto_distance(mol['mol'], ref_mol['mol'])
            # if qed <= 0 or sa <= 0 or diversity <= 0:
            #     print("Warning: find unreasonable mol")
            temp = get_new_metrics_value(vina_score_only=vina_score_only, vina_minimize=vina_minimize, vina_dock=vina_dock, qed=qed, sa=sa, diversity=diversity)
            new_metrics.append(temp)
        best_metrics[idx] = np.max(np.array(new_metrics))
    return best_metrics


def get_new_metrics_ref():
    new_metrics = []
    mols = Globals.reference_results
    for idx in range(100):
        cur_mols = mols[idx]
        if len(cur_mols) == 0:
            continue
        for mol in cur_mols:
            vina_score_only = mol['vina']['score_only'][0]['affinity']
            vina_minimize = mol['vina']['minimize'][0]['affinity']
            vina_dock = mol['vina']['dock'][0]['affinity']
            qed = mol['chem_results']['qed']
            sa = mol['chem_results']['sa']
            # if qed <= 0 or sa <= 0:
            #     print("Warning: find unreasonable mol")
            diversity = 1.0
            temp = get_new_metrics_value(vina_score_only=vina_score_only, vina_minimize=vina_minimize, vina_dock=vina_dock, qed=qed, sa=sa, diversity=diversity)
            new_metrics.append(temp)
    print(f'new Evaluation Metrics: Mean={np.mean(np.array(new_metrics)):.3f}, Median={np.median(np.array(new_metrics)):.3f}')
    return new_metrics


# fig new Evaluation Metrics
def fig_new_Evaluation_Metrics(save_path):
    # ref_MQS = get_new_metrics_ref()
    ours_MQS = get_new_metrics_best(Globals.ours_results)
    targetDiff_MQS = get_new_metrics_best(Globals.targetDiff_results)
    pocket2mol_MQS = get_new_metrics_best(Globals.pocket2mol_results)
    ar_MQS = get_new_metrics_best(Globals.ar_results)

    ours_MQS = np.where(np.array(ours_MQS) < 0.7, 0.7, np.array(ours_MQS))
    targetDiff_MQS = np.where(np.array(targetDiff_MQS) < 0.7, 0.7, np.array(targetDiff_MQS))
    pocket2mol_MQS = np.where(np.array(pocket2mol_MQS) < 0.7, 0.7, np.array(pocket2mol_MQS))
    ar_MQS = np.where(np.array(ar_MQS) < 0.7, 0.7, np.array(ar_MQS))

    # all_MQS = np.stack([ref_MQS, ours_MQS, targetDiff_MQS, pocket2mol_MQS, ar_MQS], axis=0)
    all_MQS = np.stack([ours_MQS, targetDiff_MQS, pocket2mol_MQS, ar_MQS], axis=0)
    best_MQS_idx = np.argmax(all_MQS, axis=0)

    plt.figure(figsize=(25, 8), dpi=100)

    ax = plt.subplot(1, 1, 1)
    ax.set_prop_cycle('color', plt.cm.Set1.colors)
    n_data = len(ours_MQS)
    # fig_idx = np.argsort(ours_MQS)
    ALPHA = 0.75
    POINT_SIZE = 100
    # plt.scatter(np.arange(n_data), ref_MQS, label=f'Ours (lowest in {np.mean(best_MQS_idx==0)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE)
    plt.scatter(np.arange(n_data), ours_MQS, label=f'Ours (best in {np.mean(best_MQS_idx==0)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE)
    plt.scatter(np.arange(n_data), targetDiff_MQS, label=f'targetDiff (best in {np.mean(best_MQS_idx==1)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), pocket2mol_MQS, label=f'Pocket2Mol (best in {np.mean(best_MQS_idx==2)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), ar_MQS, label=f'AR (best in {np.mean(best_MQS_idx==3)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)

    for i in range(n_data):
        if best_MQS_idx[i] == 0:
            plt.axvline(i, c='red', lw=0.2)
        else:
            plt.axvline(i, c='0.1', lw=0.2)
    plt.xlim(-1, 100)
    plt.ylim(0.695, 1.005)
    plt.yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
               ['≤0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '1.00'], fontsize=10)
    plt.ylabel('Molecule Quality Score', fontsize=20)
    plt.legend(fontsize=20, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.13), frameon=False)
    plt.xticks(np.arange(0, 101, 10), [f'target {v}' for v in np.arange(0, 101, 10)], fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    print('save path: ', save_path)
    # plt.show()


# fig new Evaluation Metrics
def fig_new_Evaluation_Metrics_sort(save_path):
    # ref_MQS = get_new_metrics_ref()
    ours_MQS = get_new_metrics_best(Globals.ours_results)
    targetDiff_MQS = get_new_metrics_best(Globals.targetDiff_results)
    pocket2mol_MQS = get_new_metrics_best(Globals.pocket2mol_results)
    ar_MQS = get_new_metrics_best(Globals.ar_results)

    ours_MQS = np.where(np.array(ours_MQS) < 0.7, 0.7, np.array(ours_MQS))
    targetDiff_MQS = np.where(np.array(targetDiff_MQS) < 0.7, 0.7, np.array(targetDiff_MQS))
    pocket2mol_MQS = np.where(np.array(pocket2mol_MQS) < 0.7, 0.7, np.array(pocket2mol_MQS))
    ar_MQS = np.where(np.array(ar_MQS) < 0.7, 0.7, np.array(ar_MQS))

    # all_MQS = np.stack([ref_MQS, ours_MQS, targetDiff_MQS, pocket2mol_MQS, ar_MQS], axis=0)
    all_MQS = np.stack([ours_MQS, targetDiff_MQS, pocket2mol_MQS, ar_MQS], axis=0)
    best_MQS_idx = np.argmax(all_MQS, axis=0)

    plt.figure(figsize=(25, 8), dpi=100)

    ax = plt.subplot(1, 1, 1)
    ax.set_prop_cycle('color', plt.cm.Set1.colors)
    n_data = len(ours_MQS)
    fig_idx = np.argsort(ours_MQS)
    ALPHA = 0.75
    POINT_SIZE = 100
    # plt.scatter(np.arange(n_data), ref_MQS, label=f'Ours (lowest in {np.mean(best_MQS_idx==0)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE)
    plt.scatter(np.arange(n_data), np.array(ours_MQS)[fig_idx], label=f'Ours (best in {np.mean(best_MQS_idx==0)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE)
    plt.scatter(np.arange(n_data), np.array(targetDiff_MQS)[fig_idx], label=f'TargetDiff (best in {np.mean(best_MQS_idx==1)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), np.array(pocket2mol_MQS)[fig_idx], label=f'Pocket2Mol (best in {np.mean(best_MQS_idx==2)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), np.array(ar_MQS)[fig_idx], label=f'AR (best in {np.mean(best_MQS_idx==3)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)

    for i in range(n_data):
        plt.axvline(i, c='0.1', lw=0.2)
        # if best_MQS_idx[fig_idx[i]] == 0:
        #     plt.axvline(i, c='red', lw=0.2)
        # else:
        #     plt.axvline(i, c='0.1', lw=0.2)
    plt.xlim(-1, 100)
    plt.ylim(0.695, 1.005)
    plt.yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
               ['≤0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '1.00'], fontsize=10)
    plt.ylabel('Molecule Quality Score', fontsize=20)
    plt.legend(fontsize=20, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.13), frameon=False)
    plt.xticks(np.arange(0, 101, 10), [f'target {v}' for v in np.arange(0, 101, 10)], fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    print('save path: ', save_path)
    # plt.show()


def get_Diversity(mols):
    mols_sim = []
    for idx in range(100):
        if len(mols[idx]) == 0:
            continue
        if Globals.reference_results[idx][0]['ligand_filename'] == mols[idx][0]['ligand_filename']:
            for item in mols[idx]:
                sim = similarity.tanimoto_distance(item['mol'], Globals.reference_results[idx][0]['mol'])
                mols_sim.append(sim)
        else:
            print("error")
            return
    print(f'Diversity: Mean={np.mean(np.array(mols_sim)):.2f}, Median={np.median(np.array(mols_sim)):.2f}')


# tab ring size
def tab_ring_size():
    def get_ring_size(ls):
        ring_size_count = {3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
        total_rings = 0
        for mol in ls:
            rings = Chem.GetSSSR(mol['mol'])
            for ring in rings:
                ring_size = len(ring)
                if ring_size in ring_size_count.keys():
                    ring_size_count[ring_size] += 1
                    total_rings += 1
                # else:
                    # ring_size_count[ring_size] = 1
                    # print("---------find new rings!---------")
        for size, count in ring_size_count.items():
            percentage = (count / total_rings) * 100
            print(f"Ring size {size}: {percentage:.1f}%")


    # rings
    targetDiffMols = [j for i in Globals.targetDiff_results for j in i]
    print("*******************  targetDiff_rings  *******************")
    get_ring_size(targetDiffMols)
    print("*******************  targetDiff_rings  *******************\n")

    ours_mols = [j for i in Globals.ours_results for j in i]
    print("*******************  ours_rings  *******************")
    get_ring_size(ours_mols)
    print("*******************  ours_rings  *******************\n")


def export_mols(mols, save_path):
    for idx in range(len(mols)):
        item = mols[idx]
        smi = item['smiles']
        mol = Chem.MolFromSmiles(smi)
        idx_str = f"{idx:04}"
        Chem.MolToMolFile(mol, save_path + "/export_mols/molecule_" + idx_str + '.mol')
    print(f'save_path: {save_path}')


def eval_by_qikpropservice(save_path):
    from qikpropservice import QikpropAsAService, QikPropOptions
    path = save_path + 'export_mols/'
    save_path = save_path + 'eval_by_qikpropservice/molecule_'
    for filename in os.listdir(path):
        mol_id = filename.split('_')[-1].split('.')[-2]
        mol_path = path + filename
        service = QikpropAsAService()
        options = QikPropOptions(fast=True, similar=30)
        success, ret_code, data = service.post_task(mol_path, options=options)
        success, ret_code, ret_data = service.get_result(task_id=data["id"], output_file=save_path + mol_id + "_result.tar.gz")
        if success:
            # print(f'success for mol: {mol_id}')
            None
        else:
            print(f'fail for {mol_id}')


def unzip_eval_result(save_path):
    import os
    import tarfile
    folder_a = save_path + "/eval_by_qikpropservice/"
    folder_b = save_path + "/unzip_eval_result/"

    for tar_gz_file in os.listdir(folder_a):
        if tar_gz_file.endswith(".tar.gz"):
            extract_folder = os.path.join(folder_b, tar_gz_file[-28:-7])
            os.makedirs(extract_folder, exist_ok=True)

            full_tar_gz_path = os.path.join(folder_a, tar_gz_file)
            try:
                with tarfile.open(full_tar_gz_path, "r:gz") as tar:
                    tar.extractall(path=extract_folder)
                # print(f"Unzipped {full_tar_gz_path} to {extract_folder}")
            except:
                print(f'************* ERROR: find unreasonable file {full_tar_gz_path}')


# filter
def filter_mol(mols, stars=1, save_pt=False, save_path="./result/filter_mols/"):
    import csv
    flag = [-1 for i in range(len(mols))]

    for idx in range(len(flag)):
        idx_str = f"{idx:04}"
        file_path = f'{save_path}/unzip_eval_result/molecule_{idx_str}_result/QP.CSV'
        if not os.path.exists(file_path):
            continue
        csv_data = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            for row in csv.reader(file):
                csv_data.append(row)
        if 'fail' in csv_data[1][0]:
            continue
        flag[idx] = int(csv_data[1][1])  # stars

    filter_mols = []
    for idx in range(len(flag)):
        if flag[idx] == -1:
            continue  # ERROR: unreasonable mol
        if flag[idx] > stars:
            continue
        filter_mols.append(mols[idx])
    print(f'filter out mols count: {len(mols)-len(filter_mols)}')
    print(f'save mols count: {len(filter_mols)}')
    print(f'total mols count: {len(mols)}')

    if save_pt:
        torch.save(filter_mols, f'{save_path}/fliter_stars_{stars}.pt')
        print(f'save path: {save_path}/fliter_stars_{stars}.pt')
    return filter_mols


def compute_high_affinity(vina_ref, results):
    percentage_good = []
    num_docked = []
    qed_good, sa_good = [], []
    for i in range(100):
        score_ref = vina_ref[i]
        pocket_results = [r for r in results[i] if r['vina'] is not None]
        if len(pocket_results) < 50:
            continue
        num_docked.append(len(pocket_results))

        scores_gen = []
        for docked in pocket_results:
            aff = docked['vina']['dock'][0]['affinity']
            scores_gen.append(aff)
            if aff <= score_ref:
                qed_good.append(docked['chem_results']['qed'])
                sa_good.append(docked['chem_results']['sa'])
        scores_gen = np.array(scores_gen)
        percentage_good.append((scores_gen <= score_ref).mean())

    percentage_good = np.array(percentage_good)
    num_docked = np.array(num_docked)

    print('[HF%%]  Avg: %.3f%% | Med: %.3f%% ' % (np.mean(percentage_good) * 100, np.median(percentage_good) * 100))
    print('[HF-QED]  Avg: %.4f | Med: %.4f ' % (np.mean(qed_good) * 100, np.median(qed_good) * 100))
    print('[HF-SA]   Avg: %.4f | Med: %.4f ' % (np.mean(sa_good) * 100, np.median(sa_good) * 100))
    print('[Success%%] %.3f%% ' % (np.mean(percentage_good > 0) * 100,))


def print_results(results, show_vina=True):
    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    mol_size = [r['mol'].GetNumAtoms() for r in results]
    print('Num results: %d' % len(results))
    if show_vina:
        vina_score_only = [x['vina']['score_only'][0]['affinity'] for x in results]
        vina_min = [x['vina']['minimize'][0]['affinity'] for x in results]
        vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
        print('[Vina Score] Avg: %.3f | Med: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        print('[Vina Min]   Avg: %.3f | Med: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        print('[Vina Dock]  Avg: %.4f | Med: %.4f' % (np.mean(vina_dock), np.median(vina_dock)))

    print('[QED]  Avg: %.4f | Med: %.4f' % (np.mean(qed), np.median(qed)))
    print('[SA]   Avg: %.4f | Med: %.4f' % (np.mean(sa), np.median(sa)))
    print('[Size] Avg: %.4f | Med: %.4f' % (np.mean(mol_size), np.median(mol_size)))


# tab Metrics Summary
def tab_Metrics_Summary(ablation_threshold, ablation_network, ablation_filter):
    # Reference
    print("*******************  Reference  *******************")
    flat_ref_docked = [r for pr in Globals.reference_results for r in pr]
    print_results(flat_ref_docked)
    get_new_metrics_ref()
    print("*******************  Reference  *******************\n")

    vina_ref = [r['vina']['dock'][0]['affinity'] for r in flat_ref_docked]

    # AR
    print("*******************  AR  *******************")
    flat_ar_docked = [r for pr in Globals.ar_results for r in pr]
    print_results(flat_ar_docked)
    compute_high_affinity(vina_ref, Globals.ar_results)
    get_new_metrics(Globals.ar_results)
    print("*******************  AR  *******************\n")

    # Pocket2Mol
    print("*******************  Pocket2Mol  *******************")
    flat_p2m_docked = [r for pr in Globals.pocket2mol_results for r in pr]
    print_results(flat_p2m_docked)
    compute_high_affinity(vina_ref, Globals.pocket2mol_results)
    get_new_metrics(Globals.pocket2mol_results)
    print("*******************  Pocket2Mol  *******************\n")

    # TargetDiff
    print("*******************  TargetDiff  *******************")
    flat_targetDiff_docked = [r for pr in Globals.targetDiff_results for r in pr]
    print_results(flat_targetDiff_docked)
    compute_high_affinity(vina_ref, Globals.targetDiff_results)
    get_new_metrics(Globals.targetDiff_results)
    print("*******************  TargetDiff  *******************\n")

    # TargetDiff checkpoint
    print("*******************  TargetDiff checkpoint  *******************")
    flat_targetDiff_checkpoint_docked = [r for pr in Globals.targetDiff_checkpoint_path for r in pr]
    print_results(flat_targetDiff_checkpoint_docked)
    compute_high_affinity(vina_ref, Globals.targetDiff_checkpoint_path)
    get_new_metrics(Globals.targetDiff_checkpoint_path)
    print("*******************  TargetDiff checkpoint  *******************\n")

    # Ours
    print("*******************  Ours  *******************")
    flat_ours_docked = [r for pr in Globals.ours_results for r in pr]
    print_results(flat_ours_docked)
    compute_high_affinity(vina_ref, Globals.ours_results)
    get_Diversity(Globals.ours_results)
    get_new_metrics(Globals.ours_results)
    print("*******************  Ours  *******************\n")

    if ablation_threshold:
        # Ours_5
        print("*******************  Ours_5  *******************")
        temp = Globals.ours_5
        flat_temp_docked = [r for pr in temp for r in pr]
        print_results(flat_temp_docked)
        compute_high_affinity(vina_ref, temp)
        get_Diversity(Globals.ours_5)
        get_new_metrics(Globals.ours_5)
        print("*******************  Ours_5  *******************\n")

        # ours_6
        print("*******************  ours_6  *******************")
        temp = Globals.ours_6
        flat_temp_docked = [r for pr in temp for r in pr]
        print_results(flat_temp_docked)
        compute_high_affinity(vina_ref, temp)
        get_Diversity(Globals.ours_6)
        get_new_metrics(Globals.ours_6)
        print("*******************  ours_6  *******************\n")

        # ours_7
        print("*******************  ours_7  *******************")
        temp = Globals.ours_7
        flat_temp_docked = [r for pr in temp for r in pr]
        print_results(flat_temp_docked)
        compute_high_affinity(vina_ref, temp)
        get_Diversity(Globals.ours_7)
        get_new_metrics(Globals.ours_7)
        print("*******************  ours_7  *******************\n")

        # ours_8
        print("*******************  ours_8  *******************")
        temp = Globals.ours_8
        flat_temp_docked = [r for pr in temp for r in pr]
        print_results(flat_temp_docked)
        compute_high_affinity(vina_ref, temp)
        get_Diversity(Globals.ours_8)
        get_new_metrics(Globals.ours_8)
        print("*******************  ours_8  *******************\n")

        # ours_9
        print("*******************  ours_9  *******************")
        temp = Globals.ours_9
        flat_temp_docked = [r for pr in temp for r in pr]
        print_results(flat_temp_docked)
        compute_high_affinity(vina_ref, temp)
        get_Diversity(Globals.ours_9)
        get_new_metrics(Globals.ours_9)
        print("*******************  ours_9  *******************\n")

    if ablation_network:
        # ours_7_32
        print("*******************  ours_7_32  *******************")
        temp = Globals.ours_7_32
        flat_temp_docked = [r for pr in temp for r in pr]
        print_results(flat_temp_docked)
        compute_high_affinity(vina_ref, temp)
        get_Diversity(Globals.ours_7_32)
        get_new_metrics(Globals.ours_7_32)
        print("*******************  ours_7_32  *******************\n")

        # ours_7_64
        print("*******************  ours_7_64  *******************")
        temp = Globals.ours_7_64
        flat_temp_docked = [r for pr in temp for r in pr]
        print_results(flat_temp_docked)
        compute_high_affinity(vina_ref, temp)
        get_Diversity(Globals.ours_7_64)
        get_new_metrics(Globals.ours_7_64)
        print("*******************  ours_7_64  *******************\n")

        # ours_7_128
        print("*******************  ours_7_128  *******************")
        temp = Globals.ours_7_128
        flat_temp_docked = [r for pr in temp for r in pr]
        print_results(flat_temp_docked)
        compute_high_affinity(vina_ref, temp)
        get_Diversity(Globals.ours_7_128)
        get_new_metrics(Globals.ours_7_128)
        print("*******************  ours_7_128  *******************\n")

        # ours_7_fuse
        print("*******************  ours_7_fuse  *******************")
        temp = Globals.ours_7_fuse
        flat_temp_docked = [r for pr in temp for r in pr]
        print_results(flat_temp_docked)
        compute_high_affinity(vina_ref, temp)
        get_Diversity(Globals.ours_7_fuse)
        get_new_metrics(Globals.ours_7_fuse)
        print("*******************  ours_7_fuse  *******************\n")

        # ours_7_KAN
        print("*******************  ours_7_KAN  *******************")
        temp = Globals.ours_7_KAN
        flat_temp_docked = [r for pr in temp for r in pr]
        print_results(flat_temp_docked)
        compute_high_affinity(vina_ref, temp)
        get_Diversity(Globals.ours_7_fuse)
        get_new_metrics(Globals.ours_7_fuse)
        print("*******************  ours_7_KAN  *******************\n")

    if ablation_filter:
        for idx in range(0, 10):
            Globals.fliter_stars = torch.load(f"./result/filter_mols/fliter_stars_{idx}.pt")
            print(f"*******************  fliter_stars {idx} *******************")
            print_results(Globals.fliter_stars)
            vina_ref = [r['vina']['dock'][0]['affinity'] for pr in Globals.reference_results for r in pr]
            Globals.fliter_stars = deal_ours(Globals.fliter_stars)
            compute_high_affinity(vina_ref, Globals.fliter_stars)
            get_Diversity(Globals.fliter_stars)
            get_new_metrics(Globals.fliter_stars)
            print(f"*******************  fliter_stars {idx} *******************\n")


# fig distances
def fig_distances(save_path):
    # Atom Distance
    def get_all_atom_distance(results):
        atom_distance_list = []
        for pocket in results:
            for ligand in pocket:
                mol = ligand['mol']
                mol = Chem.RemoveAllHs(mol)
                pos = mol.GetConformers()[0].GetPositions()
                dist = sci_spatial.distance.pdist(pos, metric='euclidean')
                atom_distance_list += dist.tolist()
        return np.array(atom_distance_list)

    Globals.reference_atom_dist = get_all_atom_distance(Globals.reference_results)
    Globals.targetDiff_atom_dist = get_all_atom_distance(Globals.targetDiff_results)
    Globals.ar_atom_dist = get_all_atom_distance(Globals.ar_results)
    Globals.pocket2mol_atom_dist = get_all_atom_distance(Globals.pocket2mol_results)
    Globals.cvae_atom_dist = get_all_atom_distance(Globals.cvae_results)
    Globals.ours_atom_dist = get_all_atom_distance(Globals.ours_results)

    def get_all_c_c_distance(results):
        c_c_distance_list = []
        for pocket in results:
            for ligand in pocket:
                mol = ligand['mol']
                mol = Chem.RemoveAllHs(mol)
                for bond_type, dist in eval_bond_length.bond_distance_from_mol(mol):
                    if bond_type[:2] == (6, 6):
                        c_c_distance_list.append(dist)
        return np.array(c_c_distance_list)

    Globals.reference_c_c_dist = get_all_c_c_distance(Globals.reference_results)
    Globals.targetDiff_c_c_dist = get_all_c_c_distance(Globals.targetDiff_results)
    Globals.ar_c_c_dist = get_all_c_c_distance(Globals.ar_results)
    Globals.pocket2mol_c_c_dist = get_all_c_c_distance(Globals.pocket2mol_results)
    Globals.cvae_c_c_dist = get_all_c_c_distance(Globals.cvae_results)
    Globals.ours_c_c_dist = get_all_c_c_distance(Globals.ours_results)

    plt.figure(figsize=(25, 11))

    LW = 2
    LABEL_FONTSIZE = 18
    ALPHA = 0.75

    BINS = np.linspace(0, 12, 100)

    def _plot_other(plot_ylabel=False):
        if plot_ylabel:
            plt.ylabel('Density', fontsize=LABEL_FONTSIZE)
        plt.xlabel('Distance of all atom pairs ($\AA$)', fontsize=LABEL_FONTSIZE)
        plt.ylim(0, 0.5)
        plt.xlim(0, 12)

    reference_profile = eval_bond_length.get_distribution(Globals.reference_atom_dist, bins=BINS)

    def _compute_jsd(atom_dist_list):
        profile = eval_bond_length.get_distribution(atom_dist_list, bins=BINS)
        return sci_spatial.distance.jensenshannon(reference_profile, profile)

    ax = plt.subplot(2, 5, 1)
    plt.hist(Globals.reference_atom_dist, bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label='reference')
    plt.hist(Globals.cvae_atom_dist, bins=BINS, histtype='step', density=True, lw=LW, color='blue', alpha=ALPHA, label='liGAN')
    jsd = _compute_jsd(Globals.cvae_atom_dist)
    ax.text(5, 0.4, f'liGAN JSD: {jsd:.3f}', fontsize=15, weight='bold')
    # plt.title(f'liGAN (JSD={jsd:.3f})')
    _plot_other(plot_ylabel=True)

    ax = plt.subplot(2, 5, 2)
    plt.hist(Globals.reference_atom_dist, bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label='reference')
    plt.hist(Globals.ar_atom_dist, bins=BINS, histtype='step', density=True, lw=LW, color='red', alpha=ALPHA, label='AR')
    jsd = _compute_jsd(Globals.ar_atom_dist)
    ax.text(5, 0.4, f'AR JSD: {jsd:.3f}', fontsize=15, weight='bold')
    # plt.title(f'AR (JSD={jsd:.3f})')
    _plot_other()

    ax = plt.subplot(2, 5, 3)
    plt.hist(Globals.reference_atom_dist, bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label='reference')
    plt.hist(Globals.pocket2mol_atom_dist, bins=BINS, histtype='step', density=True, lw=LW, color='orange', alpha=ALPHA, label='Pocket2Mol')
    jsd = _compute_jsd(Globals.pocket2mol_atom_dist)
    ax.text(3.5, 0.4, f'Pocket2Mol JSD: {jsd:.3f}', fontsize=15, weight='bold')
    # plt.title(f'Pocket2Mol (JSD={jsd:.3f})')
    _plot_other()

    ax = plt.subplot(2, 5, 4)
    plt.hist(Globals.reference_atom_dist, bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label='reference')
    plt.hist(Globals.targetDiff_atom_dist, bins=BINS, histtype='step', density=True, lw=LW, color='green', alpha=ALPHA, label='TargetDiff')
    jsd = _compute_jsd(Globals.targetDiff_atom_dist)
    ax.text(3.5, 0.4, f'targetDiff JSD:{jsd:.3f}', fontsize=15, weight='bold')
    # plt.title(f'targetDiff (JSD={jsd:.3f})')
    _plot_other()

    ax = plt.subplot(2, 5, 5)
    plt.hist(Globals.reference_atom_dist, bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label='reference')
    plt.hist(Globals.targetDiff_atom_dist, bins=BINS, histtype='step', density=True, lw=LW, color='purple', alpha=ALPHA, label='Ours')
    jsd = _compute_jsd(Globals.ours_atom_dist)
    ax.text(5, 0.4, f'Ours JSD:{jsd:.3f}', fontsize=15, weight='bold')
    # plt.title(f'Ours (JSD={jsd:.3f})')
    _plot_other()

    BINS = np.linspace(0, 2, 100)

    def _select_proximal(arr):
        arr = np.array()

    def _plot_other(plot_ylabel=False):
        if plot_ylabel:
            plt.ylabel('Density', fontsize=LABEL_FONTSIZE)
        plt.xlabel('Distance of carbon carbon bond ($\AA$)', fontsize=LABEL_FONTSIZE)
        plt.ylim(0, 12)
        plt.xlim(0, 2)

    reference_profile = eval_bond_length.get_distribution(Globals.reference_c_c_dist, bins=BINS)

    def _compute_jsd(atom_dist_list):
        profile = eval_bond_length.get_distribution(atom_dist_list, bins=BINS)
        return sci_spatial.distance.jensenshannon(reference_profile, profile)

    ax = plt.subplot(2, 5, 6)
    plt.hist(Globals.reference_c_c_dist, bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label='reference')
    plt.hist(Globals.cvae_c_c_dist, bins=BINS, histtype='step', density=True, lw=LW, color='blue', alpha=ALPHA, label='liGAN')
    jsd = _compute_jsd(Globals.cvae_c_c_dist)
    ax.text(0.1, 10, f'liGAN JSD: {jsd:.3f}', fontsize=15, weight='bold')
    # plt.title(f'liGAN (JSD={jsd:.3f})')
    _plot_other(plot_ylabel=True)

    ax = plt.subplot(2, 5, 7)
    plt.hist(Globals.reference_c_c_dist, bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label='reference')
    plt.hist(Globals.ar_c_c_dist, bins=BINS, histtype='step', density=True, lw=LW, color='red', alpha=ALPHA, label='AR')
    jsd = _compute_jsd(Globals.ar_c_c_dist)
    ax.text(0.1, 10, f'AR JSD: {jsd:.3f}', fontsize=15, weight='bold')
    # plt.title(f'AR (JSD={jsd:.3f})')
    _plot_other(plot_ylabel=True)

    ax = plt.subplot(2, 5, 8)
    plt.hist(Globals.reference_c_c_dist, bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label='reference')
    plt.hist(Globals.pocket2mol_c_c_dist, bins=BINS, histtype='step', density=True, lw=LW, color='orange', alpha=ALPHA, label='Pocket2Mol')
    jsd = _compute_jsd(Globals.pocket2mol_c_c_dist)
    ax.text(0.1, 10, f'Pocket2Mol JSD: {jsd:.3f}', fontsize=15, weight='bold')
    # plt.title(f'Pocket2Mol (JSD={jsd:.3f})')
    _plot_other(plot_ylabel=True)

    ax = plt.subplot(2, 5, 9)
    plt.hist(Globals.reference_c_c_dist, bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label='reference')
    plt.hist(Globals.targetDiff_c_c_dist, bins=BINS, histtype='step', density=True, lw=LW, color='green', alpha=ALPHA, label="targetDiff")
    jsd = _compute_jsd(Globals.targetDiff_c_c_dist)
    ax.text(0.1, 10, f'targetDiff JSD: {jsd:.3f}', fontsize=15, weight='bold')
    # plt.title(f'targetDiff (JSD={jsd:.3f})')
    _plot_other()

    ax = plt.subplot(2, 5, 10)
    plt.hist(Globals.reference_c_c_dist, bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label='reference')
    plt.hist(Globals.targetDiff_c_c_dist, bins=BINS, histtype='step', density=True, lw=LW, color='purple', alpha=ALPHA, label="Ours")
    jsd = _compute_jsd(Globals.ours_c_c_dist)
    ax.text(0.1, 10, f'Ours JSD: {jsd:.3f}', fontsize=15, weight='bold')
    # plt.title(f'Ours (JSD={jsd:.3f})')
    _plot_other()

    plt.tight_layout()
    plt.savefig(save_path)
    print('save path: ', save_path)
    # plt.show()


# tab bond
def tab_bond():
    if False:
        mols = Globals.reference_results
        # 分析每个分子中的化学键
        bond_types = []

        for mol in mols:
            mol = mol[0]['mol']
            for bond in mol.GetBonds():
                atom1_type = bond.GetBeginAtom().GetAtomicNum()
                atom2_type = bond.GetEndAtom().GetAtomicNum()
                sorted_atom_types = tuple(sorted([atom1_type, atom2_type]))
                bond_order = int(bond.GetBondTypeAsDouble())
                bond_types.append((sorted_atom_types[0], sorted_atom_types[1], bond_order))

        # 计算每种化学键的数量
        from collections import Counter
        bond_counts = Counter(bond_types)

        # 输出数量最多的五种化学键类型
        most_common_bonds = bond_counts.most_common(8)

    # Bond
    def get_bond_length_profile(results):
        bond_distances = []
        for pocket in results:
            for ligand in pocket:
                mol = ligand['mol']
                mol = Chem.RemoveAllHs(mol)
                bond_distances += eval_bond_length.bond_distance_from_mol(mol)
        return eval_bond_length.get_bond_length_profile(bond_distances)

    REPORT_TYPE = (
        (6, 6, 1),
        (6, 6, 2),
        (6, 6, 4),
        (6, 7, 1),
        (6, 7, 2),
        (6, 7, 4),
        (6, 8, 1),
        (6, 8, 2),
        (6, 8, 4),
    )

    # REPORT_TYPE = (
    #     (6, 6, 1),
    #     (6, 7, 1),
    #     (6, 8, 1),
    #     (8, 15, 1),
    #     (6, 8, 2),
    #     (8, 15, 2),
    #     (6, 6, 2),
    #     (6, 16, 1),
    # )

    def _bond_type_str(bond_type) -> str:
        atom1, atom2, bond_category = bond_type
        return f'{atom1}-{atom2}|{bond_category}'

    def eval_bond_length_profile(model_profile):
        metrics = {}

        for bond_type in REPORT_TYPE:
            metrics[f'JSD_{_bond_type_str(bond_type)}'] = sci_spatial.distance.jensenshannon(Globals.reference_bond_length_profile[bond_type],
                                                                                             model_profile[bond_type])
        return metrics

    Globals.reference_bond_length_profile = get_bond_length_profile(Globals.reference_results)
    Globals.targetDiff_bond_length_profile = get_bond_length_profile(Globals.targetDiff_results)
    Globals.ar_bond_length_profile = get_bond_length_profile(Globals.ar_results)
    Globals.pocket2mol_bond_length_profile = get_bond_length_profile(Globals.pocket2mol_results)
    Globals.cvae_bond_length_profile = get_bond_length_profile(Globals.cvae_results)
    Globals.ours_bond_length_profile = get_bond_length_profile(Globals.ours_results)

    print("*******************  targetDiff_bond_length_profile  *******************")
    metrics = eval_bond_length_profile(Globals.targetDiff_bond_length_profile)
    for item in metrics:
        print(f'{item}: {metrics[item]:.3f}')
    print("*******************  targetDiff_bond_length_profile  *******************\n")

    print("*******************  AR_bond_length_profile  *******************")
    metrics = eval_bond_length_profile(Globals.ar_bond_length_profile)
    for item in metrics:
        print(f'{item}: {metrics[item]:.3f}')
    print("*******************  AR_bond_length_profile  *******************\n")

    print("*******************  pocket2mol_bond_length_profile  *******************")
    metrics = eval_bond_length_profile(Globals.pocket2mol_bond_length_profile)
    for item in metrics:
        print(f'{item}: {metrics[item]:.3f}')
    print("*******************  pocket2mol_bond_length_profile  *******************\n")

    print("*******************  liGAN: cvae_bond_length_profile  *******************")
    metrics = eval_bond_length_profile(Globals.cvae_bond_length_profile)
    for item in metrics:
        print(f'{item}: {metrics[item]:.3f}')
    print("*******************  liGAN: cvae_bond_length_profile  *******************\n")

    print("*******************  ours_bond_length_profile  *******************")
    metrics = eval_bond_length_profile(Globals.ours_bond_length_profile)
    for item in metrics:
        print(f'{item}: {metrics[item]:.3f}')
    print("*******************  ours_bond_length_profile  *******************\n")


# fig RMSD
def fig_RMSD(save_path):
    # Rigid
    import networkx as nx
    from rdkit.Chem.rdchem import BondType
    from copy import deepcopy
    from collections import OrderedDict

    class RotBondFragmentizer():
        def __init__(self, only_single_bond=True):
            self.type = 'RotBondFragmentizer'
            self.only_single_bond = only_single_bond

        # code adapt from Torsion Diffusion
        def get_bonds(self, mol):
            bonds = []
            G = nx.Graph()
            for i, atom in enumerate(mol.GetAtoms()):
                G.add_node(i)
            # nodes = set(G.nodes())
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                G.add_edge(start, end)
            for e in G.edges():
                G2 = copy.deepcopy(G)
                G2.remove_edge(*e)
                if nx.is_connected(G2): continue
                l = list(sorted(nx.connected_components(G2), key=len)[0])
                if len(l) < 2: continue
                # n0 = list(G2.neighbors(e[0]))
                # n1 = list(G2.neighbors(e[1]))
                if self.only_single_bond:
                    bond_type = mol.GetBondBetweenAtoms(e[0], e[1]).GetBondType()
                    if bond_type != BondType.SINGLE:
                        continue
                bonds.append((e[0], e[1]))
            return bonds

        def fragmentize(self, mol, dummyStart=1, bond_list=None):
            if bond_list is None:
                # get bonds need to be break
                bonds = self.get_bonds(mol)
            else:
                bonds = bond_list
            # whether the molecule can really be break
            if len(bonds) != 0:
                bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
                bond_ids = list(set(bond_ids))
                # break the bonds & set the dummy labels for the bonds
                dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
                break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
                dummyEnd = dummyStart + len(dummyLabels) - 1
            else:
                break_mol = mol
                bond_ids = []
                dummyEnd = dummyStart - 1

            return break_mol, bond_ids, dummyEnd

    def get_clean_mol(mol):
        rdmol = deepcopy(mol)
        for at in rdmol.GetAtoms():
            at.SetAtomMapNum(0)
            at.SetIsotope(0)
        Chem.RemoveStereochemistry(rdmol)
        return rdmol

    def replace_atom_in_mol(ori_mol, src_atom, dst_atom):
        mol = deepcopy(ori_mol)
        m_mol = Chem.RWMol(mol)
        for atom in m_mol.GetAtoms():
            if atom.GetAtomicNum() == src_atom:
                atom_idx = atom.GetIdx()
                m_mol.ReplaceAtom(atom_idx, Chem.Atom(dst_atom))
        return m_mol.GetMol()

    def ff_optimize(ori_mol, addHs=False, enable_torsion=False):
        mol = deepcopy(ori_mol)
        Chem.GetSymmSSSR(mol)
        if addHs:
            mol = Chem.AddHs(mol, addCoords=True)
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        if mp is None:
            return (None,)

        # turn off angle-related terms
        mp.SetMMFFOopTerm(enable_torsion)
        mp.SetMMFFAngleTerm(True)
        mp.SetMMFFTorsionTerm(enable_torsion)

        # optimize unrelated to angles
        mp.SetMMFFStretchBendTerm(True)
        mp.SetMMFFBondTerm(True)
        mp.SetMMFFVdWTerm(True)
        mp.SetMMFFEleTerm(True)

        #     try:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        energy_before_ff = ff.CalcEnergy()
        ff.Minimize()
        energy_after_ff = ff.CalcEnergy()
        # print(f'Energy: {energy_before_ff} --> {energy_after_ff}')
        energy_change = energy_before_ff - energy_after_ff
        Chem.SanitizeMol(ori_mol)
        Chem.SanitizeMol(mol)
        rmsd = rdMolAlign.GetBestRMS(ori_mol, mol)
        #     except:
        #         return (None, )
        return energy_change, rmsd, mol

    def frag_analysis_from_mol_list(input_mol_list):
        all_frags_dict = {}
        sg = RotBondFragmentizer()
        for mol in tqdm(input_mol_list):
            frags, _, _ = sg.fragmentize(mol)
            frags = [get_clean_mol(f) for f in Chem.GetMolFrags(frags, asMols=True)]

            for frag in frags:
                num_atoms = frag.GetNumAtoms() - Chem.MolToSmiles(frag).count('*')
                if 2 < num_atoms < 10:
                    if num_atoms not in all_frags_dict:
                        all_frags_dict[num_atoms] = []

                    mol = deepcopy(frag)
                    mol_hs = replace_atom_in_mol(mol, src_atom=0, dst_atom=1)
                    mol_hs = Chem.RemoveAllHs(mol_hs)
                    all_frags_dict[num_atoms].append(mol_hs)

        all_frags_dict = OrderedDict(sorted(all_frags_dict.items()))
        all_rmsd_by_frag_size = {}
        for k, mol_list in all_frags_dict.items():
            n_fail = 0
            all_energy_diff, all_rmsd = [], []
            for mol in mol_list:
                ff_results = ff_optimize(mol, addHs=True, enable_torsion=False)
                if ff_results[0] is None:
                    n_fail += 1
                    continue
                energy_diff, rmsd, _, = ff_results
                all_energy_diff.append(energy_diff)
                all_rmsd.append(rmsd)
            print(f'Num of atoms: {k} ({n_fail} of {len(mol_list)} fail):   '
                  f'\tEnergy {np.mean(all_energy_diff):.2f} / {np.median(all_energy_diff):.2f}'
                  f'\tRMSD   {np.mean(all_rmsd):.2f} / {np.median(all_rmsd):.2f}'
                  )
            all_rmsd_by_frag_size[k] = all_rmsd
        return all_frags_dict, all_rmsd_by_frag_size

    def construct_df(rigid_dict):
        df = []
        for k, all_v in rigid_dict.items():
            for v in all_v:
                df.append({'f_size': k, 'rmsd': v})
        return pd.DataFrame(df)

    targetdiff_mols = [r['mol'] for pr in Globals.targetDiff_results for r in pr]
    ar_mols = [r['mol'] for pr in Globals.ar_results for r in pr]
    pocket2mol_mols = [r['mol'] for pr in Globals.pocket2mol_results for r in pr]
    cvae_mols = [r['mol'] for pr in Globals.cvae_results for r in pr]
    ours_mols = [r['mol'] for pr in Globals.ours_results for r in pr]

    print("*******************  targetdiff_mols  *******************")
    _, targetDiff_rigid_rmsd = frag_analysis_from_mol_list(targetdiff_mols)
    print("*******************  targetdiff_mols  *******************\n")

    print("*******************  AR_mols  *******************")
    _, ar_rigid_rmsd = frag_analysis_from_mol_list(ar_mols)
    print("*******************  AR_mols  *******************\n")

    print("*******************  pocket2mol_mols  *******************")
    _, pocket2mol_rigid_rmsd = frag_analysis_from_mol_list(pocket2mol_mols)
    print("*******************  pocket2mol_mols  *******************\n")

    print("*******************  liGAN:cvae_mols  *******************")
    _, cvae_rigid_rmsd = frag_analysis_from_mol_list(cvae_mols)
    print("*******************  liGAN:cvae_mols  *******************\n")

    print("*******************  ours_mols  *******************")
    _, ours_rigid_rmsd = frag_analysis_from_mol_list(ours_mols)
    print("*******************  ours_mols  *******************\n")

    # sns.set(style="darkgrid")
    sns.set_style("white")
    sns.set_palette("muted")

    tmp_1 = construct_df(targetDiff_rigid_rmsd)
    tmp_1['model'] = 'TargetDiff'
    tmp_2 = construct_df(ar_rigid_rmsd)
    tmp_2['model'] = 'AR'
    tmp_3 = construct_df(pocket2mol_rigid_rmsd)
    tmp_3['model'] = 'Pocket2Mol'
    tmp_4 = construct_df(cvae_rigid_rmsd)
    tmp_4['model'] = 'liGAN'
    tmp_5 = construct_df(ours_rigid_rmsd)
    tmp_5['model'] = 'Ours'

    viz_df = pd.concat([tmp_1, tmp_2, tmp_3, tmp_4, tmp_5]).reset_index()
    viz_df = viz_df.query('3<=f_size<=9')

    LABEL_FONTSIZE = 24
    TICK_FONTSIZE = 16
    LEGEND_FONTSIZE = 20
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='f_size', y='rmsd', hue='model', data=viz_df, hue_order=('liGAN', 'AR', 'Pocket2Mol', 'TargetDiff', 'Ours'), showfliers=False)
    plt.xlabel('Fragment Size', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Median RMSD ($\AA{}$)', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    plt.savefig(save_path)
    print('save path: ', save_path)
    # plt.show()


# Vina Score Median dock
def fig_Vina_Score_Median_Dock(save_path):
    ours_vina = np.array([np.median([v['vina']['dock'][0]['affinity'] for v in pocket]) for pocket in Globals.ours_results])
    targetDiff_vina = np.array([np.median([v['vina']['dock'][0]['affinity'] for v in pocket]) for pocket in Globals.targetDiff_results])
    ar_vina = np.array([np.median([v['vina']['dock'][0]['affinity'] for v in pocket]) if len(pocket) > 0 else 0. for pocket in Globals.ar_results])
    pocket2mol_vina = np.array([np.median([v['vina']['dock'][0]['affinity'] for v in pocket]) for pocket in Globals.pocket2mol_results])

    all_vina = np.stack([ours_vina, targetDiff_vina, ar_vina, pocket2mol_vina], axis=0)
    best_vina_idx = np.argmin(all_vina, axis=0)

    plt.figure(figsize=(25, 6), dpi=100)

    ax = plt.subplot(1, 1, 1)
    ax.set_prop_cycle('color', plt.cm.Set1.colors)
    n_data = len(ours_vina)
    fig1_idx = np.argsort(ours_vina)
    ALPHA = 0.75
    POINT_SIZE = 128
    plt.scatter(np.arange(n_data), ours_vina[fig1_idx], label=f'Ours (lowest in {np.mean(best_vina_idx==0)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE)
    plt.scatter(np.arange(n_data), targetDiff_vina[fig1_idx], label=f'targetDiff (lowest in {np.mean(best_vina_idx==1)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), ar_vina[fig1_idx], label=f'AR (lowest in {np.mean(best_vina_idx==2)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), pocket2mol_vina[fig1_idx], label=f'Pocket2Mol (lowest in {np.mean(best_vina_idx==3)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)

    # plt.xticks([])
    plt.yticks(fontsize=16)
    for i in range(n_data):
        plt.axvline(i, c='0.1', lw=0.2)
    plt.xlim(-1, 100)
    plt.ylim(-13, -1.5)
    # plt.yticks([-10, -8, -6, -4, -2], [-10, -8, -6, -4, '$\geq$-2'], fontsize=25)
    plt.yticks([-12, -10, -8, -6, -4, -2], [-12, -10, -8, -6, -4, -2], fontsize=25)
    plt.ylabel('Median Vina Energy', fontsize=30)
    plt.legend(fontsize=25, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.3), frameon=False)
    plt.xticks(np.arange(0, 100, 10), [f'target {v}' for v in np.arange(0, 100, 10)], fontsize=25)

    plt.tight_layout()
    plt.savefig(save_path)
    print('save path: ', save_path)
    # plt.show()


# Vina Score Median score_only
def fig_Vina_Score_Median_Score_only(save_path):
    ours_vina = np.array([np.median([v['vina']['score_only'][0]['affinity'] for v in pocket]) for pocket in Globals.ours_results])
    targetDiff_vina = np.array([np.median([v['vina']['score_only'][0]['affinity'] for v in pocket]) for pocket in Globals.targetDiff_results])
    ar_vina = np.array([np.median([v['vina']['score_only'][0]['affinity'] for v in pocket]) if len(pocket) > 0 else 0. for pocket in Globals.ar_results])
    pocket2mol_vina = np.array([np.median([v['vina']['score_only'][0]['affinity'] for v in pocket]) for pocket in Globals.pocket2mol_results])

    all_vina = np.stack([ours_vina, targetDiff_vina, ar_vina, pocket2mol_vina], axis=0)
    best_vina_idx = np.argmin(all_vina, axis=0)

    plt.figure(figsize=(25, 6), dpi=100)

    ax = plt.subplot(1, 1, 1)
    ax.set_prop_cycle('color', plt.cm.Set1.colors)
    n_data = len(ours_vina)
    fig1_idx = np.argsort(ours_vina)
    ALPHA = 0.75
    POINT_SIZE = 128
    plt.scatter(np.arange(n_data), ours_vina[fig1_idx], label=f'Ours (lowest in {np.mean(best_vina_idx==0)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE)
    plt.scatter(np.arange(n_data), targetDiff_vina[fig1_idx], label=f'targetDiff (lowest in {np.mean(best_vina_idx==1)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), ar_vina[fig1_idx], label=f'AR (lowest in {np.mean(best_vina_idx==2)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), pocket2mol_vina[fig1_idx], label=f'Pocket2Mol (lowest in {np.mean(best_vina_idx==3)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)

    # plt.xticks([])
    plt.yticks(fontsize=16)
    for i in range(n_data):
        plt.axvline(i, c='0.1', lw=0.2)
    plt.xlim(-1, 100)
    plt.ylim(-13, -1.5)
    # plt.yticks([-10, -8, -6, -4, -2], [-10, -8, -6, -4, '$\geq$-2'], fontsize=25)
    plt.yticks([-12, -10, -8, -6, -4, -2], [-12, -10, -8, -6, -4, -2], fontsize=25)
    plt.ylabel('Median Vina Energy', fontsize=30)
    plt.legend(fontsize=25, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.3), frameon=False)
    plt.xticks(np.arange(0, 100, 10), [f'target {v}' for v in np.arange(0, 100, 10)], fontsize=25)

    plt.tight_layout()
    plt.savefig(save_path)
    print('save path: ', save_path)
    # plt.show()


# Vina Score Median minimize
def fig_Vina_Score_Median_Minimize(save_path):
    ours_vina = np.array([np.median([v['vina']['minimize'][0]['affinity'] for v in pocket]) for pocket in Globals.ours_results])
    targetDiff_vina = np.array([np.median([v['vina']['minimize'][0]['affinity'] for v in pocket]) for pocket in Globals.targetDiff_results])
    ar_vina = np.array([np.median([v['vina']['minimize'][0]['affinity'] for v in pocket]) if len(pocket) > 0 else 0. for pocket in Globals.ar_results])
    pocket2mol_vina = np.array([np.median([v['vina']['minimize'][0]['affinity'] for v in pocket]) for pocket in Globals.pocket2mol_results])

    all_vina = np.stack([ours_vina, targetDiff_vina, ar_vina, pocket2mol_vina], axis=0)
    best_vina_idx = np.argmin(all_vina, axis=0)

    plt.figure(figsize=(25, 6), dpi=100)

    ax = plt.subplot(1, 1, 1)
    ax.set_prop_cycle('color', plt.cm.Set1.colors)
    n_data = len(ours_vina)
    fig1_idx = np.argsort(ours_vina)
    ALPHA = 0.75
    POINT_SIZE = 128
    plt.scatter(np.arange(n_data), ours_vina[fig1_idx], label=f'Ours (lowest in {np.mean(best_vina_idx==0)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE)
    plt.scatter(np.arange(n_data), targetDiff_vina[fig1_idx], label=f'targetDiff (lowest in {np.mean(best_vina_idx==1)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), ar_vina[fig1_idx], label=f'AR (lowest in {np.mean(best_vina_idx==2)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), pocket2mol_vina[fig1_idx], label=f'Pocket2Mol (lowest in {np.mean(best_vina_idx==3)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)

    # plt.xticks([])
    plt.yticks(fontsize=16)
    for i in range(n_data):
        plt.axvline(i, c='0.1', lw=0.2)
    plt.xlim(-1, 100)
    plt.ylim(-13, -1.5)
    # plt.yticks([-10, -8, -6, -4, -2], [-10, -8, -6, -4, '$\geq$-2'], fontsize=25)
    plt.yticks([-12, -10, -8, -6, -4, -2], [-12, -10, -8, -6, -4, -2], fontsize=25)
    plt.ylabel('Median Vina Energy', fontsize=30)
    plt.legend(fontsize=25, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.3), frameon=False)
    plt.xticks(np.arange(0, 100, 10), [f'target {v}' for v in np.arange(0, 100, 10)], fontsize=25)

    plt.tight_layout()
    plt.savefig(save_path)
    print('save path: ', save_path)
    # plt.show()


# Vina Score Mean dock
def fig_Vina_Score_Mean_Dock(save_path):
    ours_vina = np.array([np.mean([v['vina']['dock'][0]['affinity'] for v in pocket]) for pocket in Globals.ours_results])
    targetDiff_vina = np.array([np.mean([v['vina']['dock'][0]['affinity'] for v in pocket]) for pocket in Globals.targetDiff_results])
    ar_vina = np.array([np.mean([v['vina']['dock'][0]['affinity'] for v in pocket]) if len(pocket) > 0 else 0. for pocket in Globals.ar_results])
    pocket2mol_vina = np.array([np.mean([v['vina']['dock'][0]['affinity'] for v in pocket]) for pocket in Globals.pocket2mol_results])

    all_vina = np.stack([ours_vina, targetDiff_vina, ar_vina, pocket2mol_vina], axis=0)
    best_vina_idx = np.argmin(all_vina, axis=0)

    plt.figure(figsize=(25, 6), dpi=100)

    ax = plt.subplot(1, 1, 1)
    ax.set_prop_cycle('color', plt.cm.Set1.colors)
    n_data = len(ours_vina)
    fig1_idx = np.argsort(ours_vina)
    ALPHA = 0.75
    POINT_SIZE = 128
    plt.scatter(np.arange(n_data), ours_vina[fig1_idx], label=f'Ours (lowest in {np.mean(best_vina_idx==0)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE)
    plt.scatter(np.arange(n_data), targetDiff_vina[fig1_idx], label=f'targetDiff (lowest in {np.mean(best_vina_idx==1)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), ar_vina[fig1_idx], label=f'AR (lowest in {np.mean(best_vina_idx==2)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), pocket2mol_vina[fig1_idx], label=f'Pocket2Mol (lowest in {np.mean(best_vina_idx==3)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)

    # plt.xticks([])
    plt.yticks(fontsize=16)
    for i in range(n_data):
        plt.axvline(i, c='0.1', lw=0.2)
    plt.xlim(-1, 100)
    plt.ylim(-13, -1.5)
    # plt.yticks([-10, -8, -6, -4, -2], [-10, -8, -6, -4, '$\geq$-2'], fontsize=25)
    plt.yticks([-12, -10, -8, -6, -4, -2], [-12, -10, -8, -6, -4, -2], fontsize=25)
    plt.ylabel('Mean Vina Energy', fontsize=30)
    plt.legend(fontsize=25, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.3), frameon=False)
    plt.xticks(np.arange(0, 100, 10), [f'target {v}' for v in np.arange(0, 100, 10)], fontsize=25)

    plt.tight_layout()
    plt.savefig(save_path)
    print('save path: ', save_path)
    # plt.show()


# Vina Score Mean score_only
def fig_Vina_Score_Mean_Score_only(save_path):
    ours_vina = np.array([np.mean([v['vina']['score_only'][0]['affinity'] for v in pocket]) for pocket in Globals.ours_results])
    targetDiff_vina = np.array([np.mean([v['vina']['score_only'][0]['affinity'] for v in pocket]) for pocket in Globals.targetDiff_results])
    ar_vina = np.array(
        [np.mean([v['vina']['score_only'][0]['affinity'] for v in pocket]) if len(pocket) > 0 else 0. for pocket in Globals.ar_results])
    pocket2mol_vina = np.array([np.mean([v['vina']['score_only'][0]['affinity'] for v in pocket]) for pocket in Globals.pocket2mol_results])

    all_vina = np.stack([ours_vina, targetDiff_vina, ar_vina, pocket2mol_vina], axis=0)
    best_vina_idx = np.argmin(all_vina, axis=0)

    plt.figure(figsize=(25, 6), dpi=100)

    ax = plt.subplot(1, 1, 1)
    ax.set_prop_cycle('color', plt.cm.Set1.colors)
    n_data = len(ours_vina)
    fig1_idx = np.argsort(ours_vina)
    ALPHA = 0.75
    POINT_SIZE = 128
    plt.scatter(np.arange(n_data), ours_vina[fig1_idx], label=f'Ours (lowest in {np.mean(best_vina_idx == 0) * 100:.0f}%)', alpha=ALPHA, s=POINT_SIZE)
    plt.scatter(np.arange(n_data), targetDiff_vina[fig1_idx], label=f'targetDiff (lowest in {np.mean(best_vina_idx == 1) * 100:.0f}%)', alpha=ALPHA,
                s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), ar_vina[fig1_idx], label=f'AR (lowest in {np.mean(best_vina_idx == 2) * 100:.0f}%)', alpha=ALPHA,
                s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), pocket2mol_vina[fig1_idx], label=f'Pocket2Mol (lowest in {np.mean(best_vina_idx == 3) * 100:.0f}%)', alpha=ALPHA,
                s=POINT_SIZE * 0.75)

    # plt.xticks([])
    plt.yticks(fontsize=16)
    for i in range(n_data):
        plt.axvline(i, c='0.1', lw=0.2)
    plt.xlim(-1, 100)
    plt.ylim(-13, -1.5)
    # plt.yticks([-10, -8, -6, -4, -2], [-10, -8, -6, -4, '$\geq$-2'], fontsize=25)
    plt.yticks([-12, -10, -8, -6, -4, -2], [-12, -10, -8, -6, -4, -2], fontsize=25)
    plt.ylabel('Median Vina Energy', fontsize=30)
    plt.legend(fontsize=25, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.3), frameon=False)
    plt.xticks(np.arange(0, 100, 10), [f'target {v}' for v in np.arange(0, 100, 10)], fontsize=25)

    plt.tight_layout()
    plt.savefig(save_path)
    print('save path: ', save_path)
    # plt.show()


# Vina Score Mean minimize
def fig_Vina_Score_Mean_Minimize(save_path):
    ours_vina = np.array([np.mean([v['vina']['minimize'][0]['affinity'] for v in pocket]) for pocket in Globals.ours_results])
    targetDiff_vina = np.array([np.mean([v['vina']['minimize'][0]['affinity'] for v in pocket]) for pocket in Globals.targetDiff_results])
    ar_vina = np.array([np.mean([v['vina']['minimize'][0]['affinity'] for v in pocket]) if len(pocket) > 0 else 0. for pocket in Globals.ar_results])
    pocket2mol_vina = np.array([np.mean([v['vina']['minimize'][0]['affinity'] for v in pocket]) for pocket in Globals.pocket2mol_results])

    all_vina = np.stack([ours_vina, targetDiff_vina, ar_vina, pocket2mol_vina], axis=0)
    best_vina_idx = np.argmin(all_vina, axis=0)

    plt.figure(figsize=(25, 6), dpi=100)

    ax = plt.subplot(1, 1, 1)
    ax.set_prop_cycle('color', plt.cm.Set1.colors)
    n_data = len(ours_vina)
    fig1_idx = np.argsort(ours_vina)
    ALPHA = 0.75
    POINT_SIZE = 128
    plt.scatter(np.arange(n_data), ours_vina[fig1_idx], label=f'Ours (lowest in {np.mean(best_vina_idx == 0) * 100:.0f}%)', alpha=ALPHA, s=POINT_SIZE)
    plt.scatter(np.arange(n_data), targetDiff_vina[fig1_idx], label=f'targetDiff (lowest in {np.mean(best_vina_idx == 1) * 100:.0f}%)', alpha=ALPHA,
                s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), ar_vina[fig1_idx], label=f'AR (lowest in {np.mean(best_vina_idx == 2) * 100:.0f}%)', alpha=ALPHA,
                s=POINT_SIZE * 0.75)
    plt.scatter(np.arange(n_data), pocket2mol_vina[fig1_idx], label=f'Pocket2Mol (lowest in {np.mean(best_vina_idx == 3) * 100:.0f}%)', alpha=ALPHA,
                s=POINT_SIZE * 0.75)

    # plt.xticks([])
    plt.yticks(fontsize=16)
    for i in range(n_data):
        plt.axvline(i, c='0.1', lw=0.2)
    plt.xlim(-1, 100)
    plt.ylim(-13, -1.5)
    # plt.yticks([-10, -8, -6, -4, -2], [-10, -8, -6, -4, '$\geq$-2'], fontsize=25)
    plt.yticks([-12, -10, -8, -6, -4, -2], [-12, -10, -8, -6, -4, -2], fontsize=25)
    plt.ylabel('Median Vina Energy', fontsize=30)
    plt.legend(fontsize=25, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.3), frameon=False)
    plt.xticks(np.arange(0, 100, 10), [f'target {v}' for v in np.arange(0, 100, 10)], fontsize=25)

    plt.tight_layout()
    plt.savefig(save_path)
    print('save path: ', save_path)
    # plt.show()


# main
def main():
    ablation_threshold = True
    ablation_network = True
    ablation_filter = True
    filter_flag = False

    if ablation_threshold:
        Globals.ours_5 = deal_ours(torch.load(Globals.ours_5))
        Globals.ours_6 = deal_ours(torch.load(Globals.ours_6))
        Globals.ours_7 = deal_ours(torch.load(Globals.ours_7))
        Globals.ours_8 = deal_ours(torch.load(Globals.ours_8))
        Globals.ours_9 = deal_ours(torch.load(Globals.ours_9))

    if ablation_network:
        Globals.ours_7_32 = torch.load(Globals.ours_7_32)
        Globals.ours_7_64 = torch.load(Globals.ours_7_64)
        Globals.ours_7_128 = torch.load(Globals.ours_7_128)
        Globals.ours_7_fuse = torch.load(Globals.ours_7_fuse)
        Globals.ours_7_KAN = torch.load(Globals.ours_7_KAN)
        Globals.ours_7_32 = deal_ours(Globals.ours_7_32)
        Globals.ours_7_64 = deal_ours(Globals.ours_7_64)
        Globals.ours_7_128 = deal_ours(Globals.ours_7_128)
        Globals.ours_7_fuse = deal_ours(Globals.ours_7_fuse)
        Globals.ours_7_KAN = deal_ours(Globals.ours_7_KAN)

    if filter_flag:
        print("\n------------------  filter the result  ------------------")
        Globals.ours_results = torch.load(Globals.ours_path)['all_results']
        export_mols(Globals.ours_results, save_path="./result/filter_mols/")
        eval_by_qikpropservice(save_path="./result/filter_mols/")
        unzip_eval_result(save_path="./result/filter_mols/")
        for istars in range(0, 10):
            filter_mol(Globals.ours_results, stars=istars, save_pt=True, save_path="./result/filter_mols/")
        sys.exit(0)

    print("\n------------------  tab: Bond  ------------------")
    tab_bond()

    print("\n------------------  tab: Ring Size  ------------------")
    tab_ring_size()

    print("\n------------------  tab: Metrics Summary  ------------------")
    tab_Metrics_Summary(ablation_threshold=ablation_threshold, ablation_network=ablation_network, ablation_filter=ablation_filter)

    print("\n------------------  fig: New Evaluation Metrics  ------------------")
    # save_path = './result/results_evaluate/Molecule_Quality_Score.pdf'
    # fig_new_Evaluation_Metrics(save_path)
    save_path = './result/results_evaluate/Molecule_Quality_Score_Sort.pdf'
    fig_new_Evaluation_Metrics_sort(save_path)

    # print("\n------------------  fig: distances  ------------------")
    # save_path = './result/results_evaluate/distances.pdf'
    # fig_distances(save_path)

    print("\n------------------  fig: RMSD  ------------------")
    save_path = './result/results_evaluate/RMSD.pdf'
    fig_RMSD(save_path)

    # print("\n------------------  fig: Vina Score  ------------------")
    # save_path = './result/results_evaluate/binding_Median_dock.pdf'
    # fig_Vina_Score_Median_Dock(save_path)
    # save_path = './result/results_evaluate/binding_Median_score_only.pdf'
    # fig_Vina_Score_Median_Score_only(save_path)
    # save_path = './result/results_evaluate/binding_Median_minimize.pdf'
    # fig_Vina_Score_Median_Minimize(save_path)
    # save_path = './result/results_evaluate/binding_Mean_dock.pdf'
    # fig_Vina_Score_Mean_Dock(save_path)
    # save_path = './result/results_evaluate/binding_Mean_score_only.pdf'
    # fig_Vina_Score_Mean_Score_only(save_path)
    # save_path = './result/results_evaluate/binding_Mean_minimize.pdf'
    # fig_Vina_Score_Mean_Minimize(save_path)

    print()


if __name__ == '__main__':
    main()

