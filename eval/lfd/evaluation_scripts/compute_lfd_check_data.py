# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import random
import shutil

import numpy as np
import ray
import torch
import os
from tqdm import tqdm
from load_data.interface import LoadData
import pickle
from multiprocessing import Pool, cpu_count

def read_all_data(folder_list, load_data, add_model_str=True, add_ori_name=False):
    all_data = []

    for f in folder_list:
        if add_model_str:
            result = load_data.run(os.path.join(f, 'model', 'mesh'))
        elif add_ori_name:
            result = load_data.run(os.path.join(f, f.split('/')[-1], 'mesh'))
        else:
            result = load_data.run(os.path.join(f, 'mesh'))

        all_data.append(result)
    q8_table = all_data[0][0]
    align_10 = all_data[0][1]
    dest_ArtCoeff = [r[2][np.newaxis, :] for r in all_data]
    dest_FdCoeff_q8 = [r[3][np.newaxis, :] for r in all_data]
    dest_CirCoeff_q8 = [r[4][np.newaxis, :] for r in all_data]
    dest_EccCoeff_q8 = [r[5][np.newaxis, :] for r in all_data]
    SRC_ANGLE = 10
    ANGLE = 10
    CAMNUM = 10
    ART_COEF = 35
    FD_COEF = 10
    n_shape = len(all_data)
    dest_ArtCoeff = torch.from_numpy(np.ascontiguousarray(np.concatenate(dest_ArtCoeff, axis=0))).int().cuda().reshape(n_shape, SRC_ANGLE,
                                                                                                                       CAMNUM, ART_COEF)
    dest_FdCoeff_q8 = torch.from_numpy(np.ascontiguousarray(np.concatenate(dest_FdCoeff_q8, axis=0))).int().cuda().reshape(n_shape, ANGLE,
                                                                                                                           CAMNUM, FD_COEF)
    dest_CirCoeff_q8 = torch.from_numpy(np.ascontiguousarray(np.concatenate(dest_CirCoeff_q8, axis=0))).int().cuda().reshape(n_shape, ANGLE,
                                                                                                                             CAMNUM)
    dest_EccCoeff_q8 = torch.from_numpy(np.ascontiguousarray(np.concatenate(dest_EccCoeff_q8, axis=0))).int().cuda().reshape(n_shape, ANGLE,
                                                                                                                             CAMNUM)
    q8_table = torch.from_numpy(np.ascontiguousarray(q8_table)).int().cuda().reshape(256, 256)
    align_10 = torch.from_numpy(np.ascontiguousarray(align_10)).int().cuda().reshape(60, 20)  ##
    return q8_table.contiguous(), align_10.contiguous(), dest_ArtCoeff.contiguous(), \
        dest_FdCoeff_q8.contiguous(), dest_CirCoeff_q8.contiguous(), dest_EccCoeff_q8.contiguous()


def compute_lfd_all(src_folder_list, tgt_folder_list, log):
    load_data = LoadData()

    add_ori_name = False
    add_model_str = False
    src_folder_list.sort()
    tgt_folder_list.sort()

    q8_table, align_10, src_ArtCoeff, src_FdCoeff_q8, src_CirCoeff_q8, src_EccCoeff_q8 = read_all_data(src_folder_list, load_data,
                                                                                                       add_model_str=False)
    q8_table, align_10, tgt_ArtCoeff, tgt_FdCoeff_q8, tgt_CirCoeff_q8, tgt_EccCoeff_q8 = read_all_data(tgt_folder_list, load_data,
                                                                                                       add_model_str=add_model_str,
                                                                                                       add_ori_name=add_ori_name)  ###

    from lfd_all_compute.lfd import LFD
    lfd = LFD()
    lfd_matrix = lfd.forward(
            q8_table, align_10, src_ArtCoeff, src_FdCoeff_q8, src_CirCoeff_q8, src_EccCoeff_q8,
            tgt_ArtCoeff, tgt_FdCoeff_q8, tgt_CirCoeff_q8, tgt_EccCoeff_q8, log)
    # print(lfd_matrix)
    # print(lfd_matrix.shape)
    mmd = lfd_matrix.float().min(dim=0)[0].mean()
    mmd_swp = lfd_matrix.float().min(dim=1)[0].mean()
    # print(mmd)
    # print(mmd_swp)
    return lfd_matrix.data.cpu().numpy()

def get_file_size_kb(mesh_path):
    return int(os.path.getsize(mesh_path) / 1024)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True, help="path to the mesh folder")
    parser.add_argument("--lfd_feat", type=str, required=True, help="path to the preprocessed shapenet dataset")
    parser.add_argument("--save_root", type=str, required=True, help="path to the save resules shapenet dataset")
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers to run in parallel")
    parser.add_argument("--list", type=str, default=None, help="list file in the training set")
    args = parser.parse_args()
    num_workers = args.num_workers
    listfile = args.list

    mesh_folder_path = args.mesh_path
    lfd_feat_path = args.lfd_feat
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)


    print(f"mesh_path: {mesh_folder_path}")
    print(f"lfd_feat_path: {lfd_feat_path}")

    all_folders = os.listdir(mesh_folder_path)
    all_folders.sort()
    print("Get mesh_size")
    mesh_folder_list = []
    mesh_path_list = []
    # mesh_size_list = []
    for mesh_folder in tqdm(all_folders):
        mesh_path = os.path.join(mesh_folder_path, mesh_folder, "mesh.stl")
        mesh_folder_list.append(mesh_folder)
        mesh_path_list.append(mesh_path)
        # mesh_size_list.append(int(os.path.getsize(mesh_path) / 1024))

    with Pool(processes=cpu_count()) as pool:
        mesh_size_list = list(tqdm(pool.imap(get_file_size_kb, mesh_path_list), total=len(mesh_path_list)))

    # sort according to the size of the mesh file
    assert len(mesh_size_list) == len(mesh_folder_list)
    # mesh_folder_list = [x for _, x in sorted(zip(mesh_size_list, mesh_folder_list))]
    # mesh_size_list = sorted(mesh_size_list)
    mesh_size_list = np.array(mesh_size_list)
    print(f"Max size: {mesh_size_list.max()}")
    print(f"Min size: {mesh_size_list.min()}")
    print(f"Total {mesh_size_list.shape} mesh_folder to process")

    tgt_folder_list = mesh_folder_list

    if listfile is not None:
        valid_folders = [item.strip() for item in open(listfile, 'r').readlines()]
        tgt_folder_list = sorted(list(set(valid_folders) & set(tgt_folder_list)))
        tgt_folder_list = [os.path.join(lfd_feat_path, f) for f in tgt_folder_list]
    else:
        tgt_folder_list = [os.path.join(lfd_feat_path, f) for f in tgt_folder_list]

    src_folder_list = tgt_folder_list

    start_from_size_end = 0
    print(f"Start from size_end: {start_from_size_end}")
    print((mesh_size_list>start_from_size_end).sum()/mesh_size_list.shape[0])

    ray.init(
            num_cpus=os.cpu_count(),
            num_gpus=num_workers,
    )

    compute_lfd_all_remote = ray.remote(num_gpus=1, num_cpus=os.cpu_count() // num_workers)(compute_lfd_all)

    print("Check data")
    print(f"len of src_folder_list: {len(src_folder_list)}")
    print(f"len of tgt_folder_list: {len(tgt_folder_list)}")
    print(src_folder_list[0])
    print(tgt_folder_list[0])

    batch_size = 1
    offset = 2

    for size_start in tqdm(range(mesh_size_list.min(), mesh_size_list.max(), batch_size)):
        size_end = size_start + offset
        print(f"size_start: {size_start}, size_end: {size_end}, max_size: {mesh_size_list.max()}")
        if size_end <= start_from_size_end:
            continue
        # get the folder list for the current batch
        hitted_idx = np.where((mesh_size_list >= size_start) & (mesh_size_list <= size_end))[0]
        print(f"len of hitted folder: {len(hitted_idx)}")
        if len(hitted_idx) == 0:
            continue
        local_num_workers = min(num_workers, len(hitted_idx))
        local_tgt_folder_list = [tgt_folder_list[i] for i in hitted_idx]
        local_src_folder_list = local_tgt_folder_list
        results = []
        for i in range(local_num_workers):
            local_i_start = i * len(local_src_folder_list) // local_num_workers
            local_i_end = (i + 1) * len(local_src_folder_list) // local_num_workers
            results.append(compute_lfd_all_remote.remote(
                    local_src_folder_list[local_i_start:local_i_end],
                    local_tgt_folder_list,
                    i == 0))
        lfd_matrix = ray.get(results)
        lfd_matrix = np.concatenate(lfd_matrix, axis=0)

        save_name = os.path.join(save_root, f"lfd_{size_start:07d}kb_{size_end:07d}kb.pkl")
        pickle.dump([local_tgt_folder_list, lfd_matrix], open(save_name, 'wb'))
        print(f"pkl is saved to {save_name}\n\n")
