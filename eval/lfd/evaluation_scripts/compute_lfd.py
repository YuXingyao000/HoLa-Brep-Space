# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import random
import numpy as np
import ray
import torch
import os
from tqdm import tqdm
from load_data.interface import LoadData


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
    dest_ArtCoeff = torch.from_numpy(np.ascontiguousarray(np.concatenate(dest_ArtCoeff, axis=0))).int().cuda().reshape(n_shape, SRC_ANGLE, CAMNUM, ART_COEF)
    dest_FdCoeff_q8 = torch.from_numpy(np.ascontiguousarray(np.concatenate(dest_FdCoeff_q8, axis=0))).int().cuda().reshape(n_shape, ANGLE, CAMNUM, FD_COEF)
    dest_CirCoeff_q8 = torch.from_numpy(np.ascontiguousarray(np.concatenate(dest_CirCoeff_q8, axis=0))).int().cuda().reshape(n_shape, ANGLE, CAMNUM)
    dest_EccCoeff_q8 = torch.from_numpy(np.ascontiguousarray(np.concatenate(dest_EccCoeff_q8, axis=0))).int().cuda().reshape(n_shape, ANGLE, CAMNUM)
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

    q8_table, align_10, src_ArtCoeff, src_FdCoeff_q8, src_CirCoeff_q8, src_EccCoeff_q8 = read_all_data(src_folder_list, load_data, add_model_str=False)
    q8_table, align_10, tgt_ArtCoeff, tgt_FdCoeff_q8, tgt_CirCoeff_q8, tgt_EccCoeff_q8 = read_all_data(tgt_folder_list, load_data, add_model_str=add_model_str, add_ori_name=add_ori_name)  ###

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



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True, help="path to the save resules shapenet dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the preprocessed shapenet dataset")
    parser.add_argument("--gen_path", type=str, required=True, help="path to the generated models")
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers to run in parallel")
    parser.add_argument("--list", type=str, default=None, help="list file in the training set")
    args = parser.parse_args()
    save_path = '/'.join(args.save_name.split('/')[:-1])
    os.makedirs(save_path, exist_ok=True)
    num_workers = args.num_workers
    listfile = args.list
    ray.init(
        num_cpus=os.cpu_count(),
        num_gpus=num_workers,
    )
    print(f"dataset_path: {args.dataset_path}")
    print(f"gen_path: {args.gen_path}")
    assert os.path.exists(args.dataset_path) and os.path.exists(args.gen_path)

    tgt_folder_list = sorted(os.listdir(args.dataset_path))
    if listfile is not None:
        valid_folders = [item.strip() for item in open(listfile, 'r').readlines()]
        tgt_folder_list = sorted(list(set(valid_folders) & set(tgt_folder_list)))
        tgt_folder_list = [os.path.join(args.dataset_path, f) for f in tgt_folder_list]
    else:
        tgt_folder_list = [os.path.join(args.dataset_path, f) for f in tgt_folder_list]

    src_folder_list = os.listdir(args.gen_path)
    random.shuffle(src_folder_list)
    src_folder_list = sorted(src_folder_list[:3000])
    src_folder_list = [os.path.join(args.gen_path, f) for f in src_folder_list]

    compute_lfd_all_remote = ray.remote(num_gpus=1, num_cpus=os.cpu_count() // num_workers)(compute_lfd_all)

    print("Check data")
    print(f"len of src_folder_list: {len(src_folder_list)}")
    print(f"len of tgt_folder_list: {len(tgt_folder_list)}")
    # print(src_folder_list[0])
    # print(tgt_folder_list[0])

    results = []
    for i in range(num_workers):
        i_start = i * len(src_folder_list) // num_workers
        i_end = (i + 1) * len(src_folder_list) // num_workers
        # print(i, i_start, i_end)
        results.append(compute_lfd_all_remote.remote(
            src_folder_list[i_start:i_end],
            tgt_folder_list,
            i==0))

    lfd_matrix = ray.get(results)
    lfd_matrix = np.concatenate(lfd_matrix, axis=0)
    import pickle
    save_name = args.save_name
    nearest_name = [tgt_folder_list[idx].split("/")[-1] for idx in lfd_matrix.argmin(axis=1)]
    src_folder_list = [src_folder_list[idx].split("/")[-1] for idx in range(len(src_folder_list))]
    pickle.dump([src_folder_list, nearest_name, lfd_matrix], open(save_name, 'wb'))
    print(f"pkl is saved to {save_name}")
