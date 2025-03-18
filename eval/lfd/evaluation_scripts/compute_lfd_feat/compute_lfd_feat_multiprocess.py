# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse
import glob

import numpy as np
import torch
import os
import random
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
# import kaolin as kal
import point_cloud_utils as pcu
import trimesh

from tqdm import tqdm


def seed_everything(seed):
    if seed < 0:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_mesh_v(mesh_name, normalized_scale=0.9):
    if mesh_name.endswith('obj') or mesh_name.endswith('OBJ'):
        mesh_1 = kal.io.obj.import_mesh(mesh_name)
        vertices = mesh_1.vertices.cpu().numpy()
        mesh_f1 = mesh_1.faces.cpu().numpy()
    # elif mesh_name.endswith('ply'):
    #     vertices, mesh_f1 = pcu.load_mesh_vf(mesh_name)
    elif mesh_name.endswith('stl') or mesh_name.endswith('ply'):
        mesh = trimesh.load_mesh(mesh_name, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in mesh.geometry.values()))
        vertices = np.asarray(mesh.vertices)
        mesh_f1 = np.asarray(mesh.faces)
    else:
        raise NotImplementedError

    if vertices.shape[0] == 0:
        return None, None

    scale = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    mesh_v1 = vertices / (scale+1e-6) * normalized_scale
    return mesh_v1, mesh_f1


from lfd_me import MeshEncoder
from functools import partial


def align_mesh_feature(mesh_name, align_feature_sample_folder):
    # mesh_fodler = mesh_name.split('/')[-3:]
    # print(mesh_fodler)
    # mesh_fodler[-1] = mesh_fodler[-1].split('.')[0]
    # print(mesh_fodler)
    # mesh_fodler = '/'.join(mesh_fodler)
    mesh_fodler = os.path.basename(os.path.dirname(mesh_name))
    mesh_fodler = os.path.join(align_feature_sample_folder, mesh_fodler)
    # print(mesh_fodler)

    if not os.path.exists(mesh_fodler):
        os.makedirs(mesh_fodler)
    if os.path.exists(os.path.join(mesh_fodler, 'mesh_q4_v1.8.art')) and os.path.getsize(
            os.path.join(mesh_fodler, 'mesh_q4_v1.8.art')) > 1000:
        temp_dir_path = Path(mesh_fodler)
        file_name = 'mesh'
        temp_path = temp_dir_path / "{}.obj".format(file_name)
        path = temp_path.with_suffix("").as_posix()
        return path

    mesh_v, mesh_f = load_mesh_v(mesh_name, normalized_scale=1.0)
    if mesh_v is None:
        return None  # No face here

    mesh = MeshEncoder(mesh_v, mesh_f, folder=mesh_fodler, file_name='mesh', )
    mesh.align_mesh()
    return mesh.get_path()


def compute_lfd_feture(sample_pcs, n_process, save_path):
    align_feature_sample_folder = save_path
    os.makedirs(align_feature_sample_folder, exist_ok=True)
    print('==> one model')
    align_mesh_feature(sample_pcs[0], align_feature_sample_folder)
    N_process = n_process
    path_list = []
    if n_process == 0:
        for i in tqdm(range(len(sample_pcs))):
            align_mesh_feature(sample_pcs[i], align_feature_sample_folder)
        exit()
    print('==> multi process')
    pool = Pool(N_process)
    for x in tqdm(
            pool.imap_unordered(partial(align_mesh_feature, align_feature_sample_folder=align_feature_sample_folder), sample_pcs),
            total=len(sample_pcs)):
        path_list.append(x)
    pool.close()
    pool.join()


def load_data_with_prefix(root_folder, prefix, folder_list_txt=None):
    data_files = []
    folder_list = []
    if folder_list_txt is not None:
        with open(folder_list_txt, "r") as f:
            folder_list = f.read().splitlines()
    # Walk through the directory tree starting from the root folder
    for root, dirs, files in os.walk(root_folder):
        if folder_list_txt is not None and os.path.basename(root) not in folder_list:
            continue
        for filename in files:
            # Check if the file ends with the specified prefix
            if filename.endswith(prefix):
                file_path = os.path.join(root, filename)
                data_files.append(file_path)

    return data_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_path", type=str, required=True, help="path to the generated models")
    parser.add_argument("--save_path", type=str, required=True, help="path to save the generated features for each model")
    parser.add_argument("--n_models", type=int, default=-1, help="Number of models used for evaluation")
    parser.add_argument("--n_process", type=int, default=-1, help="Number of process used for evaluation")
    parser.add_argument("--prefix", type=str, required=False, default="mesh.ply")

    args = parser.parse_args()
    if args.n_process == -1:
        num_cpus = min(64, os.cpu_count())
    else:
        num_cpus = args.n_process
    models = []
    all_folders = os.listdir(args.gen_path)
    for folder in tqdm(all_folders):
        if not os.path.isdir(os.path.join(args.gen_path, folder)):
            continue
        files = glob.glob(os.path.join(args.gen_path, folder, args.prefix))
        if len(files) == 0:
            continue
        models.append(os.path.abspath(files[0]))
    models.sort()
    print(f"Loading {len(models)} models")
    compute_lfd_feture(models, num_cpus, os.path.abspath(args.save_path))
