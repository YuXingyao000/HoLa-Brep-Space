import math
import os.path
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import plyfile
import torch
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation
# from shared.common_utils import export_point_cloud, check_dir

from typing import Final
from einops import rearrange
import torchvision.transforms as T
import networkx as nx

from torch.nn.utils.rnn import pad_sequence
from scipy.spatial.transform import Rotation

import torch.nn.functional as F


def normalize_coord1112(v_points):
    points = v_points[..., :3]
    normals = v_points[..., 3:]
    shape = points.shape
    num_items = shape[0]
    points = points.reshape(num_items, -1, 3)
    target_points = points + normals.reshape(num_items, -1, 3)

    center = points.mean(dim=1, keepdim=True)
    scale = (torch.linalg.norm(points - center, dim=-1)).max(dim=1, keepdims=True)[0]
    assert scale.min() > 1e-3
    points = (points - center) / (scale[:, None] + 1e-6)
    target_points = (target_points - center) / (scale[:, None] + 1e-6)
    normals = target_points - points
    normals = normals / (1e-6 + torch.linalg.norm(normals, dim=-1, keepdim=True))

    points = points.reshape(shape)
    normals = normals.reshape(shape)

    return points, normals, center[:, 0], scale


def denormalize_coord1112(points, bbox):
    normal = points[..., 3:]
    points = points[..., :3]
    target_points = points + normal
    center = bbox[..., :3]
    scale = bbox[..., 3:4]
    while len(points.shape) > len(center.shape):
        center = center.unsqueeze(1)
        scale = scale.unsqueeze(1)
    points = points * scale + center
    target_points = target_points * scale + center
    normal = target_points - points
    normal = normal / (1e-6 + torch.linalg.norm(normal, dim=-1, keepdim=True))
    points = torch.cat((points, normal), dim=-1)
    return points


class Dummy_dataset(torch.utils.data.Dataset):
    def __init__(self, v_mode, v_conf):
        self.length = v_conf["length"]

    def __len__(self,):
        return self.length

    def __getitem__(self, idx):
        return "{:08d}".format(idx)
    
    @staticmethod
    def collate_fn(batch):
        return {
            "v_prefix"     : batch,
        }

# Input pc range from [-1,1]
def crop_pc(v_pc, v_min_points=1000):
    while True:
        num_points = v_pc.shape[0]
        index = np.arange(num_points)
        np.random.shuffle(index)
        pc_index = np.random.randint(0, num_points-1)
        center_pos = v_pc[pc_index, :3]
        length_xyz = np.random.rand(3) * 1.0
        obb = o3d.geometry.AxisAlignedBoundingBox(center_pos - length_xyz / 2, center_pos + length_xyz / 2)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(v_pc[:, :3])
        pcd.normals = o3d.utility.Vector3dVector(v_pc[:, 3:])
        inliers_indices = obb.get_point_indices_within_bounding_box(pcd.points)
        cropped = pcd.select_by_index(inliers_indices, invert=True)
        result = np.concatenate((np.asarray(cropped.points), np.asarray(cropped.normals)), axis=-1)
        if result.shape[0] > v_min_points:
            result = result[index % result.shape[0]]
            return result
   
def rotate_pc(v_pc, v_angle):
    matrix = Rotation.from_euler('xyz', v_angle).as_matrix()
    points = v_pc[:, :3]
    normals = v_pc[:, 3:]
    points1 = (matrix @ points.T).T

    ft = points + normals
    ft1 = (matrix @ ft.T).T

    fn1 = ft1 - points1

    fn1 = fn1 / (1e-6 + np.linalg.norm(fn1, axis=-1, keepdims=True))
    points = points1
    normals = fn1
    return np.concatenate((points, normals), axis=-1)

def noisy_pc(v_pc, v_length=0.02):
    noise = np.random.randn(*v_pc.shape) * v_length
    return v_pc + noise

def downsample_pc(v_pc, v_num_points):
    index = np.arange(v_pc.shape[0])
    np.random.shuffle(index)
    return v_pc[index[:v_num_points]]

def prepare_condition(v_condition_names, v_cond_root, v_folder_path, v_id_aug, 
                      v_cache_data=None, v_transform=None, 
                      v_num_points=None):
    condition = {

    }
    if "single_img" in v_condition_names or "multi_img" in v_condition_names or "sketch" in v_condition_names:
        num_max_multi_view = 8
        num_max_single_view = 64
        if v_id_aug == -1:
            v_id_aug = 0
        
        if v_cache_data:
            ori_data = np.load(v_cond_root / v_folder_path / "img_feature_dinov2.npy")
            if "single_img" in v_condition_names:
                idx = np.array([v_id_aug])
                img_features = torch.from_numpy(ori_data[v_id_aug][None,:]).float()
            elif "sketch" in v_condition_names:
                idx = np.array([v_id_aug])
                img_features = torch.from_numpy(ori_data[v_id_aug+num_max_single_view][None,:]).float()
            else:
                # idx = np.random.choice(np.arange(num_max_multi_view), 4, replace=False)
                idx = np.array([0,1,2,3])
                img_features = torch.from_numpy(ori_data[num_max_single_view + num_max_single_view + idx * num_max_single_view + v_id_aug]).float()
        
            condition["img_id"] = torch.from_numpy(idx)
            condition["img_features"] = img_features
        else:
            ori_data = np.load(v_cond_root / v_folder_path / "imgs.npz")
            if "single_img" in v_condition_names:
                idx = np.array([v_id_aug])
                imgs = ori_data["svr_imgs"][v_id_aug][None,:]
            elif "multi_img" in v_condition_names:
                # idx = np.random.choice(np.arange(num_max_multi_view), 4, replace=False)
                idx = np.array([0,1,2,3])
                imgs = ori_data["mvr_imgs"][idx * num_max_single_view + v_id_aug]
            else:
                idx = np.array([v_id_aug])
                imgs = ori_data["sketch_imgs"][v_id_aug][None,:]
            transformed_imgs = []
            for id in range(imgs.shape[0]):
                transformed_imgs.append(v_transform(imgs[id]))
            transformed_imgs = torch.stack(transformed_imgs, dim=0)
            condition["ori_imgs"] = torch.from_numpy(imgs)
            condition["imgs"] = transformed_imgs
            condition["img_id"] = torch.from_numpy(idx)
    if "pc" in v_condition_names:
        pc = o3d.io.read_point_cloud(str(v_cond_root / v_folder_path / "pc.ply"))
        points = np.concatenate((np.asarray(pc.points), np.asarray(pc.normals)), axis=-1)
        assert points.shape[0] == 10000
        # Already move to GPU
        # if v_id_aug != -1:
            # angles = np.array([
                # v_id_aug % 4,
                # v_id_aug // 4 % 4,
                # v_id_aug // 16
            # ])
            # points = rotate_pc(points, angles)
        
        # points = crop_pc(points, 1000)
        # points = noisy_pc(points)
        # points = downsample_pc(points, v_num_points)
        condition["points"] = torch.from_numpy(points).float()[None,]
    if "txt" in v_condition_names:
        if v_cache_data:
            difficulty = np.random.randint(0, 3)
            ori_data = np.load(v_cond_root / v_folder_path / "text_feat.npy")[difficulty]
            condition["txt_features"] = torch.from_numpy(ori_data).float()
        else:
            difficulty = 0
            condition["txt"] = open(v_cond_root / v_folder_path / "text.txt").readlines()[difficulty].strip()
            condition["id_txt"] = difficulty
    return condition
        

class AutoEncoder_dataset3(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(AutoEncoder_dataset3, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        self.max_intersection = 500
        self.scale_factor = 1
        if "scale_factor" in v_conf:
            self.scale_factor = int(v_conf["scale_factor"])
        if v_training_mode == "testing":
            listfile = v_conf['test_dataset']
        elif v_training_mode == "training":
            listfile = v_conf['train_dataset']
        elif v_training_mode == "validation":
            listfile = v_conf['val_dataset']
        else:
            raise

        self.data_folders = [item.strip() for item in open(listfile).readlines()]
        self.root = Path(v_conf["data_root"])

        # Cond related
        self.is_aug = v_conf["is_aug"]
        self.condition = v_conf["condition"]
        self.conditional_data_root = Path(v_conf["cond_root"]) if v_conf["cond_root"] is not None else None
        self.cached_condition = v_conf["cached_condition"]
        if v_training_mode == "validation":
            self.is_aug = 0
            self.cached_condition = False

        # Check data
        data_folders = []
        for item in self.data_folders:
            if os.path.exists(self.root / item / "data.npz"):
                data_folders.append(item)
        self.data_folders = data_folders

        # Check cond data
        if len(self.condition) > 0:
            data_folders = []
            for item in self.data_folders:
                if os.path.exists(self.conditional_data_root/item):
                    data_folders.append(item)
            print("Filter out {} folders without feat".format(len(self.data_folders)-len(data_folders)))
            self.data_folders = data_folders

        self.ori_length = len(self.data_folders)
        if v_training_mode == "testing" and self.is_aug == 1:
            self.data_folders = self.data_folders * 64
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


        if v_conf["is_overfit"]:
            self.data_folders = self.data_folders[:100]
            if v_training_mode == "training":
                self.data_folders = self.data_folders * 100
        print(len(self.data_folders))

    def __len__(self):
        # return 100
        return len(self.data_folders) * self.scale_factor

    def __getitem__(self, idx):
        # idx = 0
        prefix = self.data_folders[idx%len(self.data_folders)]
        data_npz = np.load(str(self.root / prefix / "data.npz"))
        if self.mode == "testing" and self.is_aug==1:
            prefix += "_{}".format(idx // self.ori_length)
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        edge_points = torch.from_numpy(data_npz['sample_points_lines'])

        id_aug = -1
        if self.is_aug == 0:
            matrix = np.identity(3)
        if self.is_aug == 1:
            id_aug = idx // self.ori_length
            if self.mode == "testing":
                angles = np.array([
                    id_aug % 4,
                    id_aug // 4 % 4,
                    id_aug // 16
                ])
                matrix = Rotation.from_euler('xyz', angles * np.pi / 2).as_matrix()
            else:
                matrix = Rotation.from_euler('xyz', np.random.randint(0, 3, 3) * np.pi / 2).as_matrix()
        elif self.is_aug == 2:
            matrix = Rotation.from_euler('xyz', np.random.rand(3) * np.pi * 2).as_matrix()
        if self.is_aug != 0:
            matrix = torch.from_numpy(matrix).float()
            fp = face_points[..., :3].reshape(-1, 3)
            lp = edge_points[..., :3].reshape(-1, 3)
            fp1 = (matrix @ fp.T).T
            lp1 = (matrix @ lp.T).T

            if face_points.shape[1] > 3:
                fn = face_points[..., 3:].reshape(-1, 3)
                ln = edge_points[..., 3:].reshape(-1, 3)
                ft = fp + fn
                lt = lp + ln
                ft1 = (matrix @ ft.T).T
                lt1 = (matrix @ lt.T).T

                fn1 = ft1 - fp1
                ln1 = lt1 - lp1

                fn1 = fn1 / (1e-6 + torch.linalg.norm(fn, dim=-1, keepdim=True))
                ln1 = ln1 / (1e-6 + torch.linalg.norm(ln, dim=-1, keepdim=True))
                face_points[..., 3:] = fn1.reshape(face_points[..., 3:].shape)
                edge_points[..., 3:] = ln1.reshape(edge_points[..., 3:].shape)

            face_points[..., :3] = fp1.reshape(face_points[..., :3].shape)
            edge_points[..., :3] = lp1.reshape(edge_points[..., :3].shape)

        num_faces = face_points.shape[0]
        num_edges = edge_points.shape[0]

        face_adj = torch.from_numpy(data_npz['face_adj'])
        edge_face_connectivity = torch.from_numpy(data_npz['edge_face_connectivity'])
        edge_face_connectivity = edge_face_connectivity[edge_face_connectivity[:, 1] != edge_face_connectivity[:, 2]]

        zero_positions = torch.from_numpy(data_npz['zero_positions'])
        if zero_positions.shape[0] > edge_face_connectivity.shape[0]:
            index = np.random.choice(zero_positions.shape[0], edge_face_connectivity.shape[0], replace=False)
            zero_positions = zero_positions[index]

        face_points_norm, face_normal_norm, face_center, face_scale = normalize_coord1112(face_points)
        edge_points_norm, edge_normal_norm, edge_center, edge_scale = normalize_coord1112(edge_points)

        face_norm = torch.cat((face_points_norm, face_normal_norm), dim=-1)
        edge_norm = torch.cat((edge_points_norm, edge_normal_norm), dim=-1)

        face_bbox = torch.cat((face_center, face_scale), dim=-1)
        edge_bbox = torch.cat((edge_center, edge_scale), dim=-1)

        condition = prepare_condition(self.condition, self.conditional_data_root, prefix, id_aug,
                                      self.cached_condition, self.transform, self.conf["num_points"])

        return (
            prefix,
            face_points, edge_points,
            face_norm, edge_norm,
            face_bbox, edge_bbox,
            edge_face_connectivity, zero_positions, face_adj,
            condition
        )

    @staticmethod
    def collate_fn(batch):
        (
            prefix,
            face_points, edge_points,
            face_norm, edge_norm,
            face_bbox, edge_bbox,
            edge_face_connectivity, zero_positions, face_adj,
            conditions
        ) = zip(*batch)
        bs = len(prefix)

        flat_zero_positions = []
        num_face_record = []

        num_faces = 0
        num_edges = 0
        edge_conn_num = []
        for i in range(bs):
            edge_face_connectivity[i][:, 0] += num_edges
            edge_face_connectivity[i][:, 1:] += num_faces
            edge_conn_num.append(edge_face_connectivity[i].shape[0])
            flat_zero_positions.append(zero_positions[i] + num_faces)
            num_faces += face_norm[i].shape[0]
            num_edges += edge_norm[i].shape[0]
            num_face_record.append(face_norm[i].shape[0])
        num_face_record = torch.tensor(num_face_record, dtype=torch.long)
        num_sum_edges = sum(edge_conn_num)
        edge_attn_mask = torch.ones((num_sum_edges, num_sum_edges), dtype=bool)
        id_cur = 0
        for i in range(bs):
            edge_attn_mask[id_cur:id_cur + edge_conn_num[i], id_cur:id_cur + edge_conn_num[i]] = False
            id_cur += edge_conn_num[i]

        num_max_faces = num_face_record.max()
        valid_mask = torch.zeros((bs, num_max_faces), dtype=bool)
        for i in range(bs):
            valid_mask[i, :num_face_record[i]] = True
        attn_mask = torch.ones((num_faces, num_faces), dtype=bool)
        id_cur = 0
        for i in range(bs):
            attn_mask[id_cur:id_cur + face_norm[i].shape[0], id_cur: id_cur + face_norm[i].shape[0]] = False
            id_cur += face_norm[i].shape[0]

        dtype = torch.float32
        flat_zero_positions = torch.cat(flat_zero_positions, dim=0)

        keys = conditions[0].keys()
        condition_out = {key: [] for key in keys}
        for idx in range(len(conditions)):
            for key in keys:
                condition_out[key].append(conditions[idx][key])

        for key in keys:
            condition_out[key] = torch.stack(condition_out[key], dim=0) if isinstance(condition_out[key][0], torch.Tensor) else condition_out[key]

        return {
            "v_prefix"              : prefix,
            "face_points"             : torch.cat(face_points, dim=0).to(dtype),
            "face_norm"             : torch.cat(face_norm, dim=0).to(dtype),
            "edge_points"             : torch.cat(edge_points, dim=0).to(dtype),
            "edge_norm"             : torch.cat(edge_norm, dim=0).to(dtype),
            "face_bbox"             : torch.cat(face_bbox, dim=0).to(dtype),
            "edge_bbox"             : torch.cat(edge_bbox, dim=0).to(dtype),

            "edge_face_connectivity": torch.cat(edge_face_connectivity, dim=0),
            "zero_positions"        : flat_zero_positions,
            "attn_mask"             : attn_mask,
            "edge_attn_mask"        : edge_attn_mask,

            "num_face_record"       : num_face_record,
            "valid_mask"            : valid_mask,
            "conditions"   : condition_out
        }


class Diffusion_dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Diffusion_dataset, self).__init__()        
        self.mode = v_training_mode
        self.conf = v_conf
        scale_factor = int(v_conf["scale_factor"])
        self.max_intersection = 500
        self.latent_root = Path(v_conf['face_z'])
        if v_training_mode == "testing":
            self.data_split = Path(v_conf['test_dataset'])
            scale_factor = 1
        elif v_training_mode == "training":
            self.data_split = Path(v_conf['train_dataset'])
        elif v_training_mode == "validation":
            self.data_split = Path(v_conf['val_dataset'])
            scale_factor = 1
        else:
            raise

        self.addition_tag = v_conf["addition_tag"]
        self.pad_method = v_conf["pad_method"]
        self.max_faces = v_conf["num_max_faces"]
        print("Use deduplicate list ", self.data_split)
        filelist = [item.strip() for item in open(self.data_split).readlines()]
        filelist.sort()

        # Cond related
        self.is_aug = v_conf["is_aug"]
        self.cached_condition = v_conf["cached_condition"]
        if v_training_mode == "validation":
            self.is_aug = False
            self.cached_condition = False
        self.condition = v_conf["condition"]
        self.conditional_data_root = Path(v_conf["cond_root"])
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        # Check cond data
        if len(self.condition) > 0:
            data_folders = []
            for item in filelist:
                if os.path.exists(self.conditional_data_root/item):
                    data_folders.append(item)
            print("Filter out {} folders without feat".format(len(filelist)-len(data_folders)))
            filelist = data_folders
        
        if v_conf["overfit"]:  # Overfitting mode
            self.data_folders = filelist[:100] * scale_factor
        else:
            self.data_folders = filelist * scale_factor

        print("Total data num:", len(self.data_folders))
        return

    def __len__(self):
        return len(self.data_folders)

    def __getitem__(self, idx):
        # idx = 0
        folder_path = self.data_folders[idx]
        if self.is_aug != 0:
            id_aug = np.random.randint(0,63)
            data_npz = np.load(self.latent_root / (folder_path+f"_{id_aug}") / "features.npy")
        else:
            id_aug = -1
            data_npz = np.load(self.latent_root / (folder_path+f"_{0}") / "features.npy")
        face_features = torch.from_numpy(data_npz)

        if self.pad_method == "zero":
            padded_face_features = torch.zeros((self.max_faces, 32), dtype=torch.float32)
            padded_face_features[:face_features.shape[0]] = face_features
        elif self.pad_method == "random":
            add_flag = torch.ones(self.max_faces, dtype=face_features.dtype)
            add_flag[face_features.shape[0]:] = -1
            add_flag = add_flag[:,None]

            if False:
                index = torch.randperm(face_features.shape[0])
                num_repeats = math.ceil(self.max_faces / index.shape[0])
                index = index.repeat(num_repeats)[:self.max_faces]
                index2 = torch.randperm(self.max_faces)
                index = index[index2]
            else:
                positions = torch.arange(self.max_faces, device=face_features.device)
                mandatory_mask = positions < face_features.shape[0]
                random_indices = (torch.rand((self.max_faces,), device=face_features.device) * face_features.shape[0]).long()
                indices = torch.where(mandatory_mask, positions, random_indices)
                r_indices = torch.argsort(torch.rand((self.max_faces,), device=face_features.device), dim=0)
                index = indices.gather(0, r_indices)
            padded_face_features = face_features[index]

            if self.addition_tag:
                padded_face_features = torch.cat((padded_face_features, add_flag[index2]), dim=-1)
        else:
            raise ValueError("Invalid pad method")
        condition = prepare_condition(self.condition, self.conditional_data_root, folder_path, id_aug,
                                      self.cached_condition, self.transform, self.conf["num_points"])
        return (
            folder_path,
            padded_face_features,
            condition,
            id_aug
        )

    @staticmethod
    def collate_fn(batch):
        (
            v_prefix, v_face_features, conditions, id_aug
        ) = zip(*batch)

        face_features = torch.stack(v_face_features, dim=0)
        id_aug = torch.tensor(id_aug)

        keys = conditions[0].keys()
        condition_out = {key: [] for key in keys}
        for idx in range(len(conditions)):
            for key in keys:
                condition_out[key].append(conditions[idx][key])

        for key in keys:
            condition_out[key] = torch.stack(condition_out[key], dim=0) if isinstance(condition_out[key][0], torch.Tensor) else condition_out[key]

        return {
            "v_prefix"     : v_prefix,
            "face_features": face_features,
            "conditions"   : condition_out,
            "id_aug"   : id_aug,
        }
