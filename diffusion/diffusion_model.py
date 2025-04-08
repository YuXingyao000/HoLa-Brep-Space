import importlib
import math
import numpy as np
import time
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import torch
from torch import autocast, isnan, nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from einops import rearrange, reduce

from diffusers import DDPMScheduler
from tqdm import tqdm

from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG, \
    PointnetFPModule
from scipy.spatial.transform import Rotation


def add_timer(time_statics, v_attr, timer):
    if v_attr not in time_statics:
        time_statics[v_attr] = 0.
    time_statics[v_attr] += time.time() - timer
    return time.time()


def sincos_embedding(input, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param input: a N-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=input.dtype, device=input.device) / half
    )
    for _ in range(len(input.size())):
        freqs = freqs[None]
    args = input.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def inv_sigmoid(x):
    return torch.log(x / (1 - x + 1e-6))


class Diffusion_condition(nn.Module):
    def __init__(self, v_conf, ):
        super().__init__()
        self.dim_input = 8 * 2 * 2
        self.dim_latent = v_conf["diffusion_latent"]
        self.dim_condition = 256
        self.dim_total = self.dim_latent + self.dim_condition
        self.time_statics = [0 for _ in range(10)]

        self.addition_tag = False
        if "addition_tag" in v_conf:
            self.addition_tag = v_conf["addition_tag"]
        if self.addition_tag:
            self.dim_input += 1 

        self.p_embed = nn.Sequential(
            nn.Linear(self.dim_input, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.dim_latent),
        )

        layer1 = nn.TransformerEncoderLayer(
                d_model=self.dim_total,
                nhead=self.dim_total // 64, norm_first=True, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.net1 = nn.TransformerEncoder(layer1, 24, nn.LayerNorm(self.dim_total))
        self.fc_out = nn.Sequential(
                nn.Linear(self.dim_total, self.dim_total),
                nn.LayerNorm(self.dim_total),
                nn.SiLU(),
                nn.Linear(self.dim_total, self.dim_input),
        )

        self.with_img = False
        self.with_pc = False
        self.with_txt = False
        self.is_aug = v_conf["is_aug"]
        if "single_img" in v_conf["condition"] or "multi_img" in v_conf["condition"] or "sketch" in v_conf["condition"]:
            self.with_img = True
            self.img_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
            for param in self.img_model.parameters():
                param.requires_grad = False
            self.img_model.eval()

            self.img_fc = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, self.dim_condition),
            )
            self.camera_embedding = nn.Sequential(
                nn.Embedding(8, 256),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, self.dim_condition),
            )
        if "pc" in v_conf["condition"]:
            self.with_pc = True
            self.SA_modules = nn.ModuleList()
            # PointNet2
            c_in = 6
            with_bn = False
            self.SA_modules.append(
                    PointnetSAModuleMSG(
                            npoint=1024,
                            radii=[0.05, 0.1],
                            nsamples=[16, 32],
                            mlps=[[c_in, 32], [c_in, 64]],
                            use_xyz=True,
                            bn=with_bn
                    )
            )
            c_out_0 = 32 + 64

            c_in = c_out_0
            self.SA_modules.append(
                    PointnetSAModuleMSG(
                            npoint=256,
                            radii=[0.1, 0.2],
                            nsamples=[16, 32],
                            mlps=[[c_in, 64], [c_in, 128]],
                            use_xyz=True,
                            bn=with_bn
                    )
            )
            c_out_1 = 64 + 128
            c_in = c_out_1
            self.SA_modules.append(
                    PointnetSAModuleMSG(
                            npoint=64,
                            radii=[0.2, 0.4],
                            nsamples=[16, 32],
                            mlps=[[c_in, 128], [c_in, 256]],
                            use_xyz=True,
                            bn=with_bn
                    )
            )
            c_out_2 = 128 + 256

            c_in = c_out_2
            self.SA_modules.append(
                    PointnetSAModuleMSG(
                            npoint=16,
                            radii=[0.4, 0.8],
                            nsamples=[16, 32],
                            mlps=[[c_in, 512], [c_in, 512]],
                            use_xyz=True,
                            bn=with_bn
                    )
            )
            self.fc_lyaer = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.SiLU(),
                    nn.Linear(1024, self.dim_condition),
            )
        if "txt" in v_conf["condition"]:
            self.with_txt = True
            model_path = 'Alibaba-NLP/gte-large-en-v1.5'
            from sentence_transformers import SentenceTransformer
            self.txt_model = SentenceTransformer(model_path, trust_remote_code=True)
            for param in self.txt_model.parameters():
                param.requires_grad = False
            self.txt_model.eval()
            
            self.txt_fc = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, self.dim_condition),
            )

        self.classifier = nn.Sequential(
                nn.Linear(self.dim_input, self.dim_input),
                nn.LayerNorm(self.dim_input),
                nn.SiLU(),
                nn.Linear(self.dim_input, 1),
        )    
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule=v_conf["beta_schedule"],
            prediction_type=v_conf["diffusion_type"],
            beta_start=v_conf["beta_start"],
            beta_end=v_conf["beta_end"],
            variance_type=v_conf["variance_type"],
            clip_sample=False,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim_total, self.dim_total),
            nn.LayerNorm(self.dim_total),
            nn.SiLU(),
            nn.Linear(self.dim_total, self.dim_total),
        )

        self.num_max_faces = v_conf["num_max_faces"]
        self.loss = nn.functional.l1_loss if v_conf["loss"] == "l1" else nn.functional.mse_loss
        self.diffusion_type = v_conf["diffusion_type"]
        self.pad_method = v_conf["pad_method"]

        # self.ae_model = AutoEncoder_0925(v_conf)
        model_mod = importlib.import_module("diffusion.model")
        model_mod = getattr(model_mod, v_conf["autoencoder"])
        self.ae_model = model_mod(v_conf)

        self.is_pretrained = v_conf["autoencoder_weights"] is not None
        self.is_stored_z = v_conf["stored_z"]
        self.use_mean = v_conf["use_mean"]
        self.is_train_decoder = v_conf["train_decoder"]
        if self.is_pretrained:
            # checkpoint = torch.load(v_conf["autoencoder_weights"], weights_only=False)["state_dict"]
            repo_id = Path(v_conf["autoencoder_weights"]).parent.as_posix()
            model_name = Path(v_conf["autoencoder_weights"]).name
            model_weights = hf_hub_download(repo_id=repo_id, filename=model_name)
            checkpoint = torch.load(model_weights, weights_only=False)["state_dict"]
                
            weights = {k.replace("model.", ""): v for k, v in checkpoint.items()}
            self.ae_model.load_state_dict(weights)
        if not self.is_train_decoder:
            for param in self.ae_model.parameters():
                param.requires_grad = False
            self.ae_model.eval()

    def inference(self, bs, device, v_data=None, v_log=True, **kwargs):
        face_features = torch.randn((bs, self.num_max_faces, self.dim_input)).to(device)
        condition = None
        if self.with_img or self.with_pc or self.with_txt:
            condition = self.extract_condition(v_data)[:bs]
            # face_features = face_features[:condition.shape[0]]
        # error = []
        for t in tqdm(self.noise_scheduler.timesteps):
            timesteps = t.reshape(-1).to(device)
            pred_x0 = self.diffuse(face_features, timesteps, v_condition=condition)
            face_features = self.noise_scheduler.step(pred_x0, t, face_features).prev_sample
            # error.append((v_data["face_features"] - face_features).abs().mean(dim=[1,2]))

        face_z = face_features
        if self.pad_method == "zero":
            label = torch.sigmoid(self.classifier(face_features))[..., 0]
            mask = label > 0.5
        else:
            mask = torch.ones_like(face_z[:, :, 0]).to(bool)
        
        recon_data = []
        for i in range(bs):
            face_z_item = face_z[i:i + 1][mask[i:i + 1]]
            if self.addition_tag: # Deduplicate
                flag = face_z_item[...,-1] > 0
                face_z_item = face_z_item[flag][:, :-1]
            if self.pad_method == "random": # Deduplicate
                threshold = 1e-2
                max_faces = face_z_item.shape[0]
                index = torch.stack(torch.meshgrid(torch.arange(max_faces),torch.arange(max_faces), indexing="ij"), dim=2)
                features = face_z_item[index]
                distance = (features[:,:,0]-features[:,:,1]).abs().mean(dim=-1)
                final_face_z = []
                for j in range(max_faces):
                    valid = True
                    for k in final_face_z:
                        if distance[j,k] < threshold:
                            valid = False
                            break
                    if valid:
                        final_face_z.append(j)
                face_z_item = face_z_item[final_face_z]
            data_item = self.ae_model.inference(face_z_item)
            recon_data.append(data_item)
        return recon_data

    def get_z(self, v_data, v_test):
        data = {}
        if self.is_stored_z:
            face_features = v_data["face_features"]
            bs = face_features.shape[0]
            num_face = face_features.shape[1]
            mean = face_features[..., :32]
            std = face_features[..., 32:]
            if self.use_mean:
                face_features = mean
            else:
                face_features = mean + std * torch.randn_like(mean)
            data["padded_face_z"] = face_features.reshape(bs, num_face, -1)
        else:
            with torch.no_grad() and autocast(device_type='cuda', dtype=torch.float32):
                encoding_result = self.ae_model.encode(v_data, True)
                face_features, _, _ = self.ae_model.sample(encoding_result["face_features"], v_is_test=self.use_mean)
            dim_latent = face_features.shape[-1]
            num_faces = v_data["num_face_record"]
            bs = num_faces.shape[0]
            # Fill the face_z to the padded_face_z without forloop
            if self.pad_method == "zero":
                padded_face_z = torch.zeros(
                    (bs, self.num_max_faces, dim_latent), device=face_features.device, dtype=face_features.dtype)
                mask = num_faces[:, None] > torch.arange(self.num_max_faces, device=num_faces.device)
                padded_face_z[mask] = face_features
                data["padded_face_z"] = padded_face_z
                data["mask"] = mask
            else:
                positions = torch.arange(self.num_max_faces, device=face_features.device).unsqueeze(0).repeat(bs, 1)
                mandatory_mask = positions < num_faces[:,None]
                random_indices = (torch.rand((bs, self.num_max_faces), device=face_features.device) * num_faces[:,None]).long()
                indices = torch.where(mandatory_mask, positions, random_indices)
                num_faces_cum = num_faces.cumsum(dim=0).roll(1)
                num_faces_cum[0] = 0
                indices += num_faces_cum[:,None]
                # Permute the indices
                r_indices = torch.argsort(torch.rand((bs, self.num_max_faces), device=face_features.device), dim=1)
                indices = indices.gather(1, r_indices)
                data["padded_face_z"] = face_features[indices]
        return data

    def diffuse(self, v_feature, v_timesteps, v_condition=None):
        bs = v_feature.size(0)
        de = v_feature.device
        dt = v_feature.dtype
        time_embeds = self.time_embed(sincos_embedding(v_timesteps, self.dim_total)).unsqueeze(1)
        noise_features = self.p_embed(v_feature)
        v_condition = torch.zeros((bs, 1, self.dim_condition), device=de, dtype=dt) if v_condition is None else v_condition
        v_condition = v_condition.repeat(1, v_feature.shape[1], 1)
        noise_features = torch.cat([noise_features, v_condition], dim=-1)
        noise_features = noise_features + time_embeds

        pred_x0 = self.net1(noise_features)
        pred_x0 = self.fc_out(pred_x0)
        return pred_x0

    def extract_condition(self, v_data):
        condition = None
        if self.with_img:
            if "img_features" in v_data["conditions"]:
                img_feature = v_data["conditions"]["img_features"]
                num_imgs = img_feature.shape[1]
            else:
                imgs = v_data["conditions"]["imgs"]
                num_imgs = imgs.shape[1]
                imgs = imgs.reshape(-1, 3, 224, 224)
                img_feature = self.img_model(imgs)
            img_idx = v_data["conditions"]["img_id"]
            img_feature = self.img_fc(img_feature)
            if img_idx.shape[-1] > 1:
                camera_embedding = self.camera_embedding(img_idx)
                img_feature = (img_feature.reshape(-1, num_imgs, self.dim_condition) + camera_embedding).mean(dim=1)
            else:
                img_feature = (img_feature.reshape(-1, num_imgs, self.dim_condition)).mean(dim=1)
            condition = img_feature[:, None]
        elif self.with_pc:
            pc = v_data["conditions"]["points"]

            if pc.shape[2] > 2048:
                num_points = pc.shape[2]
                index = np.arange(num_points)
                np.random.shuffle(index)
                pc = pc[:, :, index[:2048]]
            assert pc.shape[2] == 2048
            points = pc[:, 0, :, :3]
            normals = pc[:, 0, :, 3:6]
            pc = torch.cat([points, normals], dim=-1)

            if self.is_aug and self.training:
            # if self.is_aug:
                # Rotate
                if True:
                    id_aug = v_data["id_aug"]
                    angles = torch.stack([id_aug % 4 * torch.pi / 2, id_aug // 4 % 4 * torch.pi / 2, id_aug // 16 * torch.pi / 2], dim=1)
                    matrix = (Rotation.from_euler('xyz', angles.cpu().numpy()).as_matrix())
                    rotation_3d_matrix = torch.tensor(matrix, device=pc.device, dtype=pc.dtype)

                    pc2 = (rotation_3d_matrix @ points.permute(0, 2, 1)).permute(0, 2, 1)
                    tpc2 = (rotation_3d_matrix @ (points+normals).permute(0, 2, 1)).permute(0, 2, 1)
                    normals2 = tpc2 - pc2
                    pc = torch.cat([pc2, normals2], dim=-1)
                
                # Crop
                if True:
                    bs = pc.shape[0]
                    num_points = pc.shape[1]
                    pc_index = torch.randint(0, pc.shape[1], (bs,), device=pc.device)
                    center_pos = torch.gather(pc, 1, pc_index[:, None, None].repeat(1, 1, 6))[...,:3]
                    length_xyz = torch.rand((bs,3), device=pc.device) * 1.0
                    bbox_min = center_pos - length_xyz[:, None, :]
                    bbox_max = center_pos + length_xyz[:, None, :]
                    mask = torch.logical_not(((pc[:, :, :3] > bbox_min) & (pc[:, :, :3] < bbox_max)).all(dim=-1))

                    sort_results = torch.sort(mask.long(),descending=True)
                    mask=sort_results.values
                    pc_sorted = torch.gather(pc,1,sort_results.indices[:,:,None].repeat(1,1,6))
                    num_valid = mask.sum(dim=-1)
                    index1 = torch.rand((bs,num_points), device=pc.device) * num_valid[:,None]
                    index2 = torch.arange(num_points, device=pc.device)[None].repeat(bs,1)
                    index = torch.where(mask.bool(), index2, index1)
                    pc = pc_sorted[torch.arange(bs)[:, None].repeat(1, num_points), index.long()]
                
                # Downsample
                if True:
                    num_points = pc.shape[1]
                    index = np.arange(num_points)
                    np.random.shuffle(index)
                    num_points = np.random.randint(1000, num_points)
                    # pc = pc[:,index[:2048]]
                    pc = pc[:,index[:num_points]]

                # Noise
                if True:
                    noise = torch.randn_like(pc) * 0.02
                    pc = pc + noise
                
                # Mask normal
                if True:
                    # pc[...,3:] = 0.
                    pc[...,3:] = 0. if torch.rand(1) > 0.5 else pc[...,3:]
            else:
                pc = pc

            if False:
                v_pc = pc.cpu().numpy()
                import open3d as o3d
                from pathlib import Path
                root = Path(r"D:/brepnet/noisy_input/111")
                for idx in range(v_pc.shape[0]):
                    prefix = v_data["v_prefix"][idx]
                    (root/prefix).mkdir(parents=True, exist_ok=True)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(v_pc[idx,:,:3])
                    o3d.io.write_point_cloud(str(root/prefix/f"{idx}_aug.ply"), pcd)

            l_xyz, l_features = [pc[:, :, :3].contiguous().float()], [pc.permute(0, 2, 1).contiguous().float()]
            with torch.autocast(device_type=pc.device.type, dtype=torch.float32):
                for i in range(len(self.SA_modules)):
                    li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                    l_xyz.append(li_xyz)
                    l_features.append(li_features)
                features = self.fc_lyaer(l_features[-1].mean(dim=-1))
                condition = features[:, None]
        elif self.with_txt:
            if "txt_features" in v_data["conditions"]:
                txt_feat = v_data["conditions"]["txt_features"]
            else:
                txt = v_data["conditions"]["txt"]
                txt_feat = self.txt_model.encode(txt, show_progress_bar=False, convert_to_numpy=False, device=self.txt_model.device)
                txt_feat = torch.stack(txt_feat, dim=0)
            condition = self.txt_fc(txt_feat)[:, None]
        return condition

    def forward(self, v_data, v_test=False, **kwargs):
        encoding_result = self.get_z(v_data, v_test)
        face_z = encoding_result["padded_face_z"]
        device = face_z.device
        bs = face_z.size(0)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        condition = self.extract_condition(v_data)
        noise = torch.randn(face_z.shape, device=device)
        noise_input = self.noise_scheduler.add_noise(face_z, noise, timesteps)

        # Model
        pred = self.diffuse(noise_input, timesteps, condition)
        
        loss = {}
        loss_item = self.loss(pred, face_z if self.diffusion_type == "sample" else noise, reduction="none")
        loss["diffusion_loss"] = loss_item.mean()
        if self.pad_method == "zero":
            mask = torch.logical_not((face_z.abs() < 1e-4).all(dim=-1))
            label = self.classifier(pred)
            classification_loss = nn.functional.binary_cross_entropy_with_logits(label[..., 0], mask.float())
            if self.loss == nn.functional.l1_loss:
                classification_loss = classification_loss * 1e-1
            else:
                classification_loss = classification_loss * 1e-4
            loss["classification"] = classification_loss
        loss["total_loss"] = sum(loss.values())
        loss["t"] = torch.stack((timesteps, loss_item.mean(dim=1).mean(dim=1)), dim=1)

        if self.is_train_decoder:
            raise
            pred_face_z = pred[encoding_result["mask"]]
            encoding_result["face_z"] = pred_face_z
            loss, recon_data = self.ae_model.loss(v_data, encoding_result)
            loss["l2"] = self.loss(pred, face_z)
            loss["diffusion_loss"] += loss["l2"]
        return loss


class Diffusion_condition_mm(Diffusion_condition):
    def __init__(self, v_conf, ):
        super().__init__(v_conf)
        layer = nn.TransformerEncoderLayer(
                d_model=self.dim_condition,
                nhead=8, norm_first=True, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.cross_attn = nn.TransformerEncoder(layer, 8, nn.LayerNorm(self.dim_condition))
        self.learned_uncond_emb = nn.Parameter(torch.rand(self.dim_condition))
        self.learned_svr_emb = nn.Parameter(torch.rand(self.dim_condition))
        self.learned_mvr_emb = nn.Parameter(torch.rand(self.dim_condition))
        self.learned_sketch_emb = nn.Parameter(torch.rand(self.dim_condition))
        self.learned_pc_emb = nn.Parameter(torch.rand(self.dim_condition))
        self.learned_txt_emb = nn.Parameter(torch.rand(self.dim_condition))
        self.condition = v_conf["condition"]
        assert len(self.condition) == 7 # uncond, svr, mvr, sketch, pc, txt, mm
        self.cond_prob = v_conf["cond_prob"]
        self.cond_prob_acc = np.cumsum(self.cond_prob)
        assert len(self.cond_prob) == len(self.condition)
        
    def inference(self, bs, device, v_data=None, v_log=True, **kwargs):
        face_features = torch.randn((bs, self.num_max_faces, self.dim_input)).to(device)
        condition = None
        if self.with_img or self.with_pc:
            condition, cond_nehot = self.extract_condition(v_data)
            condition = condition[:bs]
            # face_features = face_features[:condition.shape[0]]
        # error = []
        for t in tqdm(self.noise_scheduler.timesteps):
            timesteps = t.reshape(-1).to(device)
            pred_x0 = self.diffuse(face_features, timesteps, v_condition=condition)
            face_features = self.noise_scheduler.step(pred_x0, t, face_features).prev_sample
            # error.append((v_data["face_features"] - face_features).abs().mean(dim=[1,2]))

        face_z = face_features
        if self.pad_method == "zero":
            label = torch.sigmoid(self.classifier(face_features))[..., 0]
            mask = label > 0.5
        else:
            mask = torch.ones_like(face_z[:, :, 0]).to(bool)
        
        recon_data = []
        for i in range(bs):
            face_z_item = face_z[i:i + 1][mask[i:i + 1]]
            if self.addition_tag: # Deduplicate
                flag = face_z_item[...,-1] > 0
                face_z_item = face_z_item[flag][:, :-1]
            if self.pad_method == "random": # Deduplicate
                threshold = 1e-2
                max_faces = face_z_item.shape[0]
                index = torch.stack(torch.meshgrid(torch.arange(max_faces),torch.arange(max_faces), indexing="ij"), dim=2)
                features = face_z_item[index]
                distance = (features[:,:,0]-features[:,:,1]).abs().mean(dim=-1)
                final_face_z = []
                for j in range(max_faces):
                    valid = True
                    for k in final_face_z:
                        if distance[j,k] < threshold:
                            valid = False
                            break
                    if valid:
                        final_face_z.append(j)
                face_z_item = face_z_item[final_face_z]
            data_item = self.ae_model.inference(face_z_item)
            recon_data.append(data_item)
        return recon_data
        
    def diffuse(self, v_feature, v_timesteps, v_condition=None):
        bs = v_feature.size(0)
        de = v_feature.device
        dt = v_feature.dtype
        time_embeds = self.time_embed(sincos_embedding(v_timesteps, self.dim_total)).unsqueeze(1)
        noise_features = self.p_embed(v_feature)
        v_condition = torch.zeros((bs, 1, self.dim_condition), device=de, dtype=dt) if v_condition is None else v_condition
        v_condition = v_condition.repeat(1, v_feature.shape[1], 1)
        noise_features = torch.cat([noise_features, v_condition], dim=-1)
        noise_features = noise_features + time_embeds

        pred_x0 = self.net1(noise_features)
        pred_x0 = self.fc_out(pred_x0)
        return pred_x0

    def extract_condition(self, v_data):
        bs = len(v_data["v_prefix"])
        device = self.learned_uncond_emb.device
        sampled_prob = np.random.rand(bs)
        idx = self.cond_prob_acc.shape[0]-(sampled_prob[:,None] < self.cond_prob_acc[None,]).sum(axis=-1)
        cond_onehot = torch.zeros((bs, 5), device=device, dtype=bool) # svr, mvr, sketch, pc, txt
        cond_onehot[idx==1, 0] = 1 # svr
        cond_onehot[idx==2, 1] = 1 # mvr
        cond_onehot[idx==3, 2] = 1 # sketch
        cond_onehot[idx==4, 3] = 1 # pc
        cond_onehot[idx==5, 4] = 1 # txt
        num_mm = (idx==6).sum()
        rand_onehot = torch.rand((num_mm, 5), device=device) > 0.5
        cond_onehot[idx==6] = rand_onehot

        # Img feat
        if "img_features" in v_data["conditions"]:
            img_feature = v_data["conditions"]["img_features"]
            num_imgs = img_feature.shape[1]
        else:
            imgs = v_data["conditions"]["imgs"]
            num_imgs = imgs.shape[1]
            imgs = imgs.reshape(-1, 3, 224, 224)
            img_feature = self.img_model(imgs)
        img_idx = v_data["conditions"]["img_id"]
        img_feature = self.img_fc(img_feature)
        if img_idx.shape[-1] > 1:
            camera_embedding = self.camera_embedding(img_idx)
            img_feature = (img_feature.reshape(-1, num_imgs, self.dim_condition) + camera_embedding).mean(dim=1)
        else:
            img_feature = (img_feature.reshape(-1, num_imgs, self.dim_condition)).mean(dim=1)
        
        # PC feat
        pc = v_data["conditions"]["points"]
        if self.is_aug:
            # Rotate
            id_aug = v_data["id_aug"]
            angles = torch.stack([id_aug % 4 * torch.pi / 2, id_aug // 4 % 4 * torch.pi / 2, id_aug // 16 * torch.pi / 2], dim=1)
            matrix = (Rotation.from_euler('xyz', angles.cpu().numpy()).as_matrix())
            rotation_3d_matrix = torch.tensor(matrix, device=pc.device, dtype=pc.dtype)
            points = pc[:, 0, :, :3]
            normals = pc[:, 0, :, 3:6]
            
            pc2 = (rotation_3d_matrix @ points.permute(0, 2, 1)).permute(0, 2, 1)
            tpc2 = (rotation_3d_matrix @ (points+normals).permute(0, 2, 1)).permute(0, 2, 1)
            normals2 = tpc2 - pc2
            pc = torch.cat([pc2, normals2], dim=-1)
            
            # Crop
            bs = pc.shape[0]
            num_points = pc.shape[1]
            pc_index = torch.randint(0, pc.shape[1], (bs,), device=pc.device)
            center_pos = torch.gather(pc, 1, pc_index[:, None, None].repeat(1, 1, 6))[...,:3]
            length_xyz = torch.rand((bs,3), device=pc.device) * 1.0
            bbox_min = center_pos - length_xyz[:, None, :]
            bbox_max = center_pos + length_xyz[:, None, :]
            mask = torch.logical_not(((pc[:, :, :3] > bbox_min) & (pc[:, :, :3] < bbox_max)).all(dim=-1))
            
            sort_results = torch.sort(mask.long(),descending=True)
            mask=sort_results.values
            pc_sorted = torch.gather(pc,1,sort_results.indices[:,:,None].repeat(1,1,6))
            num_valid = mask.sum(dim=-1)
            index1 = torch.rand((bs,num_points), device=pc.device) * num_valid[:,None]
            index2 = torch.arange(num_points, device=pc.device)[None].repeat(bs,1)
            index = torch.where(mask.bool(), index2, index1)
            pc = pc_sorted[torch.arange(bs)[:, None].repeat(1, num_points), index.long()]
            
            # Downsample
            index = np.arange(num_points)
            np.random.shuffle(index)
            num_points = np.random.randint(1000, num_points)
            pc = pc[:,index[:num_points]]
            
            # Noise
            noise = torch.randn_like(pc) * 0.02
            pc = pc + noise
            
            # Mask normal
            pc[...,3:] = 0. if torch.rand(1) > 0.5 else pc[...,3:]
        else:
            pc = pc[:, 0]
        l_xyz, l_features = [pc[:, :, :3].contiguous().float()], [pc.permute(0, 2, 1).contiguous().float()]
        with torch.autocast(device_type=pc.device.type, dtype=torch.float32):
            for i in range(len(self.SA_modules)):
                li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                l_xyz.append(li_xyz)
                l_features.append(li_features)
            pc_features = self.fc_lyaer(l_features[-1].mean(dim=-1))
        pc_features = pc_features.to(img_feature.dtype)
        # TXT feat
        if "txt_features" in v_data["conditions"]:
            txt_feat = v_data["conditions"]["txt_features"]
        else:
            txt = v_data["conditions"]["txt"]
            txt_feat = self.txt_model.encode(txt, show_progress_bar=False, convert_to_numpy=False, device=self.txt_model.device)
            txt_feat = torch.stack(txt_feat, dim=0)
        txt_features = self.txt_fc(txt_feat)
        
        condition = torch.stack([
            self.learned_svr_emb, self.learned_mvr_emb, self.learned_sketch_emb, 
            self.learned_pc_emb, self.learned_txt_emb], dim=0)[None,:].repeat(bs,1,1).to(img_feature.dtype)
        
        condition[cond_onehot[:,0],0] = img_feature[cond_onehot[:,0]]
        condition[cond_onehot[:,1],1] = img_feature[cond_onehot[:,1]]
        condition[cond_onehot[:,2],2] = img_feature[cond_onehot[:,2]]
        condition[cond_onehot[:,3],3] = pc_features[cond_onehot[:,3]]
        condition[cond_onehot[:,4],4] = txt_features[cond_onehot[:,4]]
        
        condition = self.cross_attn(condition)
        condition = condition.mean(dim=1, keepdim=True)
        return condition, cond_onehot

    def forward(self, v_data, v_test=False, **kwargs):
        encoding_result = self.get_z(v_data, v_test)
        face_z = encoding_result["padded_face_z"]
        device = face_z.device
        bs = face_z.size(0)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        condition, cond_onehot = self.extract_condition(v_data)
        noise = torch.randn(face_z.shape, device=device)
        noise_input = self.noise_scheduler.add_noise(face_z, noise, timesteps)

        # Model
        pred = self.diffuse(noise_input, timesteps, condition)
        
        loss = {}
        loss_item = self.loss(pred, face_z if self.diffusion_type == "sample" else noise, reduction="none")
        loss["diffusion_loss"] = loss_item.mean()
        if self.pad_method == "zero":
            mask = torch.logical_not((face_z.abs() < 1e-4).all(dim=-1))
            label = self.classifier(pred)
            classification_loss = nn.functional.binary_cross_entropy_with_logits(label[..., 0], mask.float())
            if self.loss == nn.functional.l1_loss:
                classification_loss = classification_loss * 1e-1
            else:
                classification_loss = classification_loss * 1e-4
            loss["classification"] = classification_loss
        loss["total_loss"] = sum(loss.values())
        
        loss["t"] = torch.stack((timesteps, loss_item.mean(dim=1).mean(dim=1)), dim=1)
        loss_item = loss_item.mean(dim=1).mean(dim=1)
        uncond_mask = cond_onehot.sum(dim=1) == 0
        loss["uncond_count"] = uncond_mask.sum().to(loss_item.dtype)
        if uncond_mask.sum() > 0:
            loss["uncond_diffusion_loss"] = loss_item[uncond_mask].mean()
        mm_mask = cond_onehot.sum(dim=1) > 1
        loss["mm_count"] = mm_mask.sum().to(loss_item.dtype)
        if mm_mask.sum() > 0:
            loss["mm_diffusion_loss"] = loss_item[mm_mask].mean()
        svr_mask = torch.logical_and(cond_onehot[:,0], torch.logical_not(mm_mask))
        loss["svr_count"] = svr_mask.sum().to(loss_item.dtype)
        if svr_mask.sum() > 0:
            loss["svr_diffusion_loss"] = loss_item[svr_mask].mean()
        mvr_mask = torch.logical_and(cond_onehot[:,1], torch.logical_not(mm_mask))
        loss["mvr_count"] = mvr_mask.sum().to(loss_item.dtype)
        if mvr_mask.sum() > 0:
            loss["mvr_diffusion_loss"] = loss_item[mvr_mask].mean()
        sketch_mask = torch.logical_and(cond_onehot[:,2], torch.logical_not(mm_mask))
        loss["sketch_count"] = sketch_mask.sum().to(loss_item.dtype)
        if sketch_mask.sum() > 0:
            loss["sketch_diffusion_loss"] = loss_item[sketch_mask].mean()
        pc_mask = torch.logical_and(cond_onehot[:,3], torch.logical_not(mm_mask))
        loss["pc_count"] = pc_mask.sum().to(loss_item.dtype)
        if pc_mask.sum() > 0:
            loss["pc_diffusion_loss"] = loss_item[pc_mask].mean()
        txt_mask = torch.logical_and(cond_onehot[:,4], torch.logical_not(mm_mask))
        loss["txt_count"] = txt_mask.sum().to(loss_item.dtype)
        if txt_mask.sum() > 0:
            loss["txt_diffusion_loss"] = loss_item[txt_mask].mean()

        if self.is_train_decoder:
            raise
            pred_face_z = pred[encoding_result["mask"]]
            encoding_result["face_z"] = pred_face_z
            loss, recon_data = self.ae_model.loss(v_data, encoding_result)
            loss["l2"] = self.loss(pred, face_z)
            loss["diffusion_loss"] += loss["l2"]
        return loss


class Diffusion_condition_mvr(Diffusion_condition):
    def __init__(self, v_conf, ):
        super().__init__(v_conf)
        if "single_img" in v_conf["condition"] or "multi_img" in v_conf["condition"] or "sketch" in v_conf["condition"]:
            self.with_img = True
            self.img_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
            for param in self.img_model.parameters():
                param.requires_grad = False
            self.img_model.eval()

            self.img_fc1 = nn.Sequential(
                nn.Linear(1024, self.dim_condition),
                nn.LayerNorm(self.dim_condition),
                nn.SiLU(),
                nn.Linear(self.dim_condition, self.dim_condition),
            )
            self.img_fc2 = nn.Sequential(
                nn.Linear(1024, self.dim_condition),
                nn.LayerNorm(self.dim_condition),
                nn.SiLU(),
                nn.Linear(self.dim_condition, self.dim_condition),
            )

    def extract_condition(self, v_data):
        condition = None
        if self.with_img:
            if "img_features" in v_data["conditions"]:
                img_feature = v_data["conditions"]["img_features"]
                num_imgs = img_feature.shape[1]
            else:
                imgs = v_data["conditions"]["imgs"]
                num_imgs = imgs.shape[1]
                imgs = imgs.reshape(-1, 3, 224, 224)
                img_feature = self.img_model(imgs)

            img_feature_down = self.img_fc1(img_feature)
            img_feature_down = img_feature_down.reshape(-1, num_imgs, self.dim_condition)
            img_feature_cat = img_feature_down.reshape(-1, num_imgs * self.dim_condition)
            img_feature = self.img_fc2(img_feature_cat)
            condition = img_feature[:, None]
        elif self.with_pc:
            pc = v_data["conditions"]["points"]

            points = pc[:, 0, :, :3]
            normals = pc[:, 0, :, 3:6]
            pc = torch.cat([points, normals], dim=-1)

            if self.is_aug and self.training:
                # if self.is_aug:
                # Rotate
                if True:
                    id_aug = v_data["id_aug"]
                    angles = torch.stack(
                        [id_aug % 4 * torch.pi / 2, id_aug // 4 % 4 * torch.pi / 2, id_aug // 16 * torch.pi / 2], dim=1)
                    matrix = (Rotation.from_euler('xyz', angles.cpu().numpy()).as_matrix())
                    rotation_3d_matrix = torch.tensor(matrix, device=pc.device, dtype=pc.dtype)

                    pc2 = (rotation_3d_matrix @ points.permute(0, 2, 1)).permute(0, 2, 1)
                    tpc2 = (rotation_3d_matrix @ (points + normals).permute(0, 2, 1)).permute(0, 2, 1)
                    normals2 = tpc2 - pc2
                    pc = torch.cat([pc2, normals2], dim=-1)

                # Crop
                if True:
                    bs = pc.shape[0]
                    num_points = pc.shape[1]
                    pc_index = torch.randint(0, pc.shape[1], (bs,), device=pc.device)
                    center_pos = torch.gather(pc, 1, pc_index[:, None, None].repeat(1, 1, 6))[..., :3]
                    length_xyz = torch.rand((bs, 3), device=pc.device) * 1.0
                    bbox_min = center_pos - length_xyz[:, None, :]
                    bbox_max = center_pos + length_xyz[:, None, :]
                    mask = torch.logical_not(((pc[:, :, :3] > bbox_min) & (pc[:, :, :3] < bbox_max)).all(dim=-1))

                    sort_results = torch.sort(mask.long(), descending=True)
                    mask = sort_results.values
                    pc_sorted = torch.gather(pc, 1, sort_results.indices[:, :, None].repeat(1, 1, 6))
                    num_valid = mask.sum(dim=-1)
                    index1 = torch.rand((bs, num_points), device=pc.device) * num_valid[:, None]
                    index2 = torch.arange(num_points, device=pc.device)[None].repeat(bs, 1)
                    index = torch.where(mask.bool(), index2, index1)
                    pc = pc_sorted[torch.arange(bs)[:, None].repeat(1, num_points), index.long()]

                # Downsample
                if True:
                    num_points = pc.shape[1]
                    index = np.arange(num_points)
                    np.random.shuffle(index)
                    num_points = np.random.randint(1000, num_points)
                    # pc = pc[:,index[:2048]]
                    pc = pc[:, index[:num_points]]

                # Noise
                if True:
                    noise = torch.randn_like(pc) * 0.02
                    pc = pc + noise

                # Mask normal
                if True:
                    # pc[...,3:] = 0.
                    pc[..., 3:] = 0. if torch.rand(1) > 0.5 else pc[..., 3:]
            else:
                pc = pc

            if False:
                v_pc = pc.cpu().numpy()
                import open3d as o3d
                from pathlib import Path
                root = Path(r"D:/brepnet/noisy_input/111")
                for idx in range(v_pc.shape[0]):
                    prefix = v_data["v_prefix"][idx]
                    (root / prefix).mkdir(parents=True, exist_ok=True)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(v_pc[idx, :, :3])
                    o3d.io.write_point_cloud(str(root / prefix / f"{idx}_aug.ply"), pcd)

            l_xyz, l_features = [pc[:, :, :3].contiguous().float()], [pc.permute(0, 2, 1).contiguous().float()]
            with torch.autocast(device_type=pc.device.type, dtype=torch.float32):
                for i in range(len(self.SA_modules)):
                    li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                    l_xyz.append(li_xyz)
                    l_features.append(li_features)
                features = self.fc_lyaer(l_features[-1].mean(dim=-1))
                condition = features[:, None]
        elif self.with_txt:
            if "txt_features" in v_data["conditions"]:
                txt_feat = v_data["conditions"]["txt_features"]
            else:
                txt = v_data["conditions"]["txt"]
                txt_feat = self.txt_model.encode(txt, show_progress_bar=False, convert_to_numpy=False,
                                                 device=self.txt_model.device)
                txt_feat = torch.stack(txt_feat, dim=0)
            condition = self.txt_fc(txt_feat)[:, None]
        return condition

