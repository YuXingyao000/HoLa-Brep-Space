import argparse
import sys
import os
import os.path
from pathlib import Path
import numpy as np
import open3d as o3d
import ray
from PIL import Image
from tqdm import tqdm
import time
from typing import Optional
import gradio as gr

from diffusion.utils import export_edges
from diffusion.diffusion_model import Diffusion_condition
from construct_brep import construct_brep_from_datanpz

import torch
from lightning_fabric import seed_everything
from huggingface_hub import hf_hub_download
import torchvision.transforms as T

os.environ["HF_HOME"] = "/data/.huggingface"
os.environ["TORCH_HOME"] = "/data/.cache/torch"


class DataProcessor():
    NUM_PROPOSALS = 32
    PC_DOWNSAMPLE_NUM = 4096
    
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def data_preprocess(self, condition: str, files: list):
        data = {
            "conditions": {}
        }

        # Point-cloud-conditioned
        if condition == "pc":
            points_tensor = self.__get_point_cloud_tensor(Path(files[0]))
            data["conditions"]["points"] = points_tensor[None, None, :, :].repeat(self.NUM_PROPOSALS, 1, 1, 1)
        # Text-conditioned
        elif condition == "txt":
            with open(Path(files[0]), 'r') as f:
                data["conditions"]["txt"] = [f.read()] * self.NUM_PROPOSALS
        # Imgae-conditioned
        elif condition == "sketch" or condition == "svr":
            data["conditions"]["imgs"] = None
            img = self.__get_img_tensor(Path(files[0]))
            data["conditions"]["imgs"] = img
            data["conditions"]["img_id"] = torch.tensor([[0]], device=self._device).repeat(self.NUM_PROPOSALS, 1)
        elif condition == "mvr":
            data["conditions"]["imgs"] = None
            for file_path in files:            
                img = self.__get_img_tensor(Path(file_path))
                if data["conditions"]["imgs"] is None:
                    data["conditions"]["imgs"] = img
                else:
                    data["conditions"]["imgs"] = torch.cat((data["conditions"]["imgs"], img), axis=1)
            data["conditions"]["img_id"] = torch.tensor([[0, 1, 2, 3]], device=self._device).repeat(self.NUM_PROPOSALS, 1)

        return data
        
    def __get_img_tensor(self, input_file: Path | str) -> torch.Tensor:
        transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        img = np.array(Image.open(input_file).convert("RGB"))
        img = transform(img).to(self._device)
        img = img[None, None, :].repeat(self.NUM_PROPOSALS, 1, 1, 1, 1)
        return img 
            
    def __get_point_cloud_tensor(self, input_file: Path | str) -> torch.Tensor:
        # Read point cloud
        pcd = o3d.io.read_point_cloud(str(input_file))
        points = np.array(pcd.points)
        
        # Check normals
        if pcd.has_normals():
            normals = np.array(pcd.normals)
        else:
            normals = np.zeros_like(points)

        # Concatenate points and normals
        points = np.concatenate([self.__normalize_points(points), normals], axis=1)

        # Downsample
        index = np.random.choice(points.shape[0], self.PC_DOWNSAMPLE_NUM, replace=False)
        points = points[index]

        return torch.tensor(points, dtype=torch.float32).to(self._device)
    
    def __normalize_points(self, points):
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        center = (bbox_min + bbox_max) / 2
        points -= center
        scale = np.max(bbox_max - bbox_min)
        points /= scale
        points *= 0.9 * 2
        return points

class ModelWeightsFactory():
    CONDITION_DIFFUSOR = {
        'pc' : 'YuXingyao/HoLa-Brep/Diffusion_pc_sq30_1600k.ckpt',
        'txt' : 'YuXingyao/HoLa-Brep/Diffusion_txt_sq30_1000k.ckpt',
        'sketch' : 'YuXingyao/HoLa-Brep/Diffusion_sketch_sq30_1500k.ckpt',
        'svr' : 'YuXingyao/HoLa-Brep/Diffusion_svr_sq30_1500k.ckpt',
        'mvr' : 'YuXingyao/HoLa-Brep/Diffusion_mvr_sq30_800k.ckpt',
        'autoencoder' : 'YuXingyao/HoLa-Brep/AE_deepcad_1100k.ckpt'
    }
    
    def get_txt_weights(self):
        return self.CONDITION_DIFFUSOR['txt']
    
    def get_pc_weights(self):
        return self.CONDITION_DIFFUSOR['pc']
    
    def get_sketch_weights(self):
        return self.CONDITION_DIFFUSOR['sketch']

    def get_svr_weights(self):
        return self.CONDITION_DIFFUSOR['svr']
    
    def get_mvr_weights(self):
        return self.CONDITION_DIFFUSOR['mvr']
    
    def get_autoencoder_weights(self):
        return self.CONDITION_DIFFUSOR['autoencoder']

class InferenceModelBuilder():
    NUM_PROPOSALS = 32
    def __init__(self):
        self.reset()

    def setup_autoencoder_weights(self, weights_path: Path | str):
        self._config["autoencoder_weights"] = weights_path
        
    def setup_diffusion_weights(self, weights_path: Path | str):
        self._config["diffusion_weights"] = weights_path
        
    def setup_condition(self, condition: list[str] | str):
        if isinstance(condition, str):
            if condition == "svr" or condition == "mvr":
                condition = "single_img" if condition == "svr" else "multi_img"
            self._config["condition"] = [condition]
        else:
            self._config["condition"] += condition
       
    def setup_output_dir(self, output_dir: Path | str):
        self._config["output_dir"] = output_dir
    
    def setup_seed(self, seed: Optional[int] = None):
        if seed is not None:
            seed_everything(seed)
        else:
            seed_everything(0)

    def make_model(self, device: Optional[torch.device] = None):
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("medium")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        model = Diffusion_condition(self._config)
        
        # diffusion_weights = torch.load(self._config["diffusion_weights"], map_location=device, weights_only=False)["state_dict"]
        repo_id = Path(self._config["diffusion_weights"]).parent.as_posix()
        model_name = Path(self._config["diffusion_weights"]).name
        model_weights = hf_hub_download(repo_id=repo_id, filename=model_name)
        diffusion_weights = torch.load(model_weights, map_location=device, weights_only=False)["state_dict"]
        diffusion_weights = {k: v for k, v in diffusion_weights.items() if "ae_model" not in k}
        diffusion_weights = {k[6:]: v for k, v in diffusion_weights.items() if "model" in k}
        
        # autoencoder_weights = torch.load(self._config["autoencoder_weights"], map_location=device, weights_only=False)["state_dict"]
        AE_repo_id = Path(self._config["autoencoder_weights"]).parent.as_posix()
        AE_model_name = Path(self._config["autoencoder_weights"]).name
        AE_model_weights = hf_hub_download(repo_id=AE_repo_id, filename=AE_model_name)
        autoencoder_weights = torch.load(AE_model_weights, map_location=device, weights_only=False)["state_dict"]
        autoencoder_weights = {k[6:]: v for k, v in autoencoder_weights.items() if "model" in k}
        autoencoder_weights = {"ae_model."+k: v for k, v in autoencoder_weights.items()}
        
        diffusion_weights.update(autoencoder_weights)
        diffusion_weights = {k: v for k, v in diffusion_weights.items() if "camera_embedding" not in k}
        model.load_state_dict(diffusion_weights, strict=True)
        model.to(device)
        model.eval()
        self._model = model
    
    def reset(self):
        self._config = self.__init_config()
           
    @property
    def model(self):
        model = self._model
        return model
        
    def __init_config(self):
        return {
        "name": "Diffusion_condition",
        "train_decoder": False,
        "stored_z": False,
        "use_mean": True,
        "diffusion_latent": 768,
        "diffusion_type": "epsilon",
        "loss": "l2",
        "pad_method": "random",
        "num_max_faces": 30,
        "beta_schedule": "squaredcos_cap_v2",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "variance_type": "fixed_small",
        "addition_tag": False,
        "autoencoder": "AutoEncoder_1119_light",
        "with_intersection": True,
        "dim_latent": 8,
        "dim_shape": 768,
        "sigmoid": False,
        "in_channels": 6,
        "gaussian_weights": 1e-6,
        "norm": "layer",
        "autoencoder_weights": "",
        "is_aug": False,
        "condition": [],
        "cond_prob": []
    }

class InferenceModelDirector():
    def __init__(self, builder: Optional[InferenceModelBuilder] = None):
        self._builder = builder
        self._factory = ModelWeightsFactory()
        
    @property
    def builder(self):
        return self._builder
    
    @builder.setter
    def builder(self, builder: InferenceModelBuilder):
        self._builder = builder
    
    def construct_txt_model(self):
        self._builder.setup_autoencoder_weights(self._factory.get_autoencoder_weights())
        self._builder.setup_diffusion_weights(self._factory.get_txt_weights())
        self._builder.setup_condition("txt")
        
    def construct_pc_model(self):
        self._builder.setup_autoencoder_weights(self._factory.get_autoencoder_weights())
        self._builder.setup_diffusion_weights(self._factory.get_pc_weights())
        self._builder.setup_condition("pc")
    
    def construct_sketch_model(self):
        self._builder.setup_autoencoder_weights(self._factory.get_autoencoder_weights())
        self._builder.setup_diffusion_weights(self._factory.get_sketch_weights())
        self._builder.setup_condition("sketch")
        
    def construct_svr_model(self):
        self._builder.setup_autoencoder_weights(self._factory.get_autoencoder_weights())
        self._builder.setup_diffusion_weights(self._factory.get_svr_weights())
        self._builder.setup_condition("svr")
    
    def construct_mvr_model(self):
        self._builder.setup_autoencoder_weights(self._factory.get_autoencoder_weights())
        self._builder.setup_diffusion_weights(self._factory.get_mvr_weights())
        self._builder.setup_condition("mvr")
    
def direct_model_builder(condition) -> InferenceModelBuilder:
    model_builder = InferenceModelBuilder()
    model_director = InferenceModelDirector(model_builder)
    if condition == "txt":
        model_director.construct_txt_model()
    elif condition == "pc":
        model_director.construct_pc_model()
    elif condition == "sketch":
        model_director.construct_sketch_model()
    elif condition == "svr":
        model_director.construct_svr_model()
    elif condition == "mvr":
        model_director.construct_mvr_model()
    else:
        raise Exception("Inference: Unknown condition")
    return model_director.builder

def inference(condition: str, input_files: list[Path | str], output_path: Path, seed: Optional[int]=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_builder = direct_model_builder(condition)
    model_builder.setup_output_dir(output_path)
    if seed is not None:
        model_builder.setup_seed(seed)
    
    model_builder.make_model(device)
    model = model_builder.model
    
    # print(input_files)
    # for file in input_files:
    #     with open(file, 'r') as f:
    #         print(f.readlines())
            
    
    data = DataProcessor().data_preprocess(condition, input_files)
    with torch.no_grad():
        pred_results = model.inference(DataProcessor.NUM_PROPOSALS, device, v_data=data, v_log=True)
    for i, result in enumerate(pred_results):
        output_dir = output_path / f"00_{i:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        export_edges(result["pred_edge"], (output_dir / "edge.obj").as_posix())
        
        np.savez_compressed(
            file                        = (output_dir / "data.npz").as_posix(),
            pred_face_adj_prob          = result["pred_face_adj_prob"],
            pred_face_adj               = result["pred_face_adj"].cpu().numpy(),
            pred_face                   = result["pred_face"],
            pred_edge                   = result["pred_edge"],
            pred_edge_face_connectivity = result["pred_edge_face_connectivity"],
        )

def inference_batch_postprocess(file_dir: Path ,output_dir: Path, num_cpus: int=4, drop_num: int=2, timeout: int=60):
    print("Start post processing")
    
    construct_brep_from_datanpz_ray = ray.remote(num_cpus=1, max_retries=0)(construct_brep_from_datanpz)
    
    all_folders = sorted(os.listdir(file_dir))
    
    tasks = []
    for i, one_folder in enumerate(all_folders):
        construct_brep_from_datanpz(
            data_root=file_dir,
            out_root=output_dir,
            folder_name=one_folder,
            v_drop_num=drop_num,
            is_ray=False,
            is_optimize_geom=False,
            from_scratch=True,
            isdebug=True,
            is_save_data=True
            )
        success_count = 0
        for done_folder in os.listdir(output_dir):
            output_files = os.listdir(Path(output_dir) / Path(done_folder))
            if 'success.txt' in output_files:
                success_count += 1
        if success_count >= 4:
            break
    print("Done.")
