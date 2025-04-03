import argparse
import sys
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
        
        # repo_id = Path(self._config["diffusion_weights"]).parent.as_posix()
        # model_name = Path(self._config["diffusion_weights"]).name
        # model_weights = hf_hub_download(repo_id=repo_id, filename=model_name)
        diffusion_weights = torch.load(self._config["diffusion_weights"], map_location=device, weights_only=False)["state_dict"]
        diffusion_weights = {k: v for k, v in diffusion_weights.items() if "ae_model" not in k}
        diffusion_weights = {k[6:]: v for k, v in diffusion_weights.items() if "model" in k}
        
        # AE_repo_id = Path(self._config["autoencoder_weights"]).parent.as_posix()
        # AE_model_name = Path(self._config["autoencoder_weights"]).name
        # AE_model_weights = hf_hub_download(repo_id=AE_repo_id, filename=AE_model_name)
        autoencoder_weights = torch.load(self._config["autoencoder_weights"], map_location=device, weights_only=False)["state_dict"]
        autoencoder_weights = {k[6:]: v for k, v in autoencoder_weights.items() if "model" in k}
        autoencoder_weights = {"ae_model."+k: v for k, v in autoencoder_weights.items()}
        
        diffusion_weights.update(autoencoder_weights)
        diffusion_weights = {k: v for k, v in diffusion_weights.items() if "camera_embedding" not in k}
        model.load_state_dict(diffusion_weights, strict=False)
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
    
    if not ray.is_initialized():
        ray.init(
            dashboard_host="0.0.0.0",
            dashboard_port=8080,
            num_cpus=num_cpus,
        )
    
    construct_brep_from_datanpz_ray = ray.remote(num_cpus=1, max_retries=0)(construct_brep_from_datanpz)
    
    all_folders = sorted(os.listdir(file_dir))
    
    tasks = []
    # for i, one_folder in enumerate(all_folders):
    #     construct_brep_from_datanpz(
    #         data_root=file_dir,
    #         out_root=output_dir,
    #         folder_name=one_folder,
    #         v_drop_num=drop_num,
    #         is_ray=False,
    #         is_optimize_geom=False,
    #         from_scratch=True,
    #         isdebug=True,
    #         is_save_data=True
    #         )
    
    for i, one_folder in enumerate(all_folders):
        tasks.append(
            construct_brep_from_datanpz_ray.remote(
                file_dir, 
                output_dir,
                one_folder,
                v_drop_num=drop_num,
                use_cuda=False, 
                from_scratch=True,
                is_log=False, 
                is_ray=True, 
                is_optimize_geom=False, 
                isdebug=True,
                is_save_data=True
            )
        )
        
    results = []
    success_count = 0
    for task in tqdm(tasks):
        try:
            results.append(ray.get(task, timeout=timeout))
            # Check whether the number of valid files is greater than 3
            for done_folder in os.listdir(output_dir):
                output_files = os.listdir(Path(output_dir) / Path(done_folder))
                if 'success.txt' in output_files:
                    success_count += 1
                    # ray.kill()
        except:
            results.append(None)
        if success_count >= 4:
            if ray.is_initialized():
                ray.shutdown()
            break
        else:
            success_count = 0

    print("Done.")
    

# Deprecated
# def setup_model(conf, device):
#     model = Diffusion_condition(conf)
#     diffusion_weights = torch.load(conf["diffusion_weights"], map_location=device, weights_only=False)["state_dict"]
#     diffusion_weights = {k: v for k, v in diffusion_weights.items() if "ae_model" not in k}
#     diffusion_weights = {k[6:]: v for k, v in diffusion_weights.items() if "model" in k}
#     autoencoder_weights = torch.load(conf["autoencoder_weights"], map_location=device, weights_only=False)["state_dict"]
#     autoencoder_weights = {k[6:]: v for k, v in autoencoder_weights.items() if "model" in k}
#     autoencoder_weights = {"ae_model."+k: v for k, v in autoencoder_weights.items()}
#     diffusion_weights.update(autoencoder_weights)
#     diffusion_weights = {k: v for k, v in diffusion_weights.items() if "camera_embedding" not in k}
#     model.load_state_dict(diffusion_weights, strict=False)
#     model.to(device)
#     model.eval()
#     return model


# def data_construction(config, device, fileitem):
#     NUM_PROPOSALS = 32
#     PC_NUM_SAMPLE = 4096
    
#     data = {
#         "conditions": {}
#         }
#     input_file = Path(fileitem)
#     if not input_file.exists():
#         print(f"File {input_file} not found.")
#         exit(1)
        
#     if "pc" in config["condition"]:
#         points_tensor = get_point_cloud_tensor(device, PC_NUM_SAMPLE, input_file)
#         data["conditions"]["points"] = points_tensor[None, None, :, :].repeat(NUM_PROPOSALS, 1, 1, 1)
#     elif "txt" in config["condition"]:
#         data["conditions"]["txt"] = [fileitem for item in range(NUM_PROPOSALS)]
#     elif "sketch" in config["condition"] or "svr" in config["condition"] or "mvr" in config["condition"]:
#         transform = T.Compose([
#                 T.ToPILImage(),
#                 T.Resize((224, 224)),
#                 T.ToTensor(),
#                 T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ])
#         img = np.array(Image.open(input_file).convert("RGB"))
#         img = transform(img).to(device)
#         img = img[None, None, :].repeat(NUM_PROPOSALS, 1, 1, 1, 1)
#         data["conditions"]["imgs"] = img
#         data["conditions"]["img_id"] = torch.tensor([[0]], device=device).repeat(NUM_PROPOSALS, 1)
#     else:
#         print("Unknown condition")
#         exit(1)
#     return data


# if __name__ == '__main__':
#     conf = {
#         "name": "Diffusion_condition",
#         "train_decoder": False,
#         "stored_z": False,
#         "use_mean": True,
#         "diffusion_latent": 768,
#         "diffusion_type": "epsilon",
#         "loss": "l2",
#         "pad_method": "random",
#         "num_max_faces": 30,
#         "beta_schedule": "squaredcos_cap_v2",
#         "beta_start": 0.0001,
#         "beta_end": 0.02,
#         "variance_type": "fixed_small",
#         "addition_tag": False,
#         "autoencoder": "AutoEncoder_1119_light",
#         "with_intersection": True,
#         "dim_latent": 8,
#         "dim_shape": 768,
#         "sigmoid": False,
#         "in_channels": 6,
#         "gaussian_weights": 1e-6,
#         "norm": "layer",
#         "autoencoder_weights": "",
#         "is_aug": False,
#         "condition": [],
#         "cond_prob": []
#     }
    
#     NUM_PROPOSALS = 32
    
#     parser = argparse.ArgumentParser(prog='Inference')
#     parser.add_argument('--autoencoder_weights', type=str, required=True)
#     parser.add_argument('--diffusion_weights', type=str, required=True)
#     parser.add_argument('--condition', nargs='+', required=True)
#     parser.add_argument('--input', nargs='+', type=str)
#     parser.add_argument('--output_dir', type=str, default="./inference_output")

#     args = parser.parse_args()
#     conf["autoencoder_weights"] = args.autoencoder_weights
#     conf["diffusion_weights"] = args.diffusion_weights
#     conf["condition"] = args.condition

#     # Model setup
#     seed_everything(int(time.time()))
#     torch.backends.cudnn.benchmark = False
#     torch.set_float32_matmul_precision("medium")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = setup_model(conf)

#     print("We have {} data".format(len(args.input)))
#     output_dir = Path(args.output_dir)
#     (output_dir / "network_pred").mkdir(parents=True, exist_ok=True)

#     # Inference
#     for id_item, fileitem in enumerate(tqdm(args.input)):
#         data = data_construction(conf, device, fileitem)

#         with torch.no_grad():
#             network_preds = model.inference(NUM_PROPOSALS, device, v_data=data, v_log=True)

#         for idx in range((len(network_preds))):
#             prefix = f"{name}_{idx:02d}"
#             (output_dir/"network_pred"/prefix).mkdir(parents=True, exist_ok=True)
#             recon_data = network_preds[idx]
#             export_edges(recon_data["pred_edge"], str(output_dir / "network_pred" / prefix / f"edge.obj"))
#             np.savez_compressed(str(output_dir / "network_pred" / prefix / f"data.npz"),
#                                 pred_face_adj_prob=recon_data["pred_face_adj_prob"],
#                                 pred_face_adj=recon_data["pred_face_adj"].cpu().numpy(),
#                                 pred_face=recon_data["pred_face"],
#                                 pred_edge=recon_data["pred_edge"],
#                                 pred_edge_face_connectivity=recon_data["pred_edge_face_connectivity"],
#                                 )
#     # Post processing
#     print("Start post processing")
#     num_cpus = 24
#     ray.init(
#         dashboard_host="0.0.0.0",
#         dashboard_port=8080,
#         num_cpus=num_cpus,
#     )
#     construct_brep_from_datanpz_ray = ray.remote(num_cpus=1, max_retries=0)(construct_brep_from_datanpz)

#     all_folders = os.listdir(output_dir / "network_pred")
#     all_folders.sort()

#     tasks = []
#     for i in range(len(all_folders)):
#         tasks.append(construct_brep_from_datanpz_ray.remote(
#             output_dir / "network_pred", output_dir/"after_post",
#             all_folders[i],
#             v_drop_num=2,
#             use_cuda=False, from_scratch=True,
#             is_log=False, is_ray=True, is_optimize_geom=True, isdebug=False,
#         ))
#     results = []
#     for i in tqdm(range(len(all_folders))):
#         try:
#             results.append(ray.get(tasks[i], timeout=60))
#         except:
#             results.append(None)
#     print("Done.")
