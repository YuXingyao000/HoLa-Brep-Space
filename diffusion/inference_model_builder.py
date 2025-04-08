import torch
from pathlib import Path
from typing import Optional
from lightning_fabric import seed_everything
from huggingface_hub import hf_hub_download

from diffusion.diffusion_model import Diffusion_condition


class ModelBuilder():
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
        # Torch condition
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("medium")

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Basic model
        model = Diffusion_condition(self._config)
        
        # Load diffusion weights
        repo_id = Path(self._config["diffusion_weights"]).parent.as_posix()
        model_name = Path(self._config["diffusion_weights"]).name
        model_weights = hf_hub_download(repo_id=repo_id, filename=model_name)
        diffusion_weights = torch.load(model_weights, map_location=device, weights_only=False)["state_dict"]
        diffusion_weights = {k: v for k, v in diffusion_weights.items() if "ae_model" not in k}
        diffusion_weights = {k[6:]: v for k, v in diffusion_weights.items() if "model" in k}
        
        # Load Autoencoder weights
        AE_repo_id = Path(self._config["autoencoder_weights"]).parent.as_posix()
        AE_model_name = Path(self._config["autoencoder_weights"]).name
        AE_model_weights = hf_hub_download(repo_id=AE_repo_id, filename=AE_model_name)
        autoencoder_weights = torch.load(AE_model_weights, map_location=device, weights_only=False)["state_dict"]
        autoencoder_weights = {k[6:]: v for k, v in autoencoder_weights.items() if "model" in k}
        autoencoder_weights = {"ae_model."+k: v for k, v in autoencoder_weights.items()}
        
        # Combine ae with diffusor
        diffusion_weights.update(autoencoder_weights)
        diffusion_weights = {k: v for k, v in diffusion_weights.items() if "camera_embedding" not in k}
        
        model.load_state_dict(diffusion_weights, strict=False)
        model.to(device)
        model.eval()
        self._model = model
        return self._model
    
    def reset(self):
        self._config = self.__init_config()         
        
    def __init_config(self):
        # Basic model setup
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
    
    @property
    def model(self):
        model = self._model
        return model