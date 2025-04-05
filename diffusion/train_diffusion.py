import sys
from functools import partial
import importlib
from datetime import datetime
from pathlib import Path
import random
import time

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.GeomAbs import GeomAbs_C0
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt
from einops import rearrange
import numpy as np
import open3d as o3d
from huggingface_hub import hf_hub_download

from diffusion.dataset import Diffusion_dataset
from diffusion.utils import triangulate_shape, triangulate_face, export_edges

import os.path

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelSummary
from lightning_fabric import seed_everything

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam, AdamW
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAveragePrecision, BinaryF1Score
from torchmetrics import MetricCollection

import trimesh
from pytorch_lightning.loggers import WandbLogger

# os.environ["HTTP_PROXY"] = "http://172.31.178.6:7890"
# os.environ["HTTPS_PROXY"] = "http://172.31.178.6:7890"


def to_mesh(face_points):
    num_u_points, num_v_points = face_points.shape[1], face_points.shape[2]
    mesh_total = trimesh.Trimesh()
    for idx in range(face_points.shape[0]):
        uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
        for u_index in range(1, num_u_points + 1):
            for v_index in range(1, num_v_points + 1):
                pt = face_points[idx][u_index - 1, v_index - 1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)

        approx_face = GeomAPI_PointsToBSplineSurface(
                uv_points_array, 3, 8,
                GeomAbs_C0, 5e-2).Surface()

        v, f = triangulate_shape(BRepBuilderAPI_MakeFace(approx_face, 5e-2).Face())
        mesh_item = trimesh.Trimesh(vertices=v, faces=f)
        mesh_total += mesh_item
    return mesh_total

def seed_worker(worker_id, rank_id=0):
    random.seed(worker_id+rank_id*10000)
    np.random.seed(worker_id+rank_id*10000)
    torch.manual_seed(worker_id+rank_id*10000)

class TrainDiffusion(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainDiffusion, self).__init__()
        
        if "GLOBAL_RANK" in os.environ:
            seed_everything(os.environ["GLOBAL_RANK"], True)
            global_rank = os.environ['GLOBAL_RANK']
            print(f"{global_rank}: {torch.initial_seed()}")
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]

        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        dataset_mod = importlib.import_module("diffusion.dataset")
        self.dataset_mod = getattr(dataset_mod, self.hydra_conf["dataset"]["name"])

        model_mod = importlib.import_module("diffusion.diffusion_model")
        model_mod = getattr(model_mod, self.hydra_conf["model"]["name"])
        self.model = model_mod(self.hydra_conf["model"])

        self.viz = {
            "time_loss": []
        }

    def train_dataloader(self):
        self.train_dataset = self.dataset_mod("training", self.hydra_conf["dataset"], )

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.dataset_mod.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          drop_last=True,
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          worker_init_fn=partial(seed_worker, rank_id=self.trainer.global_rank),
                          )

    def val_dataloader(self):
        self.valid_dataset = self.dataset_mod("validation", self.hydra_conf["dataset"], )

        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=self.dataset_mod.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          worker_init_fn=partial(seed_worker, rank_id=self.trainer.global_rank),
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000], gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        data = batch

        loss = self.model(data)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss" or key == "t":
                continue
            self.log(f"Training_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Training_Loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)
        return total_loss

    def validation_step(self, batch, batch_idx):
        data = batch
        # bs = data["face_features"].shape[0]
        loss = self.model(data, v_test=True)
        total_loss = loss["total_loss"]
        for key in loss:
            if key in ["t", "total_loss", "cond_item", "cond_onehot"]:
                continue
            self.log(f"Validation_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Validation_Loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)
        
        if batch_idx == 0:
            self.viz = {
                "time_loss": []
            }
            self.viz["time_loss"].append(loss["t"])
        
        if batch_idx == 0 and self.global_rank == 0:
            recon_faces = self.model.inference(1, self.device, data)[0]["pred_face"]
            trimesh.PointCloud(recon_faces.reshape(-1, 3)).export(str(self.log_root / "{}_faces.ply".format(self.current_epoch)))
        
            if "conditions" in data:
                bs = len(data["v_prefix"])
                for idx in range(bs):
                    if "ori_imgs" in data["conditions"]:
                        img_id = data["conditions"]["img_id"]
                        imgs = data["conditions"]["ori_imgs"][idx].cpu().numpy().astype(np.uint8)
                        for i in range(imgs.shape[0]):
                            o3d.io.write_image(str(self.log_root / f"epoch{self.current_epoch}_item{idx}_img{img_id[i]}.png"), o3d.geometry.Image(imgs[i]))
                    elif "points" in data["conditions"]:
                        points = data["conditions"]["points"][idx].cpu().numpy().astype(np.float32)[0]
                        pc = o3d.geometry.PointCloud()
                        pc.points = o3d.utility.Vector3dVector(points[:,:3])
                        pc.normals = o3d.utility.Vector3dVector(points[:,3:])
                        o3d.io.write_point_cloud(str(self.log_root / f"epoch{self.current_epoch}_item{idx}_pc.ply"), pc)
                    elif "txt" in data["conditions"]:
                        open(self.log_root / "epoch{}_item{}_txt.txt".format(self.current_epoch, idx), "w").write(data["conditions"]["txt"][idx])

        return total_loss

    def on_validation_epoch_end(self):
        # if self.trainer.sanity_checking:
        #     return

        if "time_loss" in self.viz:
            time_loss = torch.cat(self.viz["time_loss"]).cpu().numpy()
            results = []
            for i in range(10):
                results.append(time_loss[np.logical_and(time_loss[:,0]>=i*100, time_loss[:,0]<(i+1)*100), 1].mean())
                self.log(f'tloss/{i}', results[-1], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if self.global_rank != 0:
            return
        self.viz = {"time_loss": []}
        return

    def test_dataloader(self):
        self.test_dataset = self.dataset_mod("testing", self.hydra_conf["dataset"], )

        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          collate_fn=self.dataset_mod.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          )

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            seed_everything(self.hydra_conf["dataset"]["random_seed"])
            # seed_everything(self.global_rank)
        data = batch
        batch_size = min(len(batch['v_prefix']), self.batch_size)
        # Test loss
        if len(data.keys()) != 1:
            loss = self.model(data, v_test=True)
            total_loss = loss["total_loss"]
            self.log("Test_Loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                    sync_dist=True, batch_size=self.batch_size)
        
        # Generation
        results = self.model.inference(batch_size, self.device, v_data=data, v_log=self.global_rank == 0)
        log_root = Path(self.hydra_conf["trainer"]["test_output_dir"])
        os.makedirs(log_root, exist_ok=True)
        for idx in range(batch_size):
            prefix = data["v_prefix"][idx]
            item_root = (log_root / prefix)
            item_root.mkdir(parents=True, exist_ok=True)
            recon_data = results[idx]

            # mesh = to_mesh(recon_data["pred_face"])
            # mesh.export(str(item_root / f"{prefix}_face.ply"))
            export_edges(recon_data["pred_edge"], str(item_root / f"{prefix}_edge.obj"))

            np.savez_compressed(str(item_root / f"data.npz"),
                                pred_face_adj_prob=recon_data["pred_face_adj_prob"],
                                pred_face_adj=recon_data["pred_face_adj"].cpu().numpy(),
                                pred_face=recon_data["pred_face"],
                                pred_edge=recon_data["pred_edge"],
                                pred_edge_face_connectivity=recon_data["pred_edge_face_connectivity"],
                                )

            if "conditions" in data:
                if "ori_imgs" in data["conditions"]:
                    img_id = data["conditions"]["img_id"][idx]
                    imgs = data["conditions"]["ori_imgs"][idx].cpu().numpy().astype(np.uint8)
                    for i in range(imgs.shape[0]):
                        o3d.io.write_image(str(item_root / f"{prefix}_img{img_id[i].cpu().item()}.png"), o3d.geometry.Image(imgs[i]))
                elif "points" in data["conditions"]:
                    points = data["conditions"]["points"][idx].cpu().numpy().astype(np.float32)[0]
                    pc = o3d.geometry.PointCloud()
                    pc.points = o3d.utility.Vector3dVector(points[:,:3])
                    pc.normals = o3d.utility.Vector3dVector(points[:,3:])
                    o3d.io.write_point_cloud(str(item_root / f"{prefix}_pc.ply"), pc)
                elif "txt" in data["conditions"]:
                    open(item_root / f"{prefix}_txt.txt", "w").write(data["conditions"]["txt"][idx])

    def on_test_epoch_end(self):
        for loss in self.trainer.callback_metrics:
            print("{}: {:.3f}".format(loss, self.trainer.callback_metrics[loss].cpu().item()))
        return


@hydra.main(config_name="train_diffusion.yaml", config_path="./", version_base="1.1")
def main(v_cfg: DictConfig):
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
    if "LOCAL_RANK" not in os.environ:
        print(OmegaConf.to_yaml(v_cfg))

    use_wandb = v_cfg["trainer"]["wandb"] if "wandb" in v_cfg["trainer"] else False
    exp_name = "Diffusion_" + v_cfg["trainer"]["exp_name"]
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    # log_dir = hydra_cfg['runtime']['output_dir'] + "/" + exp_name + "/" + str(datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
    log_dir = hydra_cfg['runtime']['output_dir'] + "/" + exp_name
    v_cfg["trainer"]["output"] = log_dir
    print("Log in {}".format(log_dir))
    if v_cfg["trainer"]["spawn"] is True:
        torch.multiprocessing.set_start_method("spawn")

    assert v_cfg["dataset"]["num_max_faces"] == v_cfg["model"]["num_max_faces"]
    assert v_cfg["dataset"]["pad_method"] == v_cfg["model"]["pad_method"]
    assert v_cfg["dataset"]["condition"] == v_cfg["model"]["condition"]
    # assert v_cfg["dataset"]["num_points"] == v_cfg["model"]["num_points"]

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor="Validation_Loss", save_last=True, every_n_train_steps=50000, save_top_k=-1))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    if v_cfg["trainer"]["swa"]:
        callbacks.append(StochasticWeightAveraging(swa_lrs=v_cfg["trainer"]["learning_rate"], swa_epoch_start=10))
    
    model = TrainDiffusion(v_cfg)
    if not v_cfg["trainer"]["evaluate"] and exp_name!="Diffusion_test" and use_wandb:
        logger = WandbLogger(
            project='BRepNet++',
            save_dir=log_dir,
            name=exp_name,
        )
        logger.watch(model)
    else:
        logger = TensorBoardLogger(log_dir)

    trainer = Trainer(
        default_root_dir=log_dir,
        logger=logger,
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_true" if v_cfg["trainer"].gpu > 1 else "auto",
        devices=v_cfg["trainer"].gpu,
        enable_model_summary=True,
        callbacks=callbacks,
        max_epochs=int(v_cfg["trainer"]["max_epochs"]),
        max_steps=int(v_cfg["trainer"]["max_steps"]),
        # max_epochs=2,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        precision=v_cfg["trainer"]["accelerator"],

        gradient_clip_algorithm="norm",
        gradient_clip_val=0.5,
    )
    seed_everything(int(v_cfg["dataset"]["random_seed"]))

    if v_cfg["trainer"].evaluate:
        print(f"Resuming from {v_cfg['trainer'].resume_from_checkpoint}")
        # try:
        # weights = torch.load(v_cfg["trainer"].resume_from_checkpoint, weights_only=False, map_location="cpu")
        # except:
        repo_id = Path(v_cfg["trainer"].resume_from_checkpoint).parent.as_posix()
        model_name = Path(v_cfg["trainer"].resume_from_checkpoint).name
        model_weights = hf_hub_download(repo_id=repo_id, filename=model_name)
        weights = torch.load(model_weights, weights_only=False, map_location="cpu")
        # print("Checkpoint state_dict keys:", weights["state_dict"].keys())
        # print("Model state_dict keys:", model.state_dict().keys())
        weights = weights["state_dict"]
        weights = {k: v for k, v in weights.items() if "ae_model" not in k}
        # weights = {k: v for k, v in weights.items() if "camera_embedding" not in k}
        # weights = {k.replace("model.", ""): v for k, v in weights.items()}
        model.load_state_dict(weights, strict=False)
        trainer.test(model)

    else:
        if v_cfg["trainer"].resume_from_checkpoint is not None and v_cfg["trainer"].resume_from_checkpoint != "none":
            print(f"Resuming from {v_cfg['trainer'].resume_from_checkpoint}")
            # model = TrainDiffusion.load_from_checkpoint(v_cfg["trainer"].resume_from_checkpoint)
            # model.hydra_conf = v_cfg
            weights = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
            # weights = {k.replace("model.", ""): v for k, v in weights.items()}
            model.load_state_dict(weights)
        trainer.fit(model)


if __name__ == '__main__':
    main()
