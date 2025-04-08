import torch
import numpy as np
import open3d as o3d
import torchvision.transforms as T
from PIL import Image
from pathlib import Path


class DataProcessor():
    NUM_PROPOSALS = 32
    PC_DOWNSAMPLE_NUM = 4096
    
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process(self, condition: str, files: list):
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