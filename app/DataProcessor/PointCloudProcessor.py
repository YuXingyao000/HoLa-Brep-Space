import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from app.DataProcessor.DataProcessor import DataProcessor

'''
Raw Data should be a Pathlike or str path, accept file path only
'''
class PointCloudProcessor(DataProcessor):
    PC_DOWNSAMPLE_NUM = 4096
    def process_input_data(self, pc_file_path):
        points_tensor = self._get_point_cloud_tensor(Path(pc_file_path[0]))
        return {"points" : points_tensor[None, None, :, :].repeat(self.NUM_PROPOSALS, 1, 1, 1)}
    
    def _get_point_cloud_tensor(self, input_file: Path | str) -> torch.Tensor:
        # Read point cloud
        pcd = o3d.io.read_point_cloud(input_file)
        points = np.array(pcd.points)
        
        # Check normals
        if pcd.has_normals():
            normals = np.array(pcd.normals)
        else:
            normals = np.zeros_like(points)

        # Concatenate points and normals
        points = np.concatenate([self._normalize_points(points), normals], axis=1)

        # Downsample
        index = np.random.choice(points.shape[0], self.PC_DOWNSAMPLE_NUM, replace=False)
        points = points[index]

        return torch.tensor(points, dtype=torch.float32).to(self._device)
    
    def _normalize_points(self, points):
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        center = (bbox_min + bbox_max) / 2
        points -= center
        scale = np.max(bbox_max - bbox_min)
        points /= scale
        points *= 0.9 * 2
        return points