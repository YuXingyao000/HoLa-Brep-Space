import torch
import numpy as np
import open3d as o3d
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from abc import abstractmethod, ABC

class DataProcessor(ABC):
    NUM_PROPOSALS = 32
    
    def __init__(self, device=None):
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device
    
    def process(self, input_data):
        data = dict()
        data["conditions"] = self.process_input_data(input_data)
        return data
    
    @abstractmethod
    def process_input_data(self, input_data):
        pass
