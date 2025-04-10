import torch
import numpy as np
import torchvision.transforms as T

from pathlib import Path
from typing import Tuple
from app.DataProcessor.ImageProcessor import ImageProcessor

class MultiImageProcessor(ImageProcessor):
    def process_input_data(self, image_files: Tuple[str]):
        multi_imgs = None
        for one_imgage in image_files:            
                single_img = self._get_img_tensor(Path(one_imgage))[None, None, ...]
                if multi_imgs is None:
                    multi_imgs = single_img
                else:
                    multi_imgs = torch.cat((multi_imgs, single_img), axis=1)
        multi_imgs = multi_imgs.repeat(self.NUM_PROPOSALS, 1, 1, 1, 1)
        img_id = torch.tensor([list(range(len(image_files)))], device=self._device).repeat(self.NUM_PROPOSALS, 1)
        return {
            "imgs" : multi_imgs,
            "img_id" : img_id
        }