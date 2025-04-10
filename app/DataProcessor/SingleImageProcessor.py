import torch

from pathlib import Path
from app.DataProcessor.ImageProcessor import ImageProcessor

class SingleImageProcessor(ImageProcessor):
    def process_input_data(self, image_file : Path | str):
        img = self._get_img_tensor(Path(image_file[0]))
        img = img[None, None, :].repeat(self.NUM_PROPOSALS, 1, 1, 1, 1)
        img_id = torch.tensor([[0]], device=self._device).repeat(self.NUM_PROPOSALS, 1)
        return {
            "imgs" : img,
            "img_id" : img_id
        }