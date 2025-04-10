import torch
import numpy as np
import torchvision.transforms as T

from pathlib import Path
from PIL import Image
from pathlib import Path
from app.DataProcessor.DataProcessor import DataProcessor

class ImageProcessor(DataProcessor):
    def _get_img_tensor(self, image_file: Path) -> torch.Tensor:
        """
        Return a (3, 224, 224) shape tensor
        """    
        transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        img = np.array(Image.open(Path(image_file)).convert("RGB"))
        img = transform(img).to(self._device)
        return img 