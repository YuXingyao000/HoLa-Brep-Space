from typing import Optional, Callable
from abc import ABC, abstractmethod
from diffusion.inference_model_builder import ModelBuilder

class ModelDirector(ABC):
    def __init__(
        self,
        builder: ModelBuilder,
        additional_setup_fn: Optional[Callable[['ModelBuilder'], None]] = None
        ):
        self._builder = builder
        self._additional_setup_fn = additional_setup_fn
        self._ae_weights = self.get_ae_weights()
        self._diffusion_weights = self.get_diffusion_weights()
        self._condition = self.get_generating_condition()

    def construct_model(self):
        self._builder.setup_autoencoder_weights(self._ae_weights)
        self._builder.setup_diffusion_weights(self._diffusion_weights)
        
        # User defined setup
        if self._additional_setup_fn:
            self._additional_setup_fn(self._builder)
        
        self._builder.setup_condition(self._condition)
    
    @property
    def buider(self):
        return self._builder
    
    @abstractmethod
    def get_ae_weights(self):
        pass
    
    @abstractmethod
    def get_diffusion_weights(self):
        pass
    
    @abstractmethod
    def get_generating_condition(self):
        pass

class TxtDirector(ModelDirector):
    def get_ae_weights(self):
        return 'YuXingyao/HoLa-Brep/AE_deepcad_1100k.ckpt'
    
    def get_diffusion_weights(self):
        return 'YuXingyao/HoLa-Brep/Diffusion_txt_sq30_1000k.ckpt'
    
    def get_generating_condition(self):
        return 'txt'

class PCDirector(ModelDirector):
    def get_ae_weights(self):
        return 'YuXingyao/HoLa-Brep/AE_deepcad_1100k.ckpt'
    
    def get_diffusion_weights(self):
        return 'YuXingyao/HoLa-Brep/Diffusion_pc_sq30_1600k.ckpt'
    
    def get_generating_condition(self):
        return 'pc'

class SketchDirector(ModelDirector):
    def get_ae_weights(self):
        return 'YuXingyao/HoLa-Brep/AE_deepcad_1100k.ckpt'
    
    def get_diffusion_weights(self):
        return 'YuXingyao/HoLa-Brep/Diffusion_sketch_sq30_1500k.ckpt'
    
    def get_generating_condition(self):
        return 'sketch'
    
class SVRDirector(ModelDirector):
    def get_ae_weights(self):
        return 'YuXingyao/HoLa-Brep/AE_deepcad_1100k.ckpt'
    
    def get_diffusion_weights(self):
        return 'YuXingyao/HoLa-Brep/Diffusion_svr_sq30_1500k.ckpt'
    
    def get_generating_condition(self):
        return 'svr'
    
class MVRDirector(ModelDirector):
    def get_ae_weights(self):
        return 'YuXingyao/HoLa-Brep/AE_deepcad_1100k.ckpt'
    
    def get_diffusion_weights(self):
        return 'YuXingyao/HoLa-Brep/Diffusion_mvr_sq30_800k.ckpt'
    
    def get_generating_condition(self):
        return 'mvr'