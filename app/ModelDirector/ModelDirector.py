from typing import Optional, Callable
from abc import ABC, abstractmethod
from app import ModelBuilder

'''
Direct the ModelBuilder to build a model depending on the modal the user choose
'''
class ModelDirector(ABC):
    def __init__(
        self,
        builder: ModelBuilder = None,
        additional_setup_fn: Optional[Callable[['ModelBuilder'], None]] = None
        ):
        if builder is None: 
            self._builder = ModelBuilder()
        else:
            self._builder = builder
        self._additional_setup_fn = additional_setup_fn
        self._ae_weights = self.get_ae_weights()
        self._diffusion_weights = self.get_diffusion_weights()
        self._condition = self.get_generating_condition()

    def config_setup(self):
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






    

