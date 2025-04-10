from diffusion.diffusion_model import Diffusion_condition_mvr
from app.ModelDirector import ModelDirector

    
class MVRDirector(ModelDirector):
    def get_ae_weights(self):
        return 'YuXingyao/HoLa-Brep/AE_deepcad_1100k.ckpt'
    
    def get_diffusion_weights(self):
        return 'YuXingyao/HoLa-Brep/Diffusion_mvr_sq30_800k.ckpt'
    
    def get_generating_condition(self):
        return 'mvr'
    def config_setup(self):
        # Bad smell, turly. Gonna refactor in the future... Hopefully...
        super().config_setup()
        self._builder.set_up_model_template(Diffusion_condition_mvr)
        