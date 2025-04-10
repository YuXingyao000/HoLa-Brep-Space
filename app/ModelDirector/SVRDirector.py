from app.ModelDirector import ModelDirector

class SVRDirector(ModelDirector):
    def get_ae_weights(self):
        return 'YuXingyao/HoLa-Brep/AE_deepcad_1100k.ckpt'
    
    def get_diffusion_weights(self):
        return 'YuXingyao/HoLa-Brep/Diffusion_svr_sq30_1500k.ckpt'
    
    def get_generating_condition(self):
        return 'svr'
    