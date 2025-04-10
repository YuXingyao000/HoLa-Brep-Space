from app.ModelDirector import ModelDirector

class TxtDirector(ModelDirector):
    def get_ae_weights(self):
        return 'YuXingyao/HoLa-Brep/AE_deepcad_1100k.ckpt'
    
    def get_diffusion_weights(self):
        return 'YuXingyao/HoLa-Brep/Diffusion_txt_sq30_1000k.ckpt'
    
    def get_generating_condition(self):
        return 'txt'
    