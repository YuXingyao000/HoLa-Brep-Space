from app.ModelDirector import ModelDirector

class PointCloudDirector(ModelDirector):
    def get_ae_weights(self):
        return 'YuXingyao/HoLa-Brep/AE_deepcad_1100k.ckpt'
    
    def get_diffusion_weights(self):
        return 'YuXingyao/HoLa-Brep/Diffusion_pc_sq30_1600k.ckpt'
    
    def get_generating_condition(self):
        return 'pc'
    