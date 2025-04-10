import os
import shutil
import subprocess
import random
import uuid
import gradio as gr

from pathlib import Path
from typing import Tuple

from app.inference import inference_batch_postprocess

# Should be refactored in the future
class UncondGeneratingMethod():
    def generate(self):
        def generate_uncond(seed, state: gr.BrowserState):
            try:
                state = check_user_output_dir(state)

                generate_output = Path(state['user_output_dir']) / 'unconditional'
                os.makedirs(generate_output, exist_ok=True)
                if len(os.listdir(generate_output)) > 0:
                    shutil.rmtree(generate_output)
                    os.makedirs(generate_output, exist_ok=True)

                # Get the generated model
                command = [
                    "python", "-m", "diffusion.train_diffusion",
                    "trainer.evaluate=true",
                    "trainer.batch_size=1000",
                    "trainer.gpu=1",
                    f"trainer.test_output_dir={generate_output.as_posix()}",
                    "trainer.resume_from_checkpoint=YuXingyao/HoLa-Brep/Diffusion_uncond_1100k.ckpt",
                    "trainer.num_worker=1",
                    "trainer.accelerator=\"32-true\"",
                    "trainer.exp_name=test",
                    "dataset.name=Dummy_dataset",
                    "dataset.length=32",
                    "dataset.num_max_faces=30",
                    "dataset.condition=None",
                    f"dataset.random_seed={seed}",
                    "model.name=Diffusion_condition",
                    "model.autoencoder_weights=YuXingyao/HoLa-Brep/AE_deepcad_1100k.ckpt",
                    "model.autoencoder=AutoEncoder_1119_light",
                    "model.with_intersection=true",
                    "model.in_channels=6",
                    "model.dim_shape=768",
                    "model.dim_latent=8",
                    "model.gaussian_weights=1e-6",
                    "model.pad_method=random",
                    "model.diffusion_latent=768",
                    "model.diffusion_type=epsilon",
                    "model.gaussian_weights=1e-6",
                    "model.condition=None",
                    "model.num_max_faces=30",
                    "model.beta_schedule=linear",
                    "model.addition_tag=false",
                    "model.name=Diffusion_condition"
                    ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "0"

                subprocess.run(command, check=True, env=env)

                # Postprocess the generated model
                postprocess_output = Path(state['user_output_dir']) / 'unconditional_post'
                os.makedirs(postprocess_output, exist_ok=True)
                if len(os.listdir(postprocess_output)) > 0:
                    shutil.rmtree(postprocess_output)
                    os.makedirs(postprocess_output, exist_ok=True)

                inference_batch_postprocess(
                    file_dir=generate_output.as_posix(),
                    output_dir=postprocess_output.as_posix(),
                    num_cpus=2,
                    drop_num=3
                )

                model_folders = pick_valid_model_randomly(postprocess_output, seed)
                for i, model_folder in enumerate(model_folders):
                    if model_folder is None:
                        if i == 0:
                            gr.Warning("No valid model is generated! Please try again.", title="Postprocess Error")
                        else:
                            gr.Warning(f"Only {i} valid model is generated! Please try again.", title="Postprocess Error")

                        return None, None, None, [], state
                    edge = (postprocess_output / model_folder / 'debug_face_loop' / 'edge.obj').as_posix()
                    solid = (postprocess_output / model_folder / 'recon_brep.stl').as_posix()
                    step = (postprocess_output / model_folder / 'recon_brep.step').as_posix()
                    state['Unconditional'][f'Model{i+1}'] = [edge, solid, step]
                    
                if 'Model1' in state['Unconditional'].keys():
                    return *state['Unconditional']['Model1'], state['Unconditional']['Model1'], state
                else:
                    return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
            except:
                gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
            
        return generate_uncond
        
def pick_valid_model_randomly(postprocess_output: Path, seed: int =0, num=4,) -> Tuple[Path, Path, Path]:
    output_folders = [model_folder for model_folder in os.listdir(postprocess_output) if 'success.txt' in os.listdir(postprocess_output / model_folder)]
    if len(output_folders) <= 0:
        return [None, None, None]
    elif len(output_folders) <= num:
        return [Path(output_folders[i]) for i in range(len(output_folders))] + [None for _ in range(num-len(output_folders))]
    else:
        random.seed(seed)
        return [Path(output_folders[i]) for i in random.sample(range(len(output_folders)), num)]    

def check_user_output_dir(state: gr.BrowserState):
    if state['user_id'] is None:
        state['user_id'] = uuid.uuid4()
    if state['user_output_dir'] is None:
        state['user_output_dir'] = f'./outputs/user_{str(state["user_id"])}'
    os.makedirs(state['user_output_dir'], exist_ok=True)
    return state 