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
                
                gr.Info("Start diffusing", title="Runtime Info")
                subprocess.run(command, check=True, env=env)
                gr.Info("Finished diffusing", title="Runtime Info")
                
                # Post-process the generated model
                postprocess_output = Path(state['user_output_dir']) / 'unconditional_post'
                os.makedirs(postprocess_output, exist_ok=True)
                if len(os.listdir(postprocess_output)) > 0:
                    shutil.rmtree(postprocess_output)
                    os.makedirs(postprocess_output, exist_ok=True)

                gr.Info("Start post-processing.", title="Runtime Info")
                inference_batch_postprocess(
                    file_dir=generate_output.as_posix(),
                    output_dir=postprocess_output.as_posix(),
                    num_cpus=2,
                    drop_num=0
                )
                gr.Info("Finished post-processing!", title="Runtime Info")
                valid_models = get_valid_models(postprocess_output)
                
                # Should have valid outputs
                if len(valid_models) <= 0:
                    raise UncondGeneraingException("No Valid Model Generated!")
                
                # Update the user state
                state["uncond"] = list()
                for i, model_number in enumerate(valid_models):
                    edge = (postprocess_output / model_number / 'debug_face_loop' / 'edge.obj').as_posix() # Hard coding is not good.
                    solid = (postprocess_output / model_number / 'recon_brep.stl').as_posix()
                    step = (postprocess_output / model_number / 'recon_brep.step').as_posix()
                    state["uncond"].append([edge, solid, step])
                    
                gr.Info(f"{len(valid_models)} valid models generated!", title="Finished generating")
                
                edge_file = state["uncond"][0][0]
                solid_file = state["uncond"][0][1]
                step_file = state["uncond"][0][2]
                return  edge_file, solid_file, step_file, state["uncond"][0], state
            except UncondEmptyInputException as input_e:
                gr.Warning(str(input_e), title="Empty Input")
                
            except UncondGeneraingException as generating_e:
                gr.Warning(str(generating_e), title="No Valid Generation")
            
            except Exception as e:
                print(e)
                gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
            return gr.update(), gr.update(), gr.update(), gr.update(), state
         
        return generate_uncond
        
def get_valid_models(postprocess_output: Path) -> Tuple[Path, Path, Path]:
    output_folders = [model_folder for model_folder in os.listdir(postprocess_output) if 'success.txt' in os.listdir(postprocess_output / model_folder)]
    return output_folders  

def check_user_output_dir(state: gr.BrowserState):
    if state['user_id'] is None:
        state['user_id'] = uuid.uuid4()
    if state['user_output_dir'] is None:
        state['user_output_dir'] = f'./outputs/user_{str(state["user_id"])}'
    os.makedirs(state['user_output_dir'], exist_ok=True)
    return state 

class UncondGeneraingException(Exception):
    """Custom exception if generating failed."""
    pass

class UncondEmptyInputException(Exception):
    """Custom exception if the input is empty."""
    pass