
import os
from pathlib import Path
import shutil
import subprocess
import random
import uuid

from typing import Callable, Tuple
import gradio as gr
from abc import ABC, abstractmethod

import diffusion.inference
from diffusion.inference_model_builder import ModelBuilder
from diffusion.inference_model_director import *

def delegate_generate_method(radio_type: str, state: gr.BrowserState):
    method: GenerateMethod
    if radio_type == "Unconditional":
        method = UncondGenerateMethod(state)
    elif radio_type == 'Point Cloud':
        method = PCGenerateMethod(state)
    elif radio_type == 'Sketch':
        method = SketchGenerateMethod(state)
    elif radio_type == 'Text':
        method = TxtGenerateMethod(state)
    elif radio_type == 'SVR':
        method = SVRGenerateMethod(state)
    elif radio_type == 'MVR':
        method = MVRGenerateMethod(state)
    return method.get_generate_method()

def check_user_output_dir(state: gr.BrowserState):
    if state['user_id'] is None:
        state['user_id'] = uuid.uuid4()
    if state['user_output_dir'] is None:
        state['user_output_dir'] = f'./outputs/user_{str(state["user_id"])}'
    os.makedirs(state['user_output_dir'], exist_ok=True)
    return state 

def get_output_pathes(state: gr.BrowserState, condition: str):
    state = check_user_output_dir(state)
    
    generate_output = Path(state['user_output_dir']) / condition
    os.makedirs(generate_output, exist_ok=True)
    if len(os.listdir(generate_output)) > 0:
        shutil.rmtree(generate_output)
        os.makedirs(generate_output, exist_ok=True)
        
    postprocess_output = Path(state['user_output_dir']) / f'{condition}_post'
    os.makedirs(postprocess_output, exist_ok=True)
    if len(os.listdir(postprocess_output)) > 0:
        shutil.rmtree(postprocess_output)
        os.makedirs(postprocess_output, exist_ok=True)
    return generate_output, postprocess_output, state

def check_valid_and_get_return_models(postprocess_output, generate_output, condition: str, user_state: gr.BrowserState, seed=0):
    state_key = {
        'pc': "Point Cloud",
        'txt' : "Text",
        'sketch' : "Sketch",
        'svr' : "SVR",
        'mvr' : "MVR"
    }[condition]
    model_folders = pick_valid_model_randomly(postprocess_output, seed)
    user_state[state_key] = dict()
    for i, model_folder in enumerate(model_folders):
        if model_folder is None:
            if i == 0:
                gr.Warning("No valid model is generated! Please try again.", title="Postprocess Error")
            else:
                gr.Warning(f"Only {1} valid model is generated! Please try again.", title="Postprocess Error")  
            return user_state
        edge = (postprocess_output / model_folder / 'debug_face_loop' / 'edge.obj').as_posix()
        solid = (postprocess_output / model_folder / 'recon_brep.stl').as_posix()
        step = (postprocess_output / model_folder / 'recon_brep.step').as_posix()
        user_state[state_key][f"Model{i + 1}"] = [edge, solid, step]
    
    return user_state

def pick_valid_model_randomly(postprocess_output: Path, seed: int =0, num=4,) -> Tuple[Path, Path, Path]:
    output_folders = [model_folder for model_folder in os.listdir(postprocess_output) if 'success.txt' in os.listdir(postprocess_output / model_folder)]
    if len(output_folders) <= 0:
        return [None, None, None]
    elif len(output_folders) <= num:
        return [Path(output_folders[i]) for i in range(len(output_folders))] + [None for _ in range(num-len(output_folders))]
    else:
        random.seed(seed)
        return [Path(output_folders[i]) for i in random.sample(range(len(output_folders)), num)]    

def get_director(condition):
    if condition == "pc":
        return PCDirector(ModelBuilder())
    elif condition == "txt":
        return TxtDirector(ModelBuilder())
    elif condition == "sketch":
        return SketchDirector(ModelBuilder())
    elif condition == "svr":
        return SVRDirector(ModelBuilder())
    elif condition == "mvr":
        return MVRDirector(ModelBuilder())

def conditioned_generate(files: list, condition: str, generate_output:Path | str, postprocess_output: Path | str, state: gr.BrowserState):
    generate_output = Path(generate_output)
    postprocess_output = Path(postprocess_output)
    
    director = get_director(condition)
    
    diffusion.inference.inference(
            model_director=director,
            input_files=[Path(file) for file in files],
            output_path=generate_output,
            seed=0
            )
    
    diffusion.inference.inference_batch_postprocess(
        file_dir=generate_output, 
        output_dir=postprocess_output, 
        num_cpus=2,
        drop_num=0,
        timeout=60
        )


    state = check_valid_and_get_return_models(postprocess_output, generate_output, condition, state)
    return state


class GenerateMethod(ABC):
    static_state = None
    def __init__(self, state: gr.BrowserState):
        if self.static_state is None:
            self.static_state = state
        
    @abstractmethod
    def get_generate_method(self)->Callable[[list, gr.BrowserState], gr.BrowserState]:
        pass


class UncondGenerateMethod(GenerateMethod):
    def get_generate_method(self):
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

                diffusion.inference.inference_batch_postprocess(
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
        
        
class PCGenerateMethod(GenerateMethod):
    def get_generate_method(self):
        def generate_pc(file, state: gr.BrowserState):
            if file is None:
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
            else:
                try:
                    generate_output, postprocess_output, state = get_output_pathes(state, 'pc')
                    state = conditioned_generate([file], 'pc', generate_output, postprocess_output, state)
                    if 'Model1' in state['Point Cloud'].keys():
                        return *state['Point Cloud']['Model1'], state['Point Cloud']['Model1'], state
                    else:
                        return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
                except Exception as e:
                    print(e)
                    gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
                    return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
        return generate_pc
     
     
class TxtGenerateMethod(GenerateMethod):
    def get_generate_method(self):
        def generate_txt(description, state: gr.BrowserState):
            if description is None or description == "":
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
            try:
                generate_output, postprocess_output, state = get_output_pathes(state, 'txt')
                os.makedirs(Path(state['user_output_dir']) / 'tmp', exist_ok=True)
                with open(Path(state['user_output_dir']) / 'tmp' / 'description.txt', 'w') as file:
                    file.write(description)
                state = conditioned_generate([Path(state['user_output_dir']) / 'tmp' / 'description.txt'], 'txt', generate_output, postprocess_output, state)
            except UnicodeEncodeError as uni_error:
                gr.Warning("We sincerely apologize, but we currently only support English.", title="UnicodeEncodeError")
            except:
                gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
            if 'Model1' in state['Text'].keys():
                return *state['Text']['Model1'], state['Text']['Model1'], state
            else:
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
        
        return generate_txt


class SketchGenerateMethod(GenerateMethod):
    def get_generate_method(self):
        def generate_sketch(file, state: gr.BrowserState):
            if file is None:
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
            try:
                generate_output, postprocess_output, state = get_output_pathes(state, 'sketch')
                state = conditioned_generate([Path(file)], 'sketch', generate_output, postprocess_output, state)
            except Exception as e:
                gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
            if 'Model1' in state['Sketch'].keys():
                return *state['Sketch']['Model1'], state['Sketch']['Model1'], state
            else:
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
        return generate_sketch
    
    
class SVRGenerateMethod(GenerateMethod):
    def get_generate_method(self):
        def generate_svr(img, state: gr.BrowserState):
            if img is None:
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
            try:
                generate_output, postprocess_output, state = get_output_pathes(state, 'svr')
                state = conditioned_generate([Path(img)], 'svr', generate_output, postprocess_output, state)
            except Exception as e:
                print(e)
                gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
            if 'Model1' in state['SVR'].keys():
                return *state['SVR']['Model1'], state['SVR']['Model1'], state
            else:
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
        return generate_svr

    
class MVRGenerateMethod(GenerateMethod):
    def get_generate_method(self):
        def generate_mvr(img1, img2, img3, img4, state: gr.BrowserState):
            if img1 is None or img2 is None or img3 is None:
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
            try:
                generate_output, postprocess_output, state = get_output_pathes(state, 'mvr')
                state = conditioned_generate([Path(img1), Path(img2), Path(img3), Path(img4)], 'mvr', generate_output, postprocess_output, state)
            except Exception as e:
                print(e)
                gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
            if 'Model1' in state['MVR'].keys():
                return *state['MVR']['Model1'], state['MVR']['Model1'], state
            else:
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
        return generate_mvr
