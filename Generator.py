
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
    model_folder1, model_folder2, model_folder3 = pick_valid_model_randomly(postprocess_output, seed)
    
    if model_folder1 is None:
        gr.Warning("No valid model is generated! Please try again.", title="Postprocess Error")
        return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File()
    edge1 = (generate_output / model_folder1 / 'edge.obj').as_posix()
    solid1 = (postprocess_output / model_folder1 / 'recon_brep.stl').as_posix()
    step1 = (postprocess_output / model_folder1 / 'recon_brep.step').as_posix()
    user_state[state_key]["Model1"] = (edge1, solid1, step1)
    
    if model_folder2 is None:
        gr.Warning("Only one valid model is generated! Please try again.", title="Postprocess Error")
        return edge1, solid1, step1, gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File()
    edge2 = (generate_output / model_folder2 / 'edge.obj').as_posix()
    solid2 = (postprocess_output / model_folder2 / 'recon_brep.stl').as_posix()
    step2 = (postprocess_output / model_folder2 / 'recon_brep.step').as_posix()
    user_state[state_key]["Model2"] = (edge2, solid2, step2)
    
    if model_folder3 is None:
        gr.Warning("Only two valid models are generated! Please try again.", title="Postprocess Error")
        return edge1, solid1, step1, edge2, solid2, step2, gr.Model3D(), gr.Model3D(), gr.File()
    edge3 = (generate_output / model_folder3 / 'edge.obj').as_posix()
    solid3 = (postprocess_output / model_folder3 / 'recon_brep.stl').as_posix()
    step3 = (postprocess_output / model_folder3 / 'recon_brep.step').as_posix()
    user_state[state_key]["Model3"] = (edge3, solid3, step3)
    
    return edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, user_state

def pick_valid_model_randomly(postprocess_output: Path, seed: int =0) -> Tuple[Path, Path, Path]:
    output_folders = [model_folder for model_folder in os.listdir(postprocess_output) if 'success.txt' in os.listdir(postprocess_output / model_folder)]
    if len(output_folders) <= 0:
        return [None, None, None]
    elif len(output_folders) <= 3:
        return [Path(output_folders[i]) for i in range(len(output_folders))] + [None for _ in range(3-len(output_folders))]
    else:
        random.seed(seed)
        return [Path(output_folders[i]) for i in random.sample(range(len(output_folders)), 3)]    

def conditioned_generate(files: list, condition: str, generate_output:Path | str, postprocess_output: Path | str, state: gr.BrowserState):
    generate_output = Path(generate_output)
    postprocess_output = Path(postprocess_output)
    
    # try:
    diffusion.inference.inference(
            condition=condition,
            input_files=[Path(file.name) for file in files],
            output_path=generate_output,
            seed=0
            )
    # except :
    #     gr.Warning("Generation Error, please try some other models", title="Generation Error")
    #     return gr.Model3D(), gr.Model3D(), gr.File(),gr.Model3D(), gr.Model3D(), gr.File(),gr.Model3D(), gr.Model3D(), gr.File()
    
    try:
        diffusion.inference.inference_batch_postprocess(
            file_dir=generate_output, 
            output_dir=postprocess_output, 
            num_cpus=2,
            drop_num=0,
            timeout=60
            )
    except:
        gr.Warning("Please try to generate some other models", title="Postprocessing Error")
        return gr.Model3D(), gr.Model3D(), gr.File(),gr.Model3D(), gr.Model3D(), gr.File(),gr.Model3D(), gr.Model3D(), gr.File()

    edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state = check_valid_and_get_return_models(postprocess_output, generate_output, condition, state)
    return edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state

class GenerateMethod(ABC):
    static_state = None
    def __init__(self, state: gr.BrowserState):
        if self.static_state is None:
            self.static_state = state
        
    @abstractmethod
    def get_generate_method(self)->Callable[[list, gr.BrowserState], Tuple[gr.Model3D, gr.Model3D, gr.Model3D, gr.BrowserState]]:
        pass


class UncondGenerateMethod(GenerateMethod):
    def get_generate_method(self):
        def generate_uncond(seed, state: gr.BrowserState):
            # try:
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
                    # "trainer.resume_from_checkpoint=D:/HoLa-Brep/ckpt/Diffusion_uncond_1100k.ckpt",
                    "trainer.resume_from_checkpoint=YuXingyao/HoLa-Brep/Diffusion_uncond_1100k.ckpt",
                    "trainer.num_worker=2",
                    "trainer.accelerator=\"32-true\"",
                    "trainer.exp_name=test",
                    "dataset.name=Dummy_dataset",
                    "dataset.length=32",
                    "dataset.num_max_faces=30",
                    "dataset.condition=None",
                    f"dataset.random_seed={seed}",
                    "model.name=Diffusion_condition",
                    # "model.autoencoder_weights=D:/HoLa-Brep/ckpt/AE_deepcad_1100k.ckpt",
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
                # try:
                subprocess.run(command, check=True, env=env)
                # except subprocess.CalledProcessError as e:
                #     gr.Warning(f"{e.stderr}. Please try some other models", title="Generation Error")
                #     return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state
                # except:
                #     gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
                #     return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state


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
                # command = [
                #     'python', '-m', 'construct_brep', 
                #     '--data_root', f'{generate_output.as_posix()}',
                #     '--out_root', f'{postprocess_output.as_posix()}',
                #     '--use_ray', 
                #     '--num_cpus', '2',
                #     '--drop_num', '1',
                #     '--from_scratch'
                # ]
                # # try:
                # subprocess.run(command,  check=True)
                # except subprocess.CalledProcessError as e:
                #     gr.Warning(f"{e.stderr}. Please try some other models", title="Postprocessing Error")
                #     return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state
                # except:
                #     gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
                #     return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state

                model_folder1, model_folder2, model_folder3 = pick_valid_model_randomly(postprocess_output, seed)
                if model_folder1 is None:
                    gr.Warning("No valid model is generated! Please try again.", title="Postprocess Error")
                    return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state
                edge1 = (generate_output / model_folder1 / f'{model_folder1}_edge.obj').as_posix()
                solid1 = (postprocess_output / model_folder1 / 'recon_brep.stl').as_posix()
                step1 = (postprocess_output / model_folder1 / 'recon_brep.step').as_posix()
                state['Unconditional']['Model1'] = (edge1, solid1, step1)
                
                if model_folder2 is None:
                    gr.Warning("Only one valid model is generated! Please try again.", title="Postprocess Error")
                    return edge1, solid1, step1, gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state
                edge2 = (generate_output / model_folder2 / f'{model_folder2}_edge.obj').as_posix()
                solid2 = (postprocess_output / model_folder2 / 'recon_brep.stl').as_posix()
                step2 = (postprocess_output / model_folder2 / 'recon_brep.step').as_posix()
                state['Unconditional']['Model2'] = (edge2, solid2, step2)


                if model_folder3 is None:
                    gr.Warning("Only two valid models are generated! Please try again.", title="Postprocess Error")
                    return edge1, solid1, step1, edge2, solid2, step2, gr.Model3D(), gr.Model3D(), gr.File()
                edge3 = (generate_output / model_folder3 / f'{model_folder3}_edge.obj').as_posix()
                solid3 = (postprocess_output / model_folder3 / 'recon_brep.stl').as_posix()
                step3 = (postprocess_output / model_folder3 / 'recon_brep.step').as_posix()
                state['Unconditional']['Model3'] = (edge3, solid3, step3)


                return edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state
            # except:
            #     gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
            #     return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state
            
        return generate_uncond
    
    
class PCGenerateMethod(GenerateMethod):
    def get_generate_method(self):
        def generate_pc(file, state: gr.BrowserState):
            # try:
                generate_output, postprocess_output, state = get_output_pathes(state, 'pc')
                edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state = conditioned_generate([file], 'pc', generate_output, postprocess_output, state)
                return edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state
            # except:
            #     gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
            #     return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state
                
        return generate_pc
     
     
class TxtGenerateMethod(GenerateMethod):
    def get_generate_method(self):
        def generate_txt(description, state: gr.BrowserState):
            try:
                generate_output, postprocess_output, state = get_output_pathes(state, 'txt')
                os.makedirs(Path(state['user_output_dir']) / 'tmp', exist_ok=True)
                with open(Path(state['user_output_dir']) / 'tmp' / 'description.txt', 'w') as file:
                    file.write(description)
                edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state = conditioned_generate([file], 'txt', generate_output, postprocess_output, state)
                return edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state
            except UnicodeEncodeError as uni_error:
                gr.Warning("We sincerely apologize, but we currently only support English.", title="UnicodeEncodeError")
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state
            except:
                gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state
        return generate_txt


class SketchGenerateMethod(GenerateMethod):
    def get_generate_method(self):
        def generate_pc(file, state: gr.BrowserState):
            try:
                generate_output, postprocess_output, state = get_output_pathes(state, 'sketch')
                edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state = conditioned_generate([file], 'sketch', generate_output, postprocess_output, state)
                return edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state
            except:
                gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state
        return generate_pc
    
    
class SVRGenerateMethod(GenerateMethod):
    def get_generate_method(self):
        def generate_svr(img, state: gr.BrowserState):
            try:
                generate_output, postprocess_output, state = get_output_pathes(state, 'svr')
                edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state = conditioned_generate([img], 'svr', generate_output, postprocess_output, state)
                return edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state
            except:
                gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state
        return generate_svr

    
class MVRGenerateMethod(GenerateMethod):
    def get_generate_method(self):
        def generate_mvr(img1, img2, img3, state: gr.BrowserState):
            try:
                generate_output, postprocess_output, state = get_output_pathes(state, 'mvr')
                edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state = conditioned_generate([img1, img2, img3], 'mvr', generate_output, postprocess_output, state)
                return edge1, solid1, step1, edge2, solid2, step2, edge3, solid3, step3, state
            except:
                gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
                return gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), gr.Model3D(), gr.Model3D(), gr.File(), state
        return generate_mvr
