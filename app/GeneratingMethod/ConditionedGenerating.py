
import os
import shutil
import uuid
import torch
import gradio as gr
import numpy as np

from pathlib import Path

from diffusion.utils import export_edges
from construct_brep import construct_brep_from_datanpz
from app.DataProcessor import DataProcessor
from app.ModelDirector import ModelDirector

_EDGE_FILE = 0
_SOLID_FILE = 1
_STEP_FILE = 2

class ConditionedGeneratingMethod(): 
    def __init__(
        self, 
        model_building_director: ModelDirector, 
        dataprocessor: DataProcessor, 
        model_num_to_return: int,
        model_seed: int = 0
        ):
        self.director = model_building_director
        self.dataprocessor = dataprocessor
        self.model_num_to_return = model_num_to_return
        self.model_seed = model_seed
    
    def generate(self):
        def generating_method(browser_state: dict, *inputs):
            try: 
                # Some checks
                assert len(inputs) > 0
                self._user_state_check(browser_state)
                self._empty_input_check(inputs)
            
                # Inference device(also shouldn't appear here)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Process user input data
                # TODO: Refactor DataProcessor!!!
                tensor_data = self.dataprocessor.process(inputs)

                # Basic configuration of a model
                self.director.config_setup()
                model_builder = self.director.buider

                # Should be refactored in the future since picking an output folder is not the responsibility of a model
                diffusion_output_dir, postprocess_output_dir = self._get_output_dir(browser_state, self.director.get_generating_condition())
                model_builder.setup_output_dir(diffusion_output_dir)
                model_builder.setup_seed(self.model_seed)
                
                model_builder.make_model(device)
                model = model_builder.model

                #############
                # Inference #
                #############
                with torch.no_grad():
                    pred_results = model.inference(self.dataprocessor.NUM_PROPOSALS, device, v_data=tensor_data, v_log=True)

                # Save intermediate files for post-processing
                for i, result in enumerate(pred_results):
                    diffusion_output_subdir = diffusion_output_dir / f"00_{i:02d}"
                    diffusion_output_subdir.mkdir(parents=True, exist_ok=True)

                    export_edges(result["pred_edge"], (diffusion_output_subdir / "edge.obj").as_posix())

                    np.savez_compressed(
                        file                        = (diffusion_output_subdir / "data.npz").as_posix(),
                        pred_face_adj_prob          = result["pred_face_adj_prob"],
                        pred_face_adj               = result["pred_face_adj"].cpu().numpy(),
                        pred_face                   = result["pred_face"],
                        pred_edge                   = result["pred_edge"],
                        pred_edge_face_connectivity = result["pred_edge_face_connectivity"],
                    )
                
                ###################
                # Post-Processing #
                ###################
                diffusion_results = sorted(os.listdir(diffusion_output_dir))

                tasks = []
                for i, model_number in enumerate(diffusion_results):
                    construct_brep_from_datanpz(
                        data_root=diffusion_output_dir,
                        out_root=postprocess_output_dir,
                        folder_name=model_number,
                        v_drop_num=0,
                        is_ray=False,
                        is_optimize_geom=False,
                        from_scratch=True,
                        isdebug=True,
                        is_save_data=True
                        )
                    success_count = 0
                    for done_folder in list(postprocess_output_dir.iterdir()):
                        output_files = os.listdir(postprocess_output_dir / done_folder)
                        if 'success.txt' in output_files:
                            success_count += 1
                    if success_count >= self.model_num_to_return:
                        break
                
                valid_model_number = self._get_valid_model_number(
                    postprocess_output_dir,
                    num_to_pick=self.model_num_to_return
                )
                
                # Check if there's no valid output
                self._postprocess_output_check(valid_model_number)
                
                #####################
                # Update User State #
                #####################
                browser_state = self._update_user_state(browser_state, postprocess_output_dir, valid_model_number)
                
                gr.Warning(f"{len(valid_model_number)} valid models generated!", title="Finish generating")
                condition = self.director.get_generating_condition()
                
                # Return the first model as the default demonstration
                edge_file = browser_state[condition][0][_EDGE_FILE]
                solid_file = browser_state[condition][0][_SOLID_FILE]
                step_file = browser_state[condition][0][_STEP_FILE]
                
                return browser_state, edge_file, solid_file, step_file, browser_state[condition][0]
            
            except EmptyInputException as input_e:
                gr.Warning(str(input_e), title="Empty Input")
                
            except GeneraingException as generating_e:
                gr.Warning(str(generating_e), title="No Valid Generation")
                
            except UnicodeEncodeError as uni_error:
                gr.Warning("We sincerely apologize, but we currently only support English.", title="UnicodeEncodeError")
                
            except Exception as e:
                print(e)
                gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
                
            return browser_state, gr.update(), gr.update(), gr.update(), gr.update()

        return generating_method
    
    def _update_user_state(self, browser_state, postprocess_output_dir, valid_model_number):
        # Unstable. May be refactored in the future
        condition = self.director.get_generating_condition()
        browser_state[condition] = []
        for i, model_number in enumerate(valid_model_number):
            edge = (postprocess_output_dir / model_number / 'debug_face_loop' / 'edge.obj').as_posix()
            solid = (postprocess_output_dir / model_number / 'recon_brep.stl').as_posix()
            step = (postprocess_output_dir / model_number / 'recon_brep.step').as_posix()
            browser_state[condition].append([edge, solid, step])
        return browser_state
    
    def _postprocess_output_check(self, valid_model_number):
        if len(valid_model_number) <= 0:
            raise GeneraingException("No Valid Model Generated!")
    
    def _empty_input_check(self, inputs):
        for input_component in inputs:
            if input_component is None:
                raise EmptyInputException("Empty input exists!")
    
    def _user_state_check(self, state_dict):
        if state_dict['user_id'] is None:
            state_dict['user_id'] = uuid.uuid4()
        if state_dict['user_output_dir'] is None:
            state_dict['user_output_dir'] = f'./outputs/user_{str(state_dict["user_id"])}'
        os.makedirs(state_dict['user_output_dir'], exist_ok=True)
        
    def _get_valid_model_number(
            self, 
            postprocess_output: Path,
        ):
        # Get valid **model number** after postprocessing
        output_folders = [model_folder for model_folder in os.listdir(postprocess_output) if 'success.txt' in os.listdir(postprocess_output / model_folder)]
        return output_folders
    
    def _get_output_dir(self, state_dict, condition):
        # Create and clean the diffusion output directory
        diffusion_output_dir = Path(state_dict['user_output_dir']) / condition
        if len(os.listdir(diffusion_output_dir)) > 0:
            shutil.rmtree(diffusion_output_dir)
        os.makedirs(diffusion_output_dir, exist_ok=True)

        # Create and clean the postprocess output directory
        postprocess_output_dir = Path(state_dict['user_output_dir']) / f'{condition}_post'
        if len(os.listdir(postprocess_output_dir)) > 0:
            shutil.rmtree(postprocess_output_dir)
        os.makedirs(postprocess_output_dir, exist_ok=True)
        return diffusion_output_dir, postprocess_output_dir

class GeneraingException(Exception):
    """Custom exception if generating failed."""
    pass

class EmptyInputException(Exception):
    """Custom exception if the input is empty."""
    pass





# class PCGenerateMethod(GenerateMethod):
#     def get_generate_method(self):
#         def generate_pc(file, state: gr.BrowserState):
#             if file is None:
#                 return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
#             else:
#                 try:
#                     generate_output, postprocess_output, state = get_output_pathes(state, 'pc')
#                     state = conditioned_generate([file], 'pc', generate_output, postprocess_output, state)
#                     if 'Model1' in state['Point Cloud'].keys():
#                         return *state['Point Cloud']['Model1'], state['Point Cloud']['Model1'], state
#                     else:
#                         return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
#                 except Exception as e:
#                     print(e)
#                     gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
#                     return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
#         return generate_pc
     
     
# class TxtGenerateMethod(GenerateMethod):
#     def get_generate_method(self):
#         def generate_txt(description, state: gr.BrowserState):
#             if description is None or description == "":
#                 return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
#             try:
#                 generate_output, postprocess_output, state = get_output_pathes(state, 'txt')
#                 os.makedirs(Path(state['user_output_dir']) / 'tmp', exist_ok=True)
#                 with open(Path(state['user_output_dir']) / 'tmp' / 'description.txt', 'w') as file:
#                     file.write(description)
#                 state = conditioned_generate([Path(state['user_output_dir']) / 'tmp' / 'description.txt'], 'txt', generate_output, postprocess_output, state)
#             except UnicodeEncodeError as uni_error:
#                 gr.Warning("We sincerely apologize, but we currently only support English.", title="UnicodeEncodeError")
#             except:
#                 gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
#             if 'Model1' in state['Text'].keys():
#                 return *state['Text']['Model1'], state['Text']['Model1'], state
#             else:
#                 return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
        
#         return generate_txt


# class SketchGenerateMethod(GenerateMethod):
#     def get_generate_method(self):
#         def generate_sketch(file, state: gr.BrowserState):
#             if file is None:
#                 return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
#             try:
#                 generate_output, postprocess_output, state = get_output_pathes(state, 'sketch')
#                 state = conditioned_generate([Path(file)], 'sketch', generate_output, postprocess_output, state)
#             except Exception as e:
#                 gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
#             if 'Model1' in state['Sketch'].keys():
#                 return *state['Sketch']['Model1'], state['Sketch']['Model1'], state
#             else:
#                 return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
#         return generate_sketch
    
    
# class SVRGenerateMethod(GenerateMethod):
#     def get_generate_method(self):
#         def generate_svr(img, state: gr.BrowserState):
#             if img is None:
#                 return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
#             try:
#                 generate_output, postprocess_output, state = get_output_pathes(state, 'svr')
#                 state = conditioned_generate([Path(img)], 'svr', generate_output, postprocess_output, state)
#             except Exception as e:
#                 print(e)
#                 gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
#             if 'Model1' in state['SVR'].keys():
#                 return *state['SVR']['Model1'], state['SVR']['Model1'], state
#             else:
#                 return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
#         return generate_svr

    
# class MVRGenerateMethod(GenerateMethod):
#     def get_generate_method(self):
#         def generate_mvr(img1, img2, img3, img4, state: gr.BrowserState):
#             if img1 is None or img2 is None or img3 is None or img4 is None:
#                 return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
#             try:
#                 generate_output, postprocess_output, state = get_output_pathes(state, 'mvr')
#                 state = conditioned_generate([Path(img1), Path(img2), Path(img3), Path(img4)], 'mvr', generate_output, postprocess_output, state)
#             except Exception as e:
#                 print(e)
#                 gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
#             if 'Model1' in state['MVR'].keys():
#                 return *state['MVR']['Model1'], state['MVR']['Model1'], state
#             else:
#                 return gr.Model3D(), gr.Model3D(), gr.File(), gr.Files(), state
#         return generate_mvr
