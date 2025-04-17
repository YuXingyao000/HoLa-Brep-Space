import os
import shutil
import uuid
import torch
import gradio as gr
import numpy as np
import ray
import time

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
        model_seed: int = 0,
        output_main_dir: Path | str = Path('./outputs')
        ):
        self.director = model_building_director
        self.dataprocessor = dataprocessor
        self.model_num_to_return = model_num_to_return
        self.model_seed = model_seed
        self.output_main_dir = output_main_dir
    
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
                tensor_data = self.dataprocessor.process(inputs)

                # Basic configuration of a model
                self.director.config_setup()
                model_builder = self.director.buider

                # Should be refactored in the future since picking an output folder is not the responsibility of a model
                diffusion_output_dir = self._get_diffusion_output_dir(browser_state, self.director.get_generating_condition())
                postprocess_output_dir = self._get_postprocess_output_dir(browser_state, self.director.get_generating_condition())
                
                model_builder.setup_output_dir(diffusion_output_dir)
                model_builder.setup_seed(self.model_seed)
                
                model_builder.make_model(device)
                model = model_builder.model

                #############
                # Inference #
                #############
                gr.Info("Start diffusing", title="Runtime Info")
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
                gr.Info("Finished diffusing", title="Runtime Info")
                
                ###################
                # Post-Processing #
                ###################
                # Multi-thread preparation
                gr.Info("Start post-processing!", title="Runtime Info")
                if not ray.is_initialized():
                    ray.init(
                        dashboard_host="0.0.0.0",
                        dashboard_port=8080,
                        num_cpus=2,
                    )
                
                construct_brep_from_datanpz_ray = ray.remote(num_cpus=1, max_retries=0)(construct_brep_from_datanpz)
                diffusion_results = sorted(os.listdir(diffusion_output_dir))

                tasks = [
                    construct_brep_from_datanpz_ray.remote(
                        data_root=diffusion_output_dir,
                        out_root=postprocess_output_dir,
                        folder_name=model_number,
                        v_drop_num=0,
                        use_cuda=False, 
                        from_scratch=True,
                        is_log=False, 
                        is_ray=True, 
                        is_optimize_geom=False, 
                        isdebug=True,
                        is_save_data=True
                    )
                    for model_number in diffusion_results
                ]

                results = []
                success_count = 0
                while tasks and success_count < self.model_num_to_return:
                    done_ids, tasks = ray.wait(tasks, num_returns=1, timeout=60)
                    for done_id in done_ids:
                        try:
                            result = ray.get(done_id)
                            results.append(result)

                            # Delay just a bit to ensure file handles are released
                            time.sleep(0.2)

                            # Check for 'success.txt' in output folders
                            for done_folder in postprocess_output_dir.iterdir():
                                output_files = os.listdir(done_folder)
                                if 'success.txt' in output_files:
                                    success_count += 1

                        except Exception as e:
                            print(f"Task failed or timed out: {e}")
                            results.append(None)

                        if success_count >= self.model_num_to_return:
                            # Make sure the files are written successfully
                            time.sleep(5.0)
                            break

                gr.Info("Finished post-processing!", title="Runtime Info")
                # Get valid model serial numbers
                valid_models = self._get_valid_models(postprocess_output_dir)
                
                #####################
                # Update User State #
                #####################
                browser_state = self._update_user_state(browser_state, postprocess_output_dir, valid_models)
                
                # Check if there's no valid output
                self._postprocess_output_check(valid_models)
                
                
                # Multi-thread processing may return valid models more than 4 
                gr.Info(f"{len(valid_models) if len(valid_models) < 4 else 4} valid models generated!", title="Finish generating")
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
                gr.Warning("We sincerely apologize, but we currently only support English.", title="English Support Only")
                
            except FileNotFoundError as file_e:
                gr.Warning("The operation is too frequent!", title="Frequent Operation")
                
            except Exception as e:
                print(e)
                gr.Warning("Something bad happened. Please try some other models", title="Unknown Error")
                
            return browser_state, gr.update(), gr.update(), gr.update(), gr.update()

        return generating_method
    
    def _update_user_state(self, browser_state, postprocess_output_dir, valid_model):
        # Unstable. May be refactored in the future
        condition = self.director.get_generating_condition()
        browser_state[condition] = list()
        for i, model_number in enumerate(valid_model):
            edge = (postprocess_output_dir / model_number / 'debug_face_loop' / 'edge.obj').as_posix() # Hard coding is not good.
            solid = (postprocess_output_dir / model_number / 'recon_brep.stl').as_posix()
            step = (postprocess_output_dir / model_number / 'recon_brep.step').as_posix()
            browser_state[condition].append([edge, solid, step])
        return browser_state
    
    def _postprocess_output_check(self, valid_model):
        if len(valid_model) <= 0:
            raise GeneraingException("No Valid Model Generated!")
    
    def _empty_input_check(self, inputs):
        for input_component in inputs:
            if input_component is None:
                raise EmptyInputException("Empty input exists!")
    
    def _user_state_check(self, state_dict):
        if state_dict['user_id'] is None:
            state_dict['user_id'] = uuid.uuid4()
        if state_dict['user_output_dir'] is None:
            state_dict['user_output_dir'] = Path(self.output_main_dir) / f"user_{state_dict['user_id']}"
        os.makedirs(state_dict['user_output_dir'], exist_ok=True)
        
    def _get_valid_models(self, postprocess_output: Path):
        # Get valid **model number** after post-processing
        output_folders = [model_folder for model_folder in os.listdir(postprocess_output) if 'success.txt' in os.listdir(postprocess_output / model_folder)]
        return output_folders
    
    def _get_diffusion_output_dir(self, state_dict, condition):
        # Create and clean the diffusion output directory
        diffusion_output_dir = Path(state_dict['user_output_dir']) / condition
        os.makedirs(diffusion_output_dir, exist_ok=True)
        if len(os.listdir(diffusion_output_dir)) > 0:
            shutil.rmtree(diffusion_output_dir)
        return diffusion_output_dir
        
    def _get_postprocess_output_dir(self, state_dict, condition):
        # Create and clean the post-process output directory
        postprocess_output_dir = Path(state_dict['user_output_dir']) / f'{condition}_post'
        os.makedirs(postprocess_output_dir, exist_ok=True)
        if len(os.listdir(postprocess_output_dir)) > 0:
            shutil.rmtree(postprocess_output_dir)
        return postprocess_output_dir

class GeneraingException(Exception):
    """Custom exception if generating failed."""
    pass

class EmptyInputException(Exception):
    """Custom exception if the input is empty."""
    pass
