import os
from pathlib import Path
import numpy as np
from typing import Optional

from diffusion.utils import export_edges
from construct_brep import construct_brep_from_datanpz

import torch
from app.DataProcessor import DataProcessor
from app.BuildingDirector import ModelDirector

def inference(model_director: ModelDirector, input_files: list[Path | str], output_path: Path, seed: Optional[int]=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_director.construct_model()
    model_builder = model_director.buider
    
    model_builder.setup_output_dir(output_path)
    if seed is not None:
        model_builder.setup_seed(seed)
    
    model_builder.make_model(device)
    model = model_builder.model
            
    # Process user input data
    data = DataProcessor().process(model_director.get_generating_condition(), input_files)
    
    # Inference
    with torch.no_grad():
        pred_results = model.inference(DataProcessor.NUM_PROPOSALS, device, v_data=data, v_log=True)
        
    # Save intermediate files for post-processing
    for i, result in enumerate(pred_results):
        output_dir = output_path / f"00_{i:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        export_edges(result["pred_edge"], (output_dir / "edge.obj").as_posix())
        
        np.savez_compressed(
            file                        = (output_dir / "data.npz").as_posix(),
            pred_face_adj_prob          = result["pred_face_adj_prob"],
            pred_face_adj               = result["pred_face_adj"].cpu().numpy(),
            pred_face                   = result["pred_face"],
            pred_edge                   = result["pred_edge"],
            pred_edge_face_connectivity = result["pred_edge_face_connectivity"],
        )

def inference_batch_postprocess(file_dir: Path ,output_dir: Path, num_cpus: int=4, drop_num: int=2, timeout: int=60):
    print("Start post processing")
    
    all_folders = sorted(os.listdir(file_dir))
    
    tasks = []
    for i, one_folder in enumerate(all_folders):
        construct_brep_from_datanpz(
            data_root=file_dir,
            out_root=output_dir,
            folder_name=one_folder,
            v_drop_num=drop_num,
            is_ray=False,
            is_optimize_geom=False,
            from_scratch=True,
            isdebug=True,
            is_save_data=True
            )
        success_count = 0
        for done_folder in os.listdir(output_dir):
            output_files = os.listdir(Path(output_dir) / Path(done_folder))
            if 'success.txt' in output_files:
                success_count += 1
        if success_count >= 4:
            break
    print("Done.")
