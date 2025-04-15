import os
import ray
import time
from pathlib import Path

from construct_brep import construct_brep_from_datanpz

# This file still exists just because a UNSPEAKABLE evil class depends on it

def inference_batch_postprocess(file_dir: Path ,output_dir: Path, num_cpus: int=4, drop_num: int=2, timeout: int=60):
    print("Start post-processing")
    
    if not ray.is_initialized():
        ray.init(
            dashboard_host="0.0.0.0",
            dashboard_port=8080,
            num_cpus=num_cpus,
        )
    
    construct_brep_from_datanpz_ray = ray.remote(num_cpus=1, max_retries=0)(construct_brep_from_datanpz)
    
    all_folders = sorted(os.listdir(file_dir))
    
    tasks = [
        construct_brep_from_datanpz_ray.remote(
            data_root=file_dir,
            out_root=output_dir,
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
        for model_number in all_folders
    ]

        
    results = []
    success_count = 0
    while tasks and success_count < 4:
        done_ids, tasks = ray.wait(tasks, num_returns=1, timeout=60)
        for done_id in done_ids:
            try:
                result = ray.get(done_id)
                results.append(result)
                
                # Delay just a bit to ensure file handles are released
                time.sleep(0.2)
                # Check for 'success.txt' in output folders
                for done_folder in Path(output_dir).iterdir():
                    output_files = os.listdir(Path(output_dir) / done_folder)
                    if 'success.txt' in output_files:
                        success_count += 1
                        
            except Exception as e:
                print(f"Task failed or timed out: {e}")
                results.append(None)
            if success_count >= 4:
                # Make sure the files are written successfully
                time.sleep(5.0)
                break

    print("Finished post-processing")
