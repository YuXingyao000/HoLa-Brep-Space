import os
import ray
from tqdm import tqdm
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
    
    tasks = []
    
    for i, one_folder in enumerate(all_folders):
        tasks.append(
            construct_brep_from_datanpz_ray.remote(
                file_dir, 
                output_dir,
                one_folder,
                v_drop_num=drop_num,
                use_cuda=False, 
                from_scratch=True,
                is_log=False, 
                is_ray=True, 
                is_optimize_geom=False, 
                isdebug=True,
                is_save_data=True
            )
        )
        
    results = []
    success_count = 0
    for task in tqdm(tasks):
        try:
            results.append(ray.get(task, timeout=timeout))
            # Check whether the number of valid files is greater than 4
            for done_folder in os.listdir(output_dir):
                output_files = os.listdir(Path(output_dir) / Path(done_folder))
                if 'success.txt' in output_files:
                    success_count += 1
                    # ray.kill()
        except:
            results.append(None)
        if success_count >= 4:
            if ray.is_initialized():
                ray.shutdown()
            break
        else:
            success_count = 0

    print("Finished post-processing")
