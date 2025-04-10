import os
from pathlib import Path

from construct_brep import construct_brep_from_datanpz



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
