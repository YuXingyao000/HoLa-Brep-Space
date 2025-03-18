import os
import shutil
from pathlib import Path

import trimesh
import ray
from trimesh.sample import sample_surface
from tqdm import tqdm
import argparse

import glob

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance

from eval.check_valid import check_step_valid_soild


@ray.remote
def sample(prefix, root, output_root, checkvalid):
    # check valid
    if checkvalid:
        step_file = glob.glob(os.path.join(root / prefix, '*.step'))
        if len(step_file) == 0:
            return
        is_valid = check_step_valid_soild(step_file[0])
        if not is_valid:
            return

    # prefer stl
    option_file = glob.glob(os.path.join(root / prefix, '*.stl')) # + glob.glob(os.path.join(root / prefix, '*.ply'))
    if len(option_file) == 0:
        return
    file = option_file[0]
    mesh = trimesh.load(str(file), force="mesh")
    if mesh.faces.shape[0] == 0:
        return
    out_pc, _ = sample_surface(mesh, 2000)
    trimesh.PointCloud(out_pc).export(str(output_root / (prefix + ".ply")))
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct Brep From Data')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--valid', action='store_true')

    root = Path(parser.parse_args().data_root)
    output_root = Path(parser.parse_args().out_root)
    check_valid = parser.parse_args().valid
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    output_root.mkdir(exist_ok=False)

    print("\nStart Sampling PointCloud...")
    ray.init(
            # num_cpus=1,
            # local_mode=True,
    )

    tasks = []
    for prefix in tqdm(os.listdir(root)):
        tasks.append(sample.remote(prefix, root, output_root, check_valid))
    ray.get(tasks)
    print("Finish Sampling PointCloud\n")
