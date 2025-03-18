import os
from pathlib import Path

import trimesh
import ray
from trimesh.sample import sample_surface

@ray.remote
def sample(prefix, root, output_root):
    file = root / prefix / "mesh.ply"
    mesh = trimesh.load(str(file), force="mesh")
    out_pc, _ = sample_surface(mesh, 2000)
    trimesh.PointCloud(out_pc).export(str(output_root / (prefix + ".ply")))
    pass

if __name__ == '__main__':
    ray.init(
        # num_cpus=1,
        # local_mode=True,
    )

    root = Path("D:/brepnet/deepcad_test_v6")
    output_root = Path("D:/brepnet/deepcad_test_v6_points")
    output_root.mkdir(exist_ok=True, parents=True)

    tasks = []
    for prefix in os.listdir(root):
        if not prefix.startswith("00"):
            continue
        tasks.append(sample.remote(prefix, root, output_root))
    ray.get(tasks)
    os.listdir()