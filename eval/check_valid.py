import random
import shutil
import sys

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.Interface import Interface_Static
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.StepData import StepData_StepModel
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_COMPOUND, TopAbs_SHELL, TopAbs_FACE, TopAbs_EDGE
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
import os
import argparse
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from diffusion.utils import get_primitives

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

Interface_Static.SetIVal("read.precision.mode", 1)
Interface_Static.SetRVal("read.precision.val", 1e-1)
# Interface_Static.SetIVal("read.stdsameparameter.mode", 1)
# Interface_Static.SetIVal("read.surfacecurve.mode", 3)
#
# Interface_Static.SetCVal("write.step.schema", "DIS")
Interface_Static.SetIVal("write.precision.mode", 2)
Interface_Static.SetRVal("write.precision.val", 1e-1)


# Interface_Static.SetIVal("write.surfacecurve.mode", 1)


def check_step_valid_soild(step_file, precision=1e-1, return_shape=False):
    try:
        shape = read_step_file(str(step_file), as_compound=False, verbosity=False)
    except:
        if return_shape:
            return False, None
        else:
            return False
    if shape.ShapeType() != TopAbs_SOLID:
        if return_shape:
            return False, shape
        else:
            return False
    shape_tol_setter = ShapeFix_ShapeTolerance()
    shape_tol_setter.SetTolerance(shape, precision)
    analyzer = BRepCheck_Analyzer(shape)
    is_valid = analyzer.IsValid()
    if return_shape:
        return is_valid, shape
    return is_valid


def load_data_with_prefix(root_folder, prefix, folder_list_txt=None):
    data_files = []
    folder_list = []
    if folder_list_txt is not None:
        with open(folder_list_txt, "r") as f:
            folder_list = f.read().splitlines()
    # Walk through the directory tree starting from the root folder
    for root, dirs, files in os.walk(root_folder):
        if folder_list_txt is not None and os.path.basename(root) not in folder_list:
            continue
        is_found = False
        for filename in files:
            # Check if the file ends with the specified prefix
            if filename.endswith(prefix):
                file_path = os.path.join(root, filename)
                is_found = True
                data_files.append(file_path)
        if not is_found:
            print(f"No {prefix} file found in {root}")
        
    return data_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=False, default="")
    parser.add_argument("--only_success", action="store_true", default=False)
    args = parser.parse_args()
    data_root = args.data_root
    only_success = args.only_success
    folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]

    if args.prefix:
        step_file_list = load_data_with_prefix(os.path.join(data_root, args.prefix), ".step")
        assert len(step_file_list) > 0
        print(f"Checking CAD solids in {args.prefix}...")
        isvalid = check_step_valid_soild(step_file_list[0], is_set_gloabl=True)
        print("Valid" if isvalid else "Invalid")
        exit(0)

    step_file_list = load_data_with_prefix(data_root, ".step")

    print(f"Total sample features: {len(folders)}")
    print(f"Total CAD solids: {len(step_file_list)}")

    print("Start checking CAD solids...")

    exception_folders = []
    exception_out_root = data_root + "_exception"
    if os.path.exists(exception_out_root):
        shutil.rmtree(exception_out_root)
    os.makedirs(exception_out_root, exist_ok=False)

    # Load cad data
    valid_count = 0
    pbar = tqdm(step_file_list)
    num_faces = []
    num_edges = []
    for step_file in pbar:
        is_valid, shape = check_step_valid_soild(step_file, return_shape=True)
        if os.path.exists(os.path.join(os.path.dirname(step_file), "success.txt")) and not is_valid:
            folder_name = os.path.basename(os.path.dirname(step_file))
            exception_folders.append(folder_name)
            shutil.copytree(os.path.dirname(step_file), os.path.join(exception_out_root, folder_name))

        if is_valid:
            if only_success and not os.path.exists(os.path.join(os.path.dirname(step_file), "success.txt")):
                continue
            valid_count += 1
            num_faces.append(len(get_primitives(shape, TopAbs_FACE)))
            num_edges.append(len(get_primitives(shape, TopAbs_EDGE)) // 2)
            pbar.set_postfix({"valid_count": valid_count})
        # else:
            # print(f"Invalid CAD solid: {step_file}")

    fig, ax = plt.subplots(1, 2, layout="constrained")
    ax[0].set_title("Num. faces")
    ax[1].set_title("Num. edges")
    hist_f, bin_f = np.histogram(num_faces, bins=5, range=(0, 30))
    hist_e, bin_e = np.histogram(num_edges, bins=5, range=(0, 50))
    # Normalize
    hist_f = hist_f / np.sum(hist_f)
    hist_e = hist_e / np.sum(hist_e)
    ax[0].plot(bin_f[:-1], hist_f, "-")
    ax[1].plot(bin_e[:-1], hist_e, "-")
    ax[0].set_aspect(1. / ax[0].get_data_ratio())
    ax[1].set_aspect(1. / ax[1].get_data_ratio())
    plt.savefig(data_root + "_num_faces_edges.png", dpi=600)

    print(f"Number of valid CAD solids: {valid_count}")
    print(f"Valid rate: {valid_count / len(folders) * 100:.2f}%")

    if len(exception_folders) > 0:
        with open(os.path.join(exception_out_root, "exception_folders.txt"), "w") as f:
            for folder in exception_folders:
                f.write(folder + "\n")
        print(f"Exception folders are saved to {exception_out_root}")
    if len(exception_folders) == 0:
        shutil.rmtree(exception_out_root)
        print("No exception folders found.")
