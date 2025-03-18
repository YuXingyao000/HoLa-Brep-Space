from pathlib import Path

from OCC.Extend.DataExchange import write_stl_file
from lightning_fabric import seed_everything
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys, pickle
import numpy as np
import trimesh

from diffusion.utils import get_triangulations
from eval.check_valid import check_step_valid_soild
from diffusion.utils import can_be_triangularized



def remove_outliers_zscore(data, threshold=3):
    """
    Remove outliers using z-score method
    Args:
        data: 1D numpy array
        threshold: z-score threshold (default 3)
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)

    mask = z_scores < threshold
    return data[mask], mask


if __name__ == "__main__":
    seed_everything(0)
    if len(sys.argv) == 3:
        pkl = sys.argv[1]
        output_path = sys.argv[2]
        src_step_root = None
    elif len(sys.argv) == 4:
        pkl = sys.argv[1]
        output_path = sys.argv[2]
        src_step_root = sys.argv[3]

    src_folder_list, nearest_name, data_ori = pickle.load(open(pkl, "rb"))

    if src_step_root is not None:
        # check valid
        print("Checking Validity...")
        is_valid_mask = np.array([False] * len(src_folder_list))
        src_step_root = Path(src_step_root)
        for i, folder in tqdm(enumerate(src_folder_list)):
            assert (src_step_root / folder / "recon_brep.step").exists()
            is_valid = check_step_valid_soild(str(src_step_root / folder / "recon_brep.step"))
            is_valid_mask[i] = is_valid

        src_folder_list = [src_folder_list[i] for i in range(len(src_folder_list)) if is_valid_mask[i]]
        nearest_name = [nearest_name[i] for i in range(len(nearest_name)) if is_valid_mask[i]]
        data_ori = data_ori[is_valid_mask]

        print(f"Valid Shapes: {len(src_folder_list)}")

    mean_list = []
    median_list = []
    percentile_75_list = []
    data = data_ori.min(axis=1)
    for i in range(10):
        print("\nNo. {} Iteration".format(i))
        # random sample 3000 shapes
        data = np.random.choice(data, 1000)
        data, _ = remove_outliers_zscore(data, threshold=3)
        his, bin_edges = np.histogram(data, bins=45, range=(0, 4500))
        plt.xlim(0, 100)
        plt.barh(bin_edges[:-1], his, height=50)
        plt.title("Light Field Distance (LFD) Distribution of the Generated Shapes")
        plt.xlabel("Frequency")
        plt.ylabel("Light Field Distance (LFD)")
        plt.savefig(output_path, dpi=600)

        print("Mean: ", data.mean())
        print("Max: ", data.max())
        print("Min: ", data.min())
        print("Std: ", data.std())
        print("Median: ", np.median(data))
        print("1%: ", np.percentile(data, 1))
        print("10%: ", np.percentile(data, 10))
        print("25%: ", np.percentile(data, 25))
        print("50%: ", np.percentile(data, 50))
        print("75%: ", np.percentile(data, 75))
        print("90%: ", np.percentile(data, 90))
        print("99%: ", np.percentile(data, 99))
        median_list.append(np.median(data))
        mean_list.append(np.mean(data))
        percentile_75_list.append(np.percentile(data, 75))
        if True:
            small_ids = np.where(data < 100)
            # data_root = Path("/mnt/d/brepgen_train")
            # data_root = Path("/mnt/d/deepcad_train_v6")
            # pkl = Path(pkl)
            # src_dir = pkl.parent / Path(pkl).name.replace("_lfd.pkl", "_post")
            # dst_file = pkl.parent / Path(pkl).name.replace("_lfd.pkl", "_lfd.ply")
            # mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
            # offset = 0
            # for id in tqdm(small_ids[0]):
            #     item = trimesh.load(str(src_dir / src_folder_list[id] / "recon_brep.stl"))
            #     delta_x = offset % 20 * 9
            #     delta_y = offset // 20 * 3
            #     item.vertices += [delta_x, delta_y, 0]
            #     mesh = mesh + item
            #     item = trimesh.load(str(data_root / nearest_name[id] / "mesh.ply"))
            #     delta_x = offset % 20 * 9 + 3
            #     delta_y = offset // 20 * 3
            #     item.vertices += [delta_x, delta_y, 0]
            #     mesh = mesh + item
            #     offset += 1
            # mesh.export(str(dst_file))
            print(f"{small_ids[0].shape[0]} shapes' LFD is less than 100")
        pass
    print("\nMean List: ", mean_list)
    print("Mean of Mean List: ", np.mean(mean_list))
    print("\nMedian List: ", median_list)
    print("Mean of Median List: ", np.mean(median_list))
    print("\nPercentile 75 List: ", percentile_75_list)
    print("Mean of Percentile 75 List: ", np.mean(percentile_75_list))
    print(f"{np.mean(mean_list)} {np.mean(median_list)} {np.mean(percentile_75_list)}")
