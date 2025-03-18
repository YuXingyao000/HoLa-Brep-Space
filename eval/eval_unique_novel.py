import multiprocessing

import networkx as nx
import numpy as np
import argparse
import os

import trimesh
from tqdm import tqdm
import ray

from check_valid import check_step_valid_soild, load_data_with_prefix
from eval_brepgen import normalize_pc


def real2bit(data, n_bits=8, min_range=-1, max_range=1):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    range_quantize = 2 ** n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize)  # clip values
    return data_quantize.astype(int)


def build_graph(faces, faces_adj, n_bit=4):
    # faces1 and faces2 are np.array of shape (n_faces, n_points, n_points, 3)
    # faces_adj1 and faces_adj2 are lists of (face_idx, face_idx) adjacency, ex. [[0, 1], [1, 2]]
    if n_bit < 0:
        faces_bits = faces
    else:
        faces_bits = real2bit(faces, n_bits=n_bit)
    """Build a graph from a shape."""
    G = nx.Graph()
    for face_idx, face_bit in enumerate(faces_bits):
        G.add_node(face_idx, shape_geometry=face_bit)
    for pair in faces_adj:
        G.add_edge(pair[0], pair[1])
    return G


def is_graph_identical(graph1, graph2, atol=None):
    """Check if two shapes are identical."""
    # Check if the two graphs are isomorphic considering node attributes
    if atol is None:
        return nx.is_isomorphic(
                graph1, graph2,
                node_match=lambda n1, n2: np.array_equal(n1['shape_geometry'], n2['shape_geometry'])
        )
    else:
        return nx.is_isomorphic(
                graph1, graph2,
                node_match=lambda n1, n2: np.allclose(n1['shape_geometry'], n2['shape_geometry'], atol=atol, rtol=0)
        )


def is_graph_identical_batch(graph_pair_list, atol=None):
    is_identical_list = []
    for graph1, graph2 in graph_pair_list:
        is_identical = is_graph_identical(graph1, graph2, atol=atol)
        is_identical_list.append(is_identical)
    return is_identical_list


is_graph_identical_remote = ray.remote(is_graph_identical_batch)


def find_connected_components(matrix):
    N = len(matrix)
    visited = [False] * N
    components = []

    def dfs(idx, component):
        stack = [idx]
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                component.append(node)
                for neighbor in range(N):
                    if matrix[node][neighbor] and not visited[neighbor]:
                        stack.append(neighbor)

    for i in range(N):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)

    return components


def compute_gen_unique(graph_list, is_use_ray=False, batch_size=100000, atol=None):
    N = len(graph_list)
    unique_graph_idx = list(range(N))
    pair_0, pair_1 = np.triu_indices(N, k=1)
    check_pairs = list(zip(pair_0, pair_1))
    deduplicate_matrix = np.zeros((N, N), dtype=bool)

    if not is_use_ray:
        for idx1, idx2 in tqdm(check_pairs):
            is_identical = is_graph_identical(graph_list[idx1], graph_list[idx2], atol=atol)
            if is_identical:
                unique_graph_idx.remove(idx2) if idx2 in unique_graph_idx else None
                deduplicate_matrix[idx1, idx2] = True
                deduplicate_matrix[idx2, idx1] = True
    else:
        ray.init()
        N_batch = len(check_pairs) // batch_size
        futures = []
        for i in tqdm(range(N_batch)):
            batch_pairs = check_pairs[i * batch_size: (i + 1) * batch_size]
            batch_graph_pair = [(graph_list[idx1], graph_list[idx2]) for idx1, idx2 in batch_pairs]
            futures.append(is_graph_identical_remote.remote(batch_graph_pair, atol))
        results = ray.get(futures)

        for batch_idx in tqdm(range(N_batch)):
            for idx, is_identical in enumerate(results[batch_idx]):
                if not is_identical:
                    continue
                idx1, idx2 = check_pairs[batch_idx * batch_size + idx]
                deduplicate_matrix[idx1, idx2] = True
                deduplicate_matrix[idx2, idx1] = True
                if idx2 in unique_graph_idx:
                    unique_graph_idx.remove(idx2)
        ray.shutdown()

    unique = len(unique_graph_idx)
    print(f"Unique: {unique}/{N}")
    unique_ratio = unique / N

    return unique_ratio, deduplicate_matrix


def compute_gen_novel_bk(gen_graph_list, train_graph_list, is_use_ray=False, batch_size=100000):
    M, N = len(gen_graph_list), len(train_graph_list)
    deduplicate_matrix = np.zeros((M, N), dtype=bool)
    pair_0, pair_1 = np.triu_indices_from(deduplicate_matrix, k=1)
    check_pairs = list(zip(pair_0, pair_1))
    non_novel_graph_idx = np.zeros(M, dtype=bool)

    if not is_use_ray:
        for idx1, idx2 in tqdm(check_pairs):
            if non_novel_graph_idx[idx1]:
                continue
            is_identical = is_graph_identical(gen_graph_list[idx1], train_graph_list[idx2])
            if is_identical:
                non_novel_graph_idx[idx1] = True
                deduplicate_matrix[idx1, idx2] = True
    else:
        ray.init()
        N_batch = len(check_pairs) // batch_size
        futures = []
        for i in tqdm(range(N_batch)):
            batch_pairs = check_pairs[i * batch_size: (i + 1) * batch_size]
            batch_graph_pair = [(gen_graph_list[idx1], train_graph_list[idx2]) for idx1, idx2 in batch_pairs]
            futures.append(is_graph_identical_remote.remote(batch_graph_pair))
        results = ray.get(futures)

        for batch_idx in tqdm(range(N_batch)):
            for idx, is_identical in enumerate(results[batch_idx]):
                if not is_identical:
                    continue
                idx1, idx2 = check_pairs[batch_idx * batch_size + idx]
                deduplicate_matrix[idx1, idx2] = True
                non_novel_graph_idx[idx1] = True
        ray.shutdown()

    novel = M - np.sum(non_novel_graph_idx)
    print(f"Novel: {novel}/{M}")
    novel_ratio = novel / M
    return novel_ratio, deduplicate_matrix


def is_graph_identical_list(graph1, graph2_path_list):
    """Check if two shapes are identical."""
    # Check if the two graphs are isomorphic considering node attributes
    graph2_list, graph2_prefix_list = load_and_build_graph(graph2_path_list)
    for graph2 in graph2_list:
        if nx.is_isomorphic(graph1, graph2,
                            node_match=lambda n1, n2: np.array_equal(n1['shape_geometry'], n2['shape_geometry'])):
            return True
    return False


is_graph_identical_list_remote = ray.remote(is_graph_identical_list)


def test_check():
    sample = np.random.rand(3, 32, 32, 3)
    face1 = sample[[0, 1, 2]]
    face2 = sample[[0, 2, 1]]
    faces_adj1 = [[0, 1]]
    faces_adj2 = [[0, 2]]

    graph1 = build_graph(face1, faces_adj1)
    graph2 = build_graph(face2, faces_adj2)

    is_identical = is_graph_identical(graph1, graph2)
    # 判断图是否相等
    print("Graphs are equal" if is_identical else "Graphs are not equal")


def load_data_from_npz(data_npz_file):
    data_npz = np.load(data_npz_file, allow_pickle=True)
    data_npz1 = np.load(data_npz_file.replace("deepcad_32", "deepcad_train_v6"), allow_pickle=True)
    # Brepgen
    if 'face_edge_adj' in data_npz:
        faces = data_npz['pred_face']
        face_edge_adj = data_npz['face_edge_adj']
        faces_adj_pair = []
        N = face_edge_adj.shape[0]
        for face_idx1 in range(N):
            for face_idx2 in range(face_idx1 + 1, N):
                face_edges1 = face_edge_adj[face_idx1]
                face_edges2 = face_edge_adj[face_idx2]
                if sorted((face_idx1, face_idx2)) in faces_adj_pair:
                    continue
                if len(set(face_edges1).intersection(set(face_edges2))) > 0:
                    faces_adj_pair.append(sorted((face_idx1, face_idx2)))
        return faces, faces_adj_pair
    # Ours
    if 'sample_points_faces' in data_npz:
        face_points = data_npz['sample_points_faces']  # Face sample points (num_faces*20*20*3)
        edge_face_connectivity = data_npz['edge_face_connectivity']  # (num_intersection, (id_edge, id_face1, id_face2))
    elif 'pred_face' in data_npz and 'pred_edge_face_connectivity' in data_npz:
        face_points = data_npz['pred_face']
        edge_face_connectivity = data_npz['pred_edge_face_connectivity']
    else:
        raise ValueError("Invalid data format")
    faces_adj_pair = []
    for edge_idx, face_idx1, face_idx2 in edge_face_connectivity:
        faces_adj_pair.append([face_idx1, face_idx2])
    if face_points.shape[-1] != 3:
        face_points = face_points[..., :3]

    src_shape = face_points.shape
    face_points = normalize_pc(face_points.reshape(-1, 3)).reshape(src_shape)
    return face_points, faces_adj_pair


def load_and_build_graph(data_npz_file_list, gen_post_data_root=None, n_bit=4):
    gen_graph_list = []
    prefix_list = []
    for data_npz_file in data_npz_file_list:
        folder_name = os.path.basename(os.path.dirname(data_npz_file))
        if gen_post_data_root:
            step_file_list = load_data_with_prefix(os.path.join(gen_post_data_root, folder_name), ".step")
            if len(step_file_list) == 0:
                continue
            if not check_step_valid_soild(step_file_list[0]):
                continue
        prefix_list.append(folder_name)
        faces, faces_adj_pair = load_data_from_npz(data_npz_file)
        graph = build_graph(faces, faces_adj_pair, n_bit)
        gen_graph_list.append(graph)
    return gen_graph_list, prefix_list


load_and_build_graph_remote = ray.remote(load_and_build_graph)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake_root", type=str, required=True)
    parser.add_argument("--fake_post", type=str, required=True)
    parser.add_argument("--train_root", type=str, required=False)
    parser.add_argument("--n_bit", type=int, required=False)
    parser.add_argument("--atol", type=float, required=False)
    parser.add_argument("--use_ray", action='store_true')
    parser.add_argument("--load_batch_size", type=int, default=400)
    parser.add_argument("--compute_batch_size", type=int, default=200000)
    parser.add_argument("--txt", type=str, default=None)
    parser.add_argument("--num_cpus", type=int, default=32)
    parser.add_argument("--min_face", type=int, required=False)
    parser.add_argument("--only_unique", action='store_true')
    args = parser.parse_args()
    gen_data_root = args.fake_root
    gen_post_data_root = args.fake_post
    train_data_root = args.train_root
    is_use_ray = args.use_ray
    n_bit = args.n_bit
    atol = args.atol
    load_batch_size = args.load_batch_size
    compute_batch_size = args.compute_batch_size
    folder_list_txt = args.txt
    num_cpus = args.num_cpus

    if not n_bit and not atol:
        raise ValueError("Must set either n_bit or atol")
    if n_bit and atol:
        raise ValueError("Cannot set both n_bit and atol")

    if not args.only_unique and not train_data_root:
        raise ValueError("Must set train_data_root when not only_unique")

    if n_bit:
        atol = None
    if atol:
        n_bit = -1

    ################################################## Unqiue #######################################################
    # Load all the generated data files
    print("Loading generated data files...")
    gen_data_npz_file_list = load_data_with_prefix(gen_data_root, 'data.npz')
    if is_use_ray:
        ray.init()
        futures = []
        gen_graph_list = []
        gen_prefix_list = []
        for i in tqdm(range(0, len(gen_data_npz_file_list), load_batch_size)):
            batch_gen_data_npz_file_list = gen_data_npz_file_list[i: i + load_batch_size]
            futures.append(load_and_build_graph_remote.remote(batch_gen_data_npz_file_list, gen_post_data_root, n_bit))
        for future in tqdm(futures):
            result = ray.get(future)
            gen_graph_list_batch, gen_prefix_list_batch = result
            gen_graph_list.extend(gen_graph_list_batch)
            gen_prefix_list.extend(gen_prefix_list_batch)
        ray.shutdown()
    else:
        gen_graph_list, gen_prefix_list = load_and_build_graph(gen_data_npz_file_list, gen_post_data_root, n_bit)
    print(f"Loaded {len(gen_graph_list)} generated data files")

    if args.min_face:
        graph_node_num = [len(graph.nodes) for graph in gen_graph_list]
        gen_graph_list = [gen_graph_list[idx] for idx, num in enumerate(graph_node_num) if num >= args.min_face]
        gen_prefix_list = [gen_prefix_list[idx] for idx, num in enumerate(graph_node_num) if num >= args.min_face]
        print(f"Filtered sample that face_num < {args.min_face}, remain {len(gen_graph_list)}")

    print("Computing Unique ratio...")
    unique_ratio, deduplicate_matrix = compute_gen_unique(gen_graph_list, is_use_ray, compute_batch_size, atol=atol)
    print(f"Unique ratio: {unique_ratio}")

    if n_bit == -1:
        unique_txt = gen_data_root + f"_unique_atol_{atol}_results.txt"
    else:
        unique_txt = gen_data_root + f"_unique_{n_bit}bit_results.txt"
    fp = open(unique_txt, "w")
    print(f"Unique ratio: {unique_ratio}", file=fp)
    deduplicate_components = find_connected_components(deduplicate_matrix)
    for component in deduplicate_components:
        if len(component) > 1:
            component = [gen_prefix_list[idx] for idx in component]
            print(f"Component: {component}", file=fp)
    print(f"Deduplicate components are saved to {unique_txt}")
    fp.close()

    if args.only_unique:
        exit(0)

    # For accelerate, please first run the find_nerest.py to find the nearest item in train data for each fake sample
    ################################################### Novel ########################################################
    print("Computing Novel ratio...")
    print("Loading training data files...")
    # data_npz_file_list = load_data_with_prefix(train_data_root, 'data.npz', folder_list_txt=folder_list_txt)
    # load_batch_size = load_batch_size * 5

    is_identical = np.zeros(len(gen_graph_list), dtype=bool)
    if is_use_ray:
        ray.init()
        futures = []
        for gen_graph_idx, gen_graph in enumerate(tqdm(gen_graph_list)):
            nearest_txt = os.path.join(gen_post_data_root, gen_prefix_list[gen_graph_idx], "nearest.txt")
            if not os.path.exists(nearest_txt):
                continue
            with open(nearest_txt, "r+") as f:
                lines = f.readlines()
                train_folders = [os.path.join(train_data_root, line.strip().split(" ")[0], 'data.npz') for line in lines[2:]]
            futures.append(is_graph_identical_list_remote.remote(gen_graph, train_folders))
        results = ray.get(futures)
        for gen_graph_idx, result in enumerate(results):
            is_identical[gen_graph_idx] = result
        ray.shutdown()
    else:
        pbar = tqdm(gen_graph_list)
        for gen_graph_idx, gen_graph in enumerate(pbar):
            nearest_txt = os.path.join(gen_post_data_root, gen_prefix_list[gen_graph_idx], "nearest.txt")
            if not os.path.exists(nearest_txt):
                continue
            with open(nearest_txt, "r+") as f:
                lines = f.readlines()
                train_folders = [os.path.join(train_data_root, line.strip().split(" ")[0], 'data.npz') for line in lines[2:]]
            is_identical[gen_graph_idx] = is_graph_identical_list(gen_graph, train_folders)
            pbar.set_postfix({"novel_count": np.sum(~is_identical)})

    identical_folder = np.array(gen_prefix_list)[is_identical]
    print(f"Novel ratio: {np.sum(~is_identical) / len(gen_graph_list)}")
    novel_txt = gen_data_root + f"_novel_{n_bit}bit_results.txt"
    with open(novel_txt, "w") as f:
        f.write(f"Novel ratio: {np.sum(~is_identical) / len(gen_graph_list)}\n")
        for folder in identical_folder:
            f.write(folder + "\n")
    print("Done")


if __name__ == "__main__":
    main()
