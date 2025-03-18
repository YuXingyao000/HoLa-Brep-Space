import glob
import ray
import numpy as np
import argparse

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GProp import GProp_GProps
from lightning_fabric import seed_everything

from eval.eval_condition import *

import networkx as nx
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt


def remove_outliers_zscore(data, threshold=3, max_value=50):
    if len(data) == 0 or sum(data) == 0:
        return data
    mean = np.mean(data)
    std_dev = np.std(data)
    return [x for x in data if abs((x - mean) / (std_dev + 1e-8)) <= threshold and x < max_value]


def extract_edges_and_vertices(shape):
    explorer_edges = TopExp_Explorer(shape, TopAbs_EDGE)
    explorer_vertices = TopExp_Explorer(shape, TopAbs_VERTEX)

    vertex_map = {}
    edges = []

    while explorer_edges.More():
        edge = explorer_edges.Current()

        vertices_on_edge = []
        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex = vertex_explorer.Current()
            point = BRep_Tool.Pnt(vertex)
            coord = (round(point.X(), 6), round(point.Y(), 6), round(point.Z(), 6))

            if coord not in vertex_map:
                vertex_map[coord] = len(vertex_map)

            vertices_on_edge.append(vertex_map[coord])
            vertex_explorer.Next()

        if len(vertices_on_edge) == 2:
            edges.append(tuple(vertices_on_edge))

        explorer_edges.Next()

    return vertex_map, edges


def create_nx_graph(vertex_map, edges):
    graph = nx.Graph()

    for coord, node_id in vertex_map.items():
        graph.add_node(node_id, coord=coord)

    for edge in edges:
        graph.add_edge(edge[0], edge[1])

    return graph


def calculate_cyclomatic_complexity(graph):
    num_nodes = graph.number_of_nodes()  # N
    num_edges = graph.number_of_edges()  # E
    if graph.is_directed():
        num_components = nx.number_strongly_connected_components(graph)
    else:
        num_components = nx.number_connected_components(graph)
    # M = E - N + 2P
    cyclomatic_complexity = num_edges - num_nodes + 2 * num_components
    return cyclomatic_complexity


def eval_complexity_one(step_file_path):
    isvalid, shape = check_step_valid_soild(step_file_path, return_shape=True)
    if not isvalid:
        return None

    vertex_map, edges = extract_edges_and_vertices(shape)
    graph = create_nx_graph(vertex_map, edges)
    cyclomatic_complexity = calculate_cyclomatic_complexity(graph)

    face_list = get_primitives(shape, TopAbs_FACE)
    num_face = len(face_list)
    num_edge = len(vertex_map.keys())
    num_vertex = len(edges)

    sample_point_curvature = []
    num_samples = 256
    for face in face_list:
        surf_adaptor = BRepAdaptor_Surface(face)
        u_min, u_max, v_min, v_max = (surf_adaptor.FirstUParameter(), surf_adaptor.LastUParameter(), surf_adaptor.FirstVParameter(),
                                      surf_adaptor.LastVParameter())

        u_samples = np.linspace(u_min, u_max, int(np.sqrt(num_samples)))
        v_samples = np.linspace(v_min, v_max, int(np.sqrt(num_samples)))

        face_sample_point_curvature = []
        for u in u_samples:
            for v in v_samples:
                props = BRepLProp_SLProps(surf_adaptor, u, v, 2, 1e-8)
                if props.IsCurvatureDefined():
                    mean_curvature = props.MeanCurvature()
                    face_sample_point_curvature.append(abs(mean_curvature))
        face_sample_point_curvature = remove_outliers_zscore(face_sample_point_curvature)
        if len(face_sample_point_curvature) == 0:
            continue
        sample_point_curvature.append(np.median(face_sample_point_curvature))

    mean_curvature = np.mean(sample_point_curvature) if len(sample_point_curvature) > 0 else np.nan

    if num_face == 0 or mean_curvature == np.nan:
        return None

    return {
        'num_face'             : int(num_face),
        'num_edge'             : int(num_edge),
        'num_vertex'           : int(num_vertex),
        'cyclomatic_complexity': cyclomatic_complexity,
        'mean_curvature'       : mean_curvature,
    }


eval_complexity_one_remote = ray.remote(eval_complexity_one)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Brep Complexity')
    parser.add_argument('--eval_root', type=str)
    parser.add_argument('--only_valid', action='store_true')
    args = parser.parse_args()

    # 设置随机种子
    seed_everything(0)
    ray.init(ignore_reinit_error=True, local_mode=False)

    all_folders = os.listdir(args.eval_root)
    is_valid_list = []
    futures = []
    for folder in tqdm(all_folders):
        step_path_list = glob.glob(os.path.join(args.eval_root, folder, '*.step'))
        if len(step_path_list) == 0:
            is_valid_list.append(False)
            futures.append(None)
            continue
        is_valid_list.append(check_step_valid_soild(step_path_list[0], return_shape=False))
        futures.append(eval_complexity_one_remote.remote(step_path_list[0]))

    assert len(is_valid_list) == len(futures) == len(all_folders)
    all_result = {}
    for i, future in enumerate(tqdm(futures)):
        if future is None:
            continue
        result = ray.get(future)
        if args.only_valid and not is_valid_list[i]:
            continue
        all_result[all_folders[i]] = result

    num_face_list = []
    num_edge_list = []
    num_vertex_list = []
    cyclomatic_complexity_list = []
    mean_curvature_list = []
    exception_folder = []

    for folder, result in tqdm(all_result.items()):
        if result is None:
            continue
        result = dict(result)
        num_face_list.append(result['num_face'])
        num_edge_list.append(result['num_edge'])
        num_vertex_list.append(result['num_vertex'])
        cyclomatic_complexity_list.append(result['cyclomatic_complexity'])
        mean_curvature_list.append(result['mean_curvature'])
        exception_folder.append(folder)

    print(f'Num Face: {np.mean(num_face_list)}')
    print(f'Num Edge: {np.mean(num_edge_list)}')
    print(f'Num Vertex: {np.mean(num_vertex_list)}')
    print(f'Cyclomatic Complexity: {np.mean(cyclomatic_complexity_list)}')
    print(f'Mean Curvature: {np.mean(mean_curvature_list)}')
    print(f"{np.mean(num_face_list)} {np.mean(num_edge_list)} {np.mean(num_vertex_list)} "
          f"{np.mean(cyclomatic_complexity_list)} {np.mean(mean_curvature_list)}")
    ray.shutdown()
