import numpy as np
import argparse

from lightning_fabric import seed_everything

from eval_condition import *
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve


def get_fluxEE(vertices: np.ndarray, facets: np.ndarray) -> float:
    points = vertices[facets]
    a = points[:, 1] - points[:, 0]
    b = points[:, 2] - points[:, 0]
    normals = np.cross(a, b)
    norms = np.linalg.norm(normals)
    assert np.all(norms != 0)
    normals /= norms[:, None]
    d_S = 0.5 * norms
    fluxEE = np.sum(np.sum(normals, axis=1) * d_S)
    return abs(fluxEE)


def get_NormalC(v_recon_points: np.ndarray, v_gt_points: np.ndarray) -> float:
    # ACC
    acc_l1norm = np.sum(np.abs(v_gt_points[:, None, :3] - v_recon_points[:, :3]), axis=2)
    min_dist_index = np.argmin(acc_l1norm, axis=0)
    acc = np.mean(np.sum(v_recon_points[:, 3:] * v_gt_points[min_dist_index][:, 3:], axis=1))

    # Comp
    comp_l1norm = np.sum(np.abs(v_recon_points[:, None, :3] - v_gt_points[:, :3]), axis=2)
    min_dist_index = np.argmin(comp_l1norm, axis=0)
    comp = np.mean(np.sum(v_gt_points[:, 3:] * v_recon_points[min_dist_index][:, 3:], axis=1))

    return acc, comp, (acc + comp) / 2.0


def get_danglingEdgeLength(shape):
    no_directions = True
    edges = get_primitives(shape, TopAbs_EDGE, no_directions)
    faces = get_primitives(shape, TopAbs_FACE, no_directions)
    connection = {edge: set() for edge in edges}

    def EdgeBelongsToFace(edge, face):
        edgeOnFace = get_primitives(face, TopAbs_EDGE, True)
        for _edge in edgeOnFace:
            if _edge.IsSame(edge):
                return True
        return False

    # Get edge-face connections
    for edge in edges:
        for face in faces:
            if EdgeBelongsToFace(edge, face):
                connection[edge].add(face)

                # Get dangling edge length
    danglingEdgeLength = 0.0
    for edge, faces in connection.items():
        if len(faces) < 2:
            curve, _, _ = BRep_Tool.Curve(edge)
            if len(faces) == 1 and (BRep_Tool.Surface(list(faces)[0]).IsUPeriodic() or BRep_Tool.Surface(list(faces)[0]).IsVPeriodic()):
                continue
            else:
                danglingEdgeLength += GCPnts_AbscissaPoint.Length(GeomAdaptor_Curve(curve))

    return danglingEdgeLength


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate The Generated Brep')
    parser.add_argument('--eval_root', type=str)
    parser.add_argument('--gt_root', type=str)
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--num_cpus', type=int, default=16)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--list', type=str, default='')
    parser.add_argument('--from_scratch', action='store_true')
    args = parser.parse_args()

    seed_everything(0)
