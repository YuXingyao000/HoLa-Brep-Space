from __future__ import annotations

import trimesh
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Edge, TopoDS_Face

from typing import List

from diffusion.utils import get_triangulations, get_points_along_edge, get_curve_length
from eval.brep_model_constructor import BRepModelConstructor
from eval.model_loader import ModelLoader, ComplexLoader, NVDNetLoader

class DataProcessor(ABC):

    def __init__(self, file_path: str | Path, build_brep: bool = False):
        self.file_path = file_path
        
        self.model_loader:ModelLoader = self.create_loader()
        self.model_loader.load(self.file_path)
        
        if build_brep:
            self._brep_constructor = BRepModelConstructor(self.model_loader)
            self._brep_constructor.build()
        else:
            self._brep_constructor = BRepModelConstructor()
        # assert len(self.Vertices) == self.VertexPositions.shape[0]
        # assert len(self.Edges) == self.EdgeSamplePoints.shape[0]
        # assert len(self.Faces) == self.FaceSamplePoints.shape[0]

    def get_data(self, v_num_per_m: int):
        vertices, vertex_points = self.Vertices, self.VertexPositions
        edges, edge_points = self.__sample_points_on_edges(v_num_per_m)
        faces, face_points = self.__sample_points_on_faces(v_num_per_m)
        return faces, face_points, edges, edge_points, vertices, vertex_points

    def __sample_points_on_faces(self, v_num_per_m: int):
        if len(self.Faces) == 0:
            return [], self.FaceSamplePoints
        
        faces, face_points = [], []

        for face in self.Faces:
            try:
                v, f = get_triangulations(face, 0.1, 0.1)
                if len(f) == 0:
                    print("Ignore 0 face")
                    continue
            except:
                print("Ignore 1 face")
                continue
            mesh_item = trimesh.Trimesh(vertices=v, faces=f)
            area = mesh_item.area
            num_samples = min(max(int(v_num_per_m * v_num_per_m * area), 5), 10000)
            pc_item, id_face = trimesh.sample.sample_surface(mesh_item, num_samples)
            normals = mesh_item.face_normals[id_face]
            faces.append(face)
            face_points.append(np.concatenate((pc_item, normals), axis=1))

        return faces, face_points

    def __sample_points_on_edges(self, v_num_per_m: int):
        if len(self.Edges) == 0:
            return [], self.EdgeSamplePoints
        
        edges, edge_points = [], []

        for edge in self.Edges:
            length = get_curve_length(edge)
            num_samples = min(max(int(v_num_per_m * length), 5), 10000)
            v = get_points_along_edge(edge, num_samples)
            edges.append(edge)
            edge_points.append(v)

        return edges, edge_points
    
    @abstractmethod
    def create_loader(self) -> ModelLoader:
        pass
    
    @property
    def Vertices(self) -> List[TopoDS_Vertex]:
        """
        The list of TopoDS_Vertex
        """
        vertices = []
        if self._brep_constructor.IsDone():
            vertices = self._brep_constructor.BRepVertices
        return vertices

    @property
    def Edges(self) -> List[TopoDS_Edge]:
        """
        The list of TopoDS_Edge
        """
        edges = []
        if self._brep_constructor.IsDone():
            edges = self._brep_constructor.BRepEdges
        return edges

    @property
    def Faces(self) -> List[TopoDS_Face]:
        """
        The list of TopoDS_Face
        """
        faces = []
        if self._brep_constructor.IsDone():
            faces = self._brep_constructor.BRepFaces
        return faces

    @property
    def VertexPositions(self) -> np.ndarray:
        """
        Vertex positions

        Returns:
            np.ndarray: (num_vertices, 3)
        """
        
        vertex_sample_points = self.model_loader.Vertices
        return vertex_sample_points

    @property
    def EdgeSamplePoints(self) -> np.ndarray:
        """
        Sample points on every edge

        Returns:
            np.ndarray: (num_edges, edge_sample_frequency, 3)
        """
        edge_sample_points = self.model_loader.Edges
        return edge_sample_points

    @property
    def FaceSamplePoints(self) -> np.ndarray:
        """
        Sample points on every face

        Returns:
            np.ndarray: (num_faces, u_sample_frequency, v_sample_frequency, 3)
        """
        face_sample_points = self.model_loader.Faces
        return face_sample_points

    @property
    def FaceEdge(self) -> dict:
        face_edge = self.model_loader.FaceEdge
        return face_edge

    @property
    def EdgeVertex(self) -> dict:
        edge_vertex = self.model_loader.EdgeVertex
        return edge_vertex


class ComplexGenProcessor(DataProcessor):

    def __init__(self, file_path, build_brep = False):
        super().__init__(file_path, build_brep)

    def create_loader(self):
        return ComplexLoader()


class NVDNetProcessor(DataProcessor):
    def __init__(self, file_path, build_brep = False):
        super().__init__(file_path, build_brep)

    def create_loader(self):
        return NVDNetLoader()    

