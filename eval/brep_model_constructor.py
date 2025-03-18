import numpy as np

from typing import List

from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Edge, TopoDS_Face
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt

from OCC.Core.gp import gp_Pnt

from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2, GeomAbs_C3

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface, ShapeAnalysis_Curve, shapeanalysis

from eval.model_loader import ModelLoader

class BRepModelConstructor():
    _BSPLINE_CURVE_DEGREE_MIN = 0
    _BSPLINE_CURVE_DEGREE_MAX = 8
    _BSPLINE_FACE_DEGREE_MIN = 3
    _BSPLINE_FACE_DEGREE_MAX = 8
    _CONTINUITY = GeomAbs_C2
    _CURVELENGTH = 1
    _CURVATURE = 1
    _TORSION = 1
    _EDGE_FITTING_TOLERANCE = [1e-3, 5e-3, 8e-3, 5e-2]
    _FACE_FITTING_TOLERANCE = [1e-3, 1e-2, 3e-2, 5e-2, 8e-2]
    def __init__(self, model_sample_point: ModelLoader | None = None, tolerance: float | None = None):
        if  model_sample_point is not None and model_sample_point.IsLoaded():
            self.set_vertex_primitives(model_sample_point.Vertices)
            self.set_edge_primitives(model_sample_point.Edges)
            self.set_face_primitives(model_sample_point.Faces)
        else:
            self._vertex_positions = None
            self._edge_sample_points = None
            self._face_sample_points = None
            
        self._transfer_tolerance = 1e-6 if tolerance is None else tolerance
        self._brep_vertices = None
        self._brep_edges = None
        self._brep_faces = None
        self._topology = None
        self._is_done = False
    
    def set_vertex_primitives(self, vertex_positions: np.ndarray):
        """
        Set the positions of vertices

        Args:
            vertex_positions (np.ndarray): (num_vertices, 1, 3)
        """
        self._vertex_positions = vertex_positions
    
    def set_edge_primitives(self, edge_sample_points: np.ndarray | list):
        """
        Prepare data to construct a BSpline.
        Set sample points on each edge.

        Args:
            edge_sample_points (np.ndarray | list): (num_edges, sample_frequency_of_each_edge, 3)
        """
        self._edge_sample_points = edge_sample_points
    
    def set_face_primitives(self, face_sample_points: np.ndarray | list):
        """
        Prepare data to construct a BSpline surface.
        Set sample points on each face

        Args:
            face_sample_points (np.ndarray | list): (num_edges, u_sample_frequency, v_sample_frequency, 3)
        """
        self._face_sample_points = face_sample_points
    
    def set_topology(self, topology: dict):
        self._topology = topology
    
    def IsDone(self):
        return self._is_done
    
    def build(self):
        self._is_done = True
        if self._vertex_positions is not None:
            self.build_brep_vertices()
        
        if self._edge_sample_points is not None:
            self.build_brep_edges()
        
        if self._face_sample_points is not None:
            self.build_brep_faces()
        
    def build_brep_vertices(self):
        vertices = []
        for vertex_position in self._vertex_positions:
            vertex_position = vertex_position[0]
            vertices.append(BRepBuilderAPI_MakeVertex(gp_Pnt(*vertex_position)))
        self._brep_vertices = vertices
    
    def build_brep_edges(self):
        edges = []
        for i, edge_sample_points in enumerate(self._edge_sample_points):
            # 34 points
            gp_array = TColgp_Array1OfPnt(1, len(edge_sample_points))
            for index_i, sample_point in enumerate(edge_sample_points):
                gp_array.SetValue(index_i + 1, gp_Pnt(*sample_point))

            for precision in self._EDGE_FITTING_TOLERANCE:
                try:
                    bspline_edge = GeomAPI_PointsToBSpline(
                            gp_array,
                            self._CURVELENGTH,
                            self._CURVATURE,
                            self._TORSION,
                            self._BSPLINE_CURVE_DEGREE_MAX,
                            self._CONTINUITY,
                            precision
                    ).Curve()
                    edges.append(BRepBuilderAPI_MakeEdge(bspline_edge).Edge())
                    break
                except Exception as e:
                    print(e)
                    self._is_done = False
                    continue
        self._brep_edges = edges
    
    def build_brep_faces(self):
        faces = []
        for i, one_face_sample_points in enumerate(self._face_sample_points):
            u_dim = one_face_sample_points.shape[0]
            v_dim = one_face_sample_points.shape[1]
            gp_array = TColgp_Array2OfPnt(1, u_dim, 1, v_dim)
            for index_i in range(1, u_dim + 1):
                for index_j in range(1, v_dim + 1):
                    gp_array.SetValue(index_i, index_j, gp_Pnt(*one_face_sample_points[index_i - 1, index_j - 1]))

            for precision in self._FACE_FITTING_TOLERANCE:
                try:
                    bspline_face = GeomAPI_PointsToBSplineSurface(
                            gp_array,
                            self._CURVELENGTH,
                            self._CURVATURE,
                            self._TORSION,
                            self._BSPLINE_CURVE_DEGREE_MAX,
                            self._CONTINUITY,
                            precision
                    ).Surface()
                    faces.append(BRepBuilderAPI_MakeFace(bspline_face, self.TransferTolerance).Face())
                    break
                except Exception as e:
                    print(e)
                    self._is_done = False
                    continue

        self._brep_faces = faces
    
    @property
    def BRepVertices(self) -> List[TopoDS_Vertex]:
        return self._brep_vertices
    
    @property
    def BRepEdges(self) -> List[TopoDS_Edge]:
        return self._brep_edges
    
    @property
    def BRepFaces(self) -> List[TopoDS_Face]:
        return self._brep_faces
    
    @property
    def TransferTolerance(self):
        """
        The tolerance to build a TopoDS_Face
        """
        return self._transfer_tolerance

    @TransferTolerance.setter
    def TransferTolerance(self, value):
        self._transfer_tolerance = value
        self.build_brep_faces()
    