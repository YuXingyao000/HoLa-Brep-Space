import json
import numpy as np
import trimesh

from pathlib import Path
from abc import ABC, abstractmethod

class ModelLoader(ABC):
    def __init__(self, file_path: str | Path | None = None):
        self.file_path = file_path
        self._vertices = np.array([],dtype=np.float64)
        self._edges = np.array([],dtype=np.float64)
        self._faces = np.array([],dtype=np.float64)
        self._topology = dict()
        self._is_loaded = False
        
        if file_path is not None:
            self.load(self.file_path)
    
    def IsLoaded(self):
        return self._is_loaded
    
    def load(self, file_path: str | Path):
        self._is_loaded = True
        self._file_content = self._read_file(file_path)
        self.set_vertices()
        self.set_edges()
        self.set_faces()
        self.set_topology()
    
    @abstractmethod
    def _read_file(self):
        pass
    
    @abstractmethod
    def set_vertices(self):
        pass
    
    @abstractmethod
    def set_edges(self):
        pass
    
    @abstractmethod
    def set_faces(self):
        pass
    
    @abstractmethod
    def set_topology(self):
        pass
        
    @property
    def Vertices(self):
        vertices = self._vertices
        return vertices
    
    @property
    def Edges(self):
        edges = self._edges
        return edges
    
    @property
    def Faces(self):
        faces = self._faces
        return faces
    
    @property
    def FaceEdge(self):
        fe_topology = self._topology['FE']
        return fe_topology
    
    @property
    def EdgeVertex(self):
        ev_topology = self._topology['EV']
        return ev_topology
    
class ComplexLoader(ModelLoader):
    def _read_file(self, file_path: str | Path) -> dict:
        """Return the raw file content from a JSON file

        Args:
            file_path (str | Path): file path

        Returns:
            dict: JSON dictionary
        """
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    
    def set_vertices(self):
        vertices = []
        if self._file_content['corners'] is None:
            self._vertices = np.asarray((0,0,0), dtype=np.float64)[None, None]
            return
        for vertex_data in self._file_content['corners']:
            vertices.append(vertex_data['pts'])
        self._vertices = np.array(vertices, dtype=np.float64)[:, None, :]
        return
    
    def set_edges(self):
        edges = []
        if self._file_content['curves'] is None:
            self._edges = np.array(edges, dtype=np.float64)
            return
        for edge_data in self._file_content['curves']:
            edges.append(np.array(edge_data['pts']).reshape([-1, 3]))
        self._edges = np.array(edges, dtype=np.float64)
        return
    
    def set_faces(self):
        faces = []
        if self._file_content['patches'] is None:
            self._faces = np.array(faces, dtype=np.float64)
            return
        for face_data in self._file_content['patches']:
            faces.append(np.array(face_data['grid']).reshape([self.FaceSampleFrequency, self.FaceSampleFrequency, 3]))
        self._faces = np.array(faces, dtype=np.float64)
        return

    def set_topology(self):
        face_edge_matrix = self._file_content['patch2curve']
        self._topology["FE"] = self.__convert_topo_matrix(face_edge_matrix)
        edge_vertex_matrix = self._file_content['curve2corner']
        self._topology["EV"] = self.__convert_topo_matrix(edge_vertex_matrix)
    
    @property
    def FaceSampleFrequency(self):
        return int(np.sqrt(len(self._file_content['patches'][0]['grid']) / 3))

    def __convert_topo_matrix(self, topo_matrix):
        topo_dict = dict()
        if topo_matrix is None:
            return topo_dict
        for main_index, topo_one_hot in enumerate(topo_matrix):
            connectivity = []
            for secondary_index, is_connected in enumerate(topo_one_hot):
                if is_connected == 1:
                    connectivity.append(secondary_index)
            if len(connectivity) > 0:
                topo_dict[main_index] = connectivity
        return topo_dict

class NVDNetLoader(ModelLoader):
    def _read_file(self, file_path: str | Path):
        file_content = dict()
        eval_folder = Path(file_path) / "eval"
        
        def extract_ply_data(primitive_type):
            return trimesh.load(eval_folder / f'{primitive_type}.ply').metadata['_ply_raw']['vertex']["data"]
        
        file_content['vertex_data'] = extract_ply_data('vertices')
        file_content['curve_data'] = extract_ply_data('curves')
        file_content['surface_data'] = extract_ply_data('surfaces')
        
        with open(eval_folder / 'adj_matrix.txt', 'r') as topo_file:
            topo_data = [line.strip() for line in topo_file.readlines()]
            
            p_FE = np.where(np.array(topo_data) == 'FE')[0][0]
            p_EV = np.where(np.array(topo_data) == 'EV')[0][0]
            
            FE_data = topo_data[p_FE + 1 : p_EV]    if p_FE + 1 < p_EV              else []
            EV_data = topo_data[p_EV + 1 :]         if p_EV + 1 < len(topo_data)    else []
            
            file_content['topology'] = dict()
            file_content['topology']['FE'] = FE_data
            file_content['topology']['EV'] = EV_data
        
        return file_content 
    
    def set_vertices(self):
        self._vertices = np.stack([self._file_content['vertex_data']['x'], self._file_content['vertex_data']['y'], self._file_content['vertex_data']['z']], axis=1)[:, None, :]
    
    def set_edges(self):
        curve_data = self._file_content['curve_data']
        
        sample_points = np.stack([curve_data['x'], curve_data['y'], curve_data['z']], axis=1)
        point_indices = curve_data['primitive_index']
        
        _, sample_point_group = np.unique(point_indices, return_index=True)
        
        self._edges = np.split(sample_points, sample_point_group[1:])
    
    def set_faces(self):
        surface_data = self._file_content['surface_data']
        
        sample_points = np.stack([surface_data['x'], surface_data['y'], surface_data['z']], axis=1)
        point_indices = surface_data['primitive_index']
        
        _, sample_point_group = np.unique(point_indices, return_index=True)
        self._faces = np.split(sample_points, sample_point_group[1:])
        # self._faces = [array[:int(np.sqrt(array.shape[0]))**2].reshape([int(np.sqrt(array.shape[0])), int(np.sqrt(array.shape[0])), -1]) for array in np.split(sample_points, sample_point_group[1:])]  
    
    def set_topology(self):
        self._topology['FE'] = self.__process_topology_table(self._file_content['topology']['FE'])
        self._topology['EV'] = self.__process_topology_table(self._file_content['topology']['EV'])
    
    def __process_topology_table(self, topology_table):
        result = dict()
        for connectivity in topology_table:
            connectivity = connectivity.split()
            if len(connectivity) > 1:
                primary_index = int(connectivity[0])
                secondary_index = [int(index) for index in connectivity[1:]]
                result[primary_index] = secondary_index
        return result
        
        