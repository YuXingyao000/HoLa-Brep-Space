import time

import gradio as gr
from typing import List, Callable
from abc import ABC, abstractmethod

# Tab Interface
class AppLayout(ABC):
    static_state = None
    
    @abstractmethod
    def get_note(self) -> gr.Markdown:
        pass
    
    @abstractmethod
    def get_input_components(self) -> List[gr.Component]:
        pass


def build_layout(radio_type: str, user_state: gr.State) -> AppLayout:
    if AppLayout.static_state is None:
        AppLayout.static_state = user_state
    if radio_type == "Unconditional":
        return UncondLayout()
    elif radio_type == 'Point Cloud':
        return PCLayout()
    elif radio_type == 'Sketch':
        return SketchLayout()
    elif radio_type == 'Text':
        return TextLayout()
    elif radio_type == 'SVR':
        return SVRLayout()
    elif radio_type == 'MVR':
        return MVRLayout()


    
 
# Concrete Implementation
class UncondLayout(AppLayout):

    def get_note(self):
        return gr.Markdown(
            """
            **Note:**
            
            - ✨ Randomly generate 4 models
            """
            )

    def get_input_components(self) -> List[gr.Component]:
        return [
            gr.Number(
                label="Seed",
                value=int(time.time()), 
                minimum=0, 
                maximum=2**31-1, 
                step=1
            ),
        ]


class TextLayout(AppLayout):

    def get_note(self):
        return gr.Markdown(
                        """
                        **Note:**
                        
                        - ❔ Text prompts describe the shape and features of an object using natural language, providing an intuitive way to generate 3D models. 
                        - 👉 HoLa-BRep supports generating B-rep models based on text prompts that align with your descriptions. 
                        - ✨ Input your text description to generate 4 possible B-rep results and experience seamless conversion from language to CAD models.
                        
                        <br>
                        """
                    )
    
    def get_input_components(self) -> List[gr.Component]:
        return [
            gr.Textbox(lines = 8,max_length=1024, label="Text"),
        ]


class PCLayout(AppLayout):

    def get_note(self):
        return gr.Markdown(
                        """
                        **Note:**
                        - ❔ Point clouds are sets of 3D coordinates obtained from 3D scanning or other processes, typically used to represent the shape of an object. 
                        - 👉 HoLa-BRep can transform point cloud data into complete B-rep models. By encoding the geometric and topological information of the point cloud, the generated B-rep results accurately capture the original shape's details and provide high-quality CAD models. 
                        - ✨ Upload your point cloud file to explore 4 plausible B-rep results generated from it.
                        
                        <br>
                        """
                    )
    
    def get_input_components(self):
        return [
            gr.File(
                label='PC',
                file_count='single', 
                file_types=['.ply'], 
                interactive=True, 
                allow_reordering=True
            ),
        ]


class SketchLayout(AppLayout):

    def get_note(self):
        return gr.Markdown(
                        """
                        **Note:**
                        - ❔ A 2D sketch is a simplified representation of an object's shape, usually presented in the form of lines and geometric outlines. 
                        - 👉 HoLa-BRep can learn geometric and topological information from 2D sketches and convert them into complete 3D B-rep models. 
                        - ✨ Upload your 2D sketch to generate 4 plausible B-rep results and quickly transform your design sketches into high-quality CAD models.
                        
                        <br>
                        """
                    )

    def get_input_components(self) -> List[gr.Component]:
        return [
            gr.Image(
                label='Sketch',
                type='filepath', 
                sources=["upload"],
                interactive=True, 
            )
        ]


class SVRLayout(AppLayout):

    def get_note(self):
        return gr.Markdown(
                        """
                        **Note:**
                        
                        - ❔ A single-view image provides a flat view of an object, containing limited geometric and topological information. 
                        - 👉 HoLa-BRep extracts shape features from single-view images and converts them into complete B-rep models. 
                        - ✨ Using the powerful generative capabilities of the model, you can obtain 4 plausible B-rep results inferred from the single-view image, showcasing various potential shape details. Upload an image to start generating.
                        
                        <br>
                        <br>
                        <br>
                        """
                    )

    def get_input_components(self) -> List[gr.Component]:
        return [
            gr.Image(
                label='Image',
                type='filepath', 
                sources=["upload"],
                interactive=True, 
            ),
        ]


class MVRLayout(AppLayout):

    def get_note(self):
        return gr.Markdown(
                        """
                        **Note:**
                        
                        - ❔ Multi-view images provide multiple photos of an object from different angles, offering a more comprehensive description of the object's overall shape and structure. 
                        - 👉 HoLa-BRep supports extracting rich geometric features from multi-view images and converts them into high-quality B-rep models. 
                        - ✨ Upload multiple images to explore 4 possible B-rep results and experience efficient conversion from multi-view images to CAD models.
                        
                        <br>
                        """
                    )
    
    def get_input_components(self) -> List[gr.Component]:
        return [
            gr.Image(
                label='View1',
                type='filepath', 
                interactive=True, 
                sources=["upload"]
            ),
            gr.Image(
                label='View2',
                type='filepath', 
                interactive=True, 
                sources=["upload"]

            ),
            gr.Image(
                label='View3',
                type='filepath', 
                interactive=True, 
                sources=["upload"]

            ),
        ]
