import time

import gradio as gr
from typing import List, Callable
from Generator import GenerateMethod, UncondGenerateMethod, TxtGenerateMethod, PCGenerateMethod, SketchGenerateMethod, SVRGenerateMethod, MVRGenerateMethod
from abc import ABC, abstractmethod

# Tab Interface
class GenerateTab(ABC):
    static_state = None
    def __init__(self):
        self._button_callback = self.generator_factory().get_generate_method()
    
    @abstractmethod
    def generator_factory(self) -> GenerateMethod:
        pass
    
    @abstractmethod
    def get_input_components(self) -> List[gr.Component]:
        pass
    
    @property
    def ButtonCallBack(self) -> Callable:
        if self._button_callback is None:
            raise AttributeError("ButtonCallBack must be assigned before use.")
        return self._button_callback

def generate_tab(radio_type: str, user_state: gr.State) -> GenerateTab:
    if GenerateTab.static_state is None:
        GenerateTab.static_state = user_state
    if radio_type == "Unconditional":
        return UncondTab()
    elif radio_type == 'Point Cloud':
        return PCTab()
    elif radio_type == 'Sketch':
        return SketchTab()
    elif radio_type == 'Text':
        return TextTab()
    elif radio_type == 'SVR':
        return SVRTab()
    elif radio_type == 'MVR':
        return MVRTab()

def delegate_generate_method(radio_type: str):
    if radio_type == "Unconditional":
        return UncondTab().ButtonCallBack
    elif radio_type == 'Point Cloud':
        return PCTab().ButtonCallBack
    elif radio_type == 'Sketch':
        return SketchTab().ButtonCallBack
    elif radio_type == 'Text':
        return TextTab().ButtonCallBack
    elif radio_type == 'SVR':
        return SVRTab().ButtonCallBack
    elif radio_type == 'MVR':
        return MVRTab().ButtonCallBack
    
 
# Concrete Implementation
class UncondTab(GenerateTab):
    def generator_factory(self) -> GenerateMethod:
        return UncondGenerateMethod(self.static_state)
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


class TextTab(GenerateTab):
    def generator_factory(self) -> GenerateMethod:
        return TxtGenerateMethod(self.static_state)
    
    def get_input_components(self) -> List[gr.Component]:
        return [
            gr.Textbox(max_length=1024, label="Text"),
        ]


class PCTab(GenerateTab):
    def generator_factory(self) -> GenerateMethod:
        return PCGenerateMethod(self.static_state)
    
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


class SketchTab(GenerateTab):
    def generator_factory(self) -> GenerateMethod:
        return SketchGenerateMethod(self.static_state)
    def get_input_components(self) -> List[gr.Component]:
        return [
            gr.File(
                label='Sketch',
                file_count='single', 
                file_types=['image'], 
                interactive=True, 
                allow_reordering=True
            )
        ]


class SVRTab(GenerateTab):
    def generator_factory(self) -> GenerateMethod:
        return SVRGenerateMethod(self.static_state)
    def get_input_components(self) -> List[gr.Component]:
        return [
            gr.File(
                label='Image',
                file_count='single', 
                file_types=['image'], 
                interactive=True, 
                allow_reordering=True
            ),
        ]


class MVRTab(GenerateTab):
    def generator_factory(self) -> GenerateMethod:
        return MVRGenerateMethod(self.static_state)
    def get_input_components(self) -> List[gr.Component]:
        return [
            gr.File(
                label='View1',
                file_count='single', 
                file_types=['image'], 
                interactive=True, 
                allow_reordering=True
            ),
            gr.File(
                label='View2',
                file_count='single', 
                file_types=['image'], 
                interactive=True, 
                allow_reordering=True
            ),
            gr.File(
                label='View3',
                file_count='single', 
                file_types=['image'], 
                interactive=True, 
                allow_reordering=True
            ),
        ]
