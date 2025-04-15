import time

import gradio as gr
from typing import List, Callable
from abc import ABC, abstractmethod

# Tab Interface
class AppLayout(ABC):
    @abstractmethod
    def get_English_note(self) -> gr.Markdown:
        pass
    
    @abstractmethod
    def get_Chinese_note(self):
        pass
    
    @abstractmethod
    def get_input_components(self) -> List[gr.Component]:
        pass
    
 
# Concrete Implementation
class UncondLayout(AppLayout):

    def get_English_note(self):
        return gr.Markdown(
            """
            **Note:**
            
            - We generate 4 BRep models from sampled noise in Gaussian distribution.
            - The model is trained on ABC dataset with a complexity range of 10~100 surface primitives.
            - Compared with the state-of-the-art BRep generation methods, HoLa-BRep has a 20%-40% improvement in the validity ratio of the generated models on both the DeepCAD dataset and the ABC dataset.
            - Feel free to adjust the seed for various results.

            <br>
            <br>
            <br>
            <br>
            <br>
            <br>
            <br>
            <br>
            """
            )
    
    def get_Chinese_note(self):
        return gr.Markdown(
            """
            **无条件输入的生成介绍:**
            
            - 我们通过从高斯分布中采样噪声生成**4**个 BRep 模型。
            - 模型基于 ABC 数据集训练，支持生成复杂度为 10~100 个曲面基元的模型。
            - 与最先进的 BRep 生成方法相比，HoLa-BRep 在 DeepCAD 数据集和 ABC 数据集上生成模型的有效性比率提升 20%-40%。
            - 可自由调整种子值以获取多样化生成结果。
            
            <br>
            <br>
            <br>
            <br>
            <br>
            <br>
            <br>
            <br>
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

    def get_English_note(self):
        return gr.Markdown(
            """
            **Note:**
            
            - Text can be either abstract or descriptive.  
            - We use a frozen sentence transformer to extract the feature from the text description.
            - While we use the existing Text2CAD dataset which contains more descriptive text, the out of distribution abstract text prompt also works.
            
            <br>
            <br>
            """
        )
    def get_Chinese_note(self):
        return gr.Markdown(
            """
            **基于文本输入条件的生成介绍:**
            
            - 文本输入可为抽象型或描述型。
            - 我们采用冻结的sentence transformer 从文本描述中提取特征。
            - 虽然当前使用包含更多描述性文本的现有 Text2CAD 数据集，但分布外的抽象文本提示同样适用。
            - **当前文本输入仅支持英文，敬请谅解。**
            
            <br>
            """
        )
    
    def get_input_components(self) -> List[gr.Component]:
        return [
            gr.Textbox(lines = 8,max_length=1024, label="Text"),
        ]


class PCLayout(AppLayout):

    def get_English_note(self):
        return gr.Markdown(
            """
            **Note:**
            
            - The input point cloud should be in .ply format with the position in -1~+1 and normal vectors.
            - The input point cloud can be either sparse or dense. We will downsample the point cloud into 2048 points.
            - We use a small and trainable PointNet++ to extract the feature from the point cloud.
            - This checkpoint is only for a clean point cloud without any noise. 
            - Point cloud contains less ambiguity and usually yields the best conditional generation results compared to other modalities.
            """
        )
    def get_Chinese_note(self):
        return gr.Markdown(
            """
            **基于点云输入条件的生成介绍:**
            
            - 输入点云需为包含坐标范围 -1~+1 及法向量的 .ply 格式文件。
            - 输入点云可稀疏或稠密，系统将自动下采样至 2048 个点。
            - 我们使用了一个小型可训练的PointNet++来从点云中提取特征。
            - 当前checkpoint仅适用于无噪声的点云。
            - 相较于其他模态，点云歧义性较低，通常能获得最佳条件生成结果。
            """
        )
    
    def get_input_components(self):
        return [
            gr.File(
                label='PC',
                file_count='single', 
            ),
        ]


class SketchLayout(AppLayout):

    def get_English_note(self):
        return gr.Markdown(
            """
            **Note:**
            
            - The input sketch is in 1:1 resolution and on a white background.
            - The input sketch should be a perspective projection rather than an orthogonal projection.
            - We use a frozen DINOv2 to extract the feature from the sketch image.
            - We obtained the training sketches using wireframe rendering in OpenCascade.
            
            <br>
            <br>
            """
        )

    def get_Chinese_note(self):
        return gr.Markdown(
            """
            **基于草图输入条件的生成介绍:**
            
            - 输入草图需为1：1分辨率且置于白色背景之上。
            - 输入草图应使用透视投影，而非正交投影。
            - 我们采用冻结的DINOv2从草图图像中提取特征。
            - 训练过程中使用的草图是经由 OpenCascade 中的线框渲染得到的。
            
            <br>
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

    def get_English_note(self):
        return gr.Markdown(
            """
            **Note:**
            
            - The input image is in 1:1 resolution and has a white background. 
            - Keep the object in grey for better generation results.
            - We use a frozen DINOv2 to extract the feature from the sketch image.
            - We obtained the training images using solid rendering in OpenCascade.
            
            <br>
            <br>
            """
        )

    def get_Chinese_note(self):
        return gr.Markdown(
            """
            **基于单视图输入条件的生成介绍:**
            - 输入图像需为1：1分辨率且具有白色背景。
            - 建议将物体保持为灰色以获得更佳生成效果。
            - 我们采用冻结的DINOv2从视图图像中提取特征。
            - 训练过程中使用的图像是经由 OpenCascade 中的实体渲染得到的。
            
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

    def get_English_note(self):
        return gr.Markdown(
            """
            **Note:**
            
            - Similar to the single-view condition, the input image should be in xxxx resolution and 4 fixed angles, see the examples below. 
            - Image features are extracted by a frozen DINOv2 and averaged after adding the positional encoding on the camera information. 
            """
        )
    
    def get_Chinese_note(self):
        return gr.Markdown(
            """
            **基于多视图输入条件的生成介绍:**
            
            - 与单视角输入条件类似，多视角输入图像需为1：1分辨率且包含 4 个固定视角的视图，请参考下方示例。
            - 我们采用冻结的DINOv2从视图图像中提取特征，并在添加相机的位置编码后执行取平均值的操作。
            
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
            
            gr.Image(
                label='View4',
                type='filepath', 
                interactive=True, 
                sources=["upload"]

            ),
        ]
