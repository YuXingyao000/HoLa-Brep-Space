#Frontend
import gradio as gr
from pathlib import Path
import os
from app.AppLayout import *
from app.GeneratingMethod import *
from app.ModelDirector import *
from app.DataProcessor import *

os.environ["HF_HOME"] = "/data/.huggingface"
os.environ["TORCH_HOME"] = "/data/.cache/torch"

# Theme
theme = gr.themes.Soft(
    primary_hue="slate",
    text_size="lg",
    font=['IBM Plex Sans', 'ui-sans-serif', 'system-ui', gr.themes.GoogleFont('sans-serif')],
).set(
    block_background_fill='*primary_200',
    button_primary_background_fill='*primary_100',
    body_background_fill='*secondary_50',
)

force_light = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

# 自定义CSS样式
custom_css = """
.gr-tabs.gr-tab-label {
    text-align: center;
}
button[role="tab"] {
    font-size: 20px;
}
div[role="tablist"] {
    height: var(--size-12);
}

#top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
#button-group {
    display: flex;
    gap: 10px;
}
.small-button {
    max-width: 80px;
    padding: 6px 10px;
    font-size: 14px;
}

@media (min-width: 1024px) {
    div[role="tablist"] {
        /* 电脑端居中 */
        justify-content: center;
    }
}

.tabs {
    margin-top: 20px;
}

.gradio-container-5-18-0 .prose p {
    line-height: unset;
}
div[data-testid="markdown"] span p:not(:first-child) {
    margin-top: unset;
}

/* 媒体查询 */
@media (max-width: 768px) {
    button[role="tab"] {
        font-size: 15px;
    }
    p.title1 {
        font-size: 45px !important;
        letter-spacing: unset !important;
    }
    p.title2 {
        font-size: 20px !important;
    }
    p.title3 {
        font-size: 12px !important;
    }
}
"""

DEMO_NUM = 4
WIREFRAME_FILE = 0
SOLID_FILE = 1
STEP_FILE = 2

BACKEND_CONDITION_DICT = {
    'Unconditional': 'uncond',
    'Point Cloud' : 'pc',
    'Text' : 'txt',
    'Sketch' : 'sketch',
    'SVR' : 'single_img',
    'MVR': 'multi_img'
}

# Dynamically registered functions 
def switch_model(user_state: dict, generate_mode: str,  model_index: int, offset: int):
    model_index =  (model_index + offset) % DEMO_NUM
    generate_mode = BACKEND_CONDITION_DICT[generate_mode]
    # Check if the condition has been generated
    if generate_mode not in user_state.keys():
        return model_index, gr.update(value='empty.obj', label=f'Wireframe{model_index + 1}'), gr.update(value='empty.obj', label=f'Solid{model_index + 1}'), gr.update( value=["app/examples/empty_examples/sample.stl", "app/examples/empty_examples/sample.ply", "app/examples/empty_examples/sample.step"], label=f'Models{model_index + 1}')
    
    # Check if model_index exceeds the number of current valid models 
    if model_index >= len(user_state[generate_mode]):
        return model_index, gr.update(value='empty.obj', label=f'Wireframe{model_index + 1}'), gr.update(value='empty.obj', label=f'Solid{model_index + 1}'), gr.update( value=["app/examples/empty_examples/sample.stl", "app/examples/empty_examples/sample.ply", "app/examples/empty_examples/sample.step"], label=f'Models{model_index + 1}')
    
    wireframe_model = user_state[generate_mode][model_index][WIREFRAME_FILE]
    solid_model = user_state[generate_mode][model_index][SOLID_FILE]
    if not os.path.exists(wireframe_model) or os.path.exists(solid_model):
        gr.Warning("The operation is too frequent!", title="Frequent Operation")
        return gr.update(), gr.update(), gr.update(), gr.update()
    return model_index, gr.Model3D(wireframe_model, label=f'Wireframe{model_index + 1}'), gr.Model3D(solid_model, label=f'Solid{model_index + 1}'), gr.Files(user_state[generate_mode][model_index], label=f'Models{model_index + 1}', interactive=False)
        
    
def set_generating_type(mode):
    return gr.Text(mode, visible=False)

def make_Chinese_descriptions(*text_component):
    return title_cn, description_cn, UncondLayout().get_Chinese_note(), PCLayout().get_Chinese_note(), SketchLayout().get_Chinese_note(), TextLayout().get_Chinese_note(), SVRLayout().get_Chinese_note(), MVRLayout().get_Chinese_note()

def make_English_descriptions(*text_component):
    return title_en, description_en, UncondLayout().get_English_note(), PCLayout().get_English_note(), SketchLayout().get_English_note(), TextLayout().get_English_note(), SVRLayout().get_English_note(), MVRLayout().get_English_note()

# Declarations for pre-rendering
model_solid = gr.Model3D(label=f'Solid1', value='empty.obj', key="Solid")
model_wireframe = gr.Model3D(label=f'Wireframe1', value='empty.obj', key="Wireframe")
step_file = gr.File(label=f'Step', file_count='single', file_types=['.step'], interactive=False, visible=False)
download_files = gr.Files(label=f"Models1", value=["app/examples/empty_examples/sample.stl", "app/examples/empty_examples/sample.ply", "app/examples/empty_examples/sample.step"], interactive=False, key="Downloads")

input_tab = gr.Tabs()

generating_type = gr.Text("Unconditional",visible=False)

title_en = gr.Markdown(
        """
        <style>
            .container-title{
                display: block;
                position: relative;
                text-align: center;
                text-rendering: optimizelegibility;
            }
        </style>        
        <h1 class="container-title">
            <p class='title1' style='font-size: 100px; text-align: center;'> 
            HoLa-BRep 
            </p> 
            <p class='title2' style='font-size: 25px; text-align: center;'>
            Holistic Latent Representation for BRep Generation
            <p class='title2' style='font-size: 25px; text-align: center;'>
            ACM Trans. on Graphics (SIGGRAPH 2025)
            </p>
            <p class='title2' style='font-size: 25px; text-align: center;'>
            Yilin Liu, Duoteng Xu, Xinyao Yu, Xiang Xu, Daniel Cohen-Or, Hao Zhang, Hui Huang*
            </p>
            </sp>
            <p class='title3' style='font-size: 20px; text-align: center;'>
            (Visual Computing Research Center, Shenzhen University)
            </sp>
        </h1>
        """
    )
title_cn = gr.Markdown(
        """
        <style>
            .container-title{
                display: block;
                position: relative;
                text-align: center;
                text-rendering: optimizelegibility;
            }
        </style>        
        <h1 class="container-title">
            <p class='title1' style='font-size: 100px; text-align: center;'> 
            HoLa-BRep 
            </p> 
            <p class='title2' style='font-size: 25px; text-align: center;'>
            Holistic Latent Representation for BRep Generation
            <p class='title2' style='font-size: 25px; text-align: center;'>
            ACM Trans. on Graphics (SIGGRAPH 2025)
            </p>
            <p class='title2' style='font-size: 25px; text-align: center;'>
            刘奕林, 许铎腾, 余星耀, 徐翔, Daniel Cohen-Or, 张皓, 黄惠*
            </p>
            </sp>
            <p class='title3' style='font-size: 20px; text-align: center;'>
            (深圳大学可视计算研究中心)
            </sp>
        </h1>
        """
    )

description_en = gr.Markdown(
            """
            # <h2>What is HoLa-BRep</h2>
            HoLa-BRep is a generative model that produces CAD models in boundary representation (BRep) based on various conditions, including point cloud, single-view image, multi-view images, single-view sketch or text description.
            It contains **1 unified** BRep variational encoder (VAE) to encode a BRep model's topological and geometric information into a holistic latent space, and a latent diffusion model (LDM) to generate such latent from multiple modalities. 
            Compared with the state-of-the-art method, HoLa-BRep only has 1 unified VAE and the corresponding latent space and 1 LDM for generation, so it is easier to train the model without any inter-dependency of the model. This is extremely useful when incorporating more modalities and even mix-modality training.

            # <h2>How to use it</h2>
            + Please refer to the example below for more details. You can also select the desired **modality** below and upload your own data. 
            + We generate **4** plausible BRep models for each input and visualize them in the 3D viewer. 
            + Feel free to explore the generated BRep models by rotating, zooming, and panning the 3D viewer, or **download** either the wireframe, surface mesh, or solid BRep model as STL or STEP files.
            """
        )
description_cn = gr.Markdown(
            """
            # <h2>What is HoLa-BRep</h2>
            HoLa-BRep 是一个生成模型，能够基于多种条件（包括点云、单视图图像、多视图图像、单视图草图或文本描述）生成以边界表示（BRep）形式呈现的 CAD 模型。
            它包含 1 个统一的 BRep 变分编码器（VAE），用于将 BRep 模型的拓扑与几何信息编码至结构化潜空间；以及 1 个潜在扩散模型（LDM），用于从多模态输入生成此类潜在表示。
            与最先进方法相比，HoLa-BRep 仅需 1 个统一 VAE 及其对应潜在空间、1 个 LDM 即可完成生成，因此无需处理模型间的相互依赖，训练过程更为简便。这一特性在整合更多模态甚至进行混合模态训练时具有显著优势。
            # <h2>How to use it</h2>
            + 更多细节请参考下方示例。您也可在下述选项中选取所需**模态**并上传自定义数据。
            + 我们为每个输入生成**4**个合理的 BRep 模型，并在 3D 查看器中可视化呈现。
            + 可通过旋转、缩放、平移 3D 查看器自由探索生成的 BRep 模型，或**下载**线框、曲面网格、实体 BRep 模型（格式为 STL 或 STEP 文件）。
            """
        )
descriptions = []

# Main body
with gr.Blocks(js=force_light, theme=theme, css=custom_css) as inference:
    with gr.Row(elem_id="top-row"):
        gr.HTML(
                    """
                    <div style="text-align: left;">
                        <a href="https://visitorbadge.io/status?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2FYuXingyao%2FHoLa-BRep">
                            <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2FYuXingyao%2FHoLa-BRep&labelColor=%23d9e3f0&countColor=%23555555" />
                        </a>
                    </div>
                    """
                )
        
        with gr.Row(elem_id="button-group"):
            btn_cn = gr.Button("中文", elem_classes="small-button")
            btn_en = gr.Button("English", elem_classes="small-button")
            btn_cn.click(fn=make_Chinese_descriptions, inputs=descriptions, outputs=descriptions)
            btn_en.click(fn=make_English_descriptions, inputs=descriptions, outputs=descriptions)
                
    title_en.render()
    descriptions.append(title_en)
    
    description_en.render()
    descriptions.append(description_en)
    
    user_state = gr.BrowserState({
            "user_id" : None,
            "user_output_dir" : None,
        })

    generating_type.render()
    
    with gr.Row():
        # Input Column
        with gr.Column() as input_col:
            with gr.Tabs() as input_tab:
                with gr.Tab("Unconditional") as uncond_tab: 
                    uncond_layout = UncondLayout()
                    uncond_description = uncond_layout.get_English_note()
                    descriptions.append(uncond_description)
                    uncond_input_components = uncond_layout.get_input_components()
                    
                    uncond_button = gr.Button("Generate")
                    uncond_button.click(
                        fn=UncondGeneratingMethod().generate(), 
                        inputs=[*uncond_input_components, user_state],
                        outputs=[model_wireframe, model_solid, step_file, download_files, user_state]
                    )
                    
                with gr.Tab("Point Cloud") as pc_tab:
                    pc_layout = PCLayout()
                    pc_description = pc_layout.get_English_note()
                    descriptions.append(pc_description)
                    pc_input_components = pc_layout.get_input_components()
                    
                    pc_button = gr.Button("Generate")
                    pc_button.click(
                        fn=ConditionedGeneratingMethod(PointCloudDirector(), PointCloudProcessor(), DEMO_NUM).generate(), 
                        inputs=[user_state, *pc_input_components],
                        outputs=[user_state, model_wireframe, model_solid, step_file, download_files]
                    )
                    
                with gr.Tab("Sketch") as sketch_tab:
                    sketch_layout = SketchLayout()
                    sketch_description = sketch_layout.get_English_note()
                    descriptions.append(sketch_description)
                    sketch_input_components = sketch_layout.get_input_components()
                    
                    sketch_button = gr.Button("Generate")
                    sketch_button.click(
                        fn=ConditionedGeneratingMethod(SketchDirector(), SingleImageProcessor(), DEMO_NUM).generate(), 
                        inputs=[user_state, *sketch_input_components],
                        outputs=[user_state, model_wireframe, model_solid, step_file, download_files]
                    )
                    
                with gr.Tab("Text") as text_tab:
                    text_layout = TextLayout()
                    text_description = text_layout.get_English_note()
                    descriptions.append(text_description)
                    text_input_components = text_layout.get_input_components()
                    
                    text_button = gr.Button("Generate")
                    text_button.click(
                        fn=ConditionedGeneratingMethod(TextDirector(), TextProcessor(), DEMO_NUM).generate(), 
                        inputs=[user_state, *text_input_components],
                        outputs=[user_state, model_wireframe, model_solid, step_file, download_files]
                    )
                    
                with gr.Tab("SVR") as svr_tab:
                    svr_layout = SVRLayout()
                    svr_description = svr_layout.get_English_note()
                    descriptions.append(svr_description)
                    svr_input_components = svr_layout.get_input_components()
                    
                    svr_button = gr.Button("Generate")
                    svr_button.click(
                        fn=ConditionedGeneratingMethod(SVRDirector(), SingleImageProcessor(), DEMO_NUM).generate(), 
                        inputs=[user_state, *svr_input_components],
                        outputs=[user_state, model_wireframe, model_solid, step_file, download_files]
                    )
                    
                with gr.Tab("MVR") as mvr_tab:
                    mvr_layout = MVRLayout()
                    mvr_description = mvr_layout.get_English_note()
                    descriptions.append(mvr_description)
                    with gr.Accordion("MVR input notification:", open=False):
                        gr.Image(value='app\examples\mvr.jpg',show_download_button=False, show_label=False,show_share_button=False,interactive=False)

                    with gr.Row():
                        mvr_input_components = mvr_layout.get_input_components()
                    mvr_button = gr.Button("Generate")
                    mvr_button.click(
                        fn=ConditionedGeneratingMethod(MVRDirector(), MultiImageProcessor(), DEMO_NUM).generate(), 
                        inputs=[user_state, *mvr_input_components],
                        outputs=[user_state, model_wireframe, model_solid, step_file, download_files]
                    )
                
                uncond_tab.select(fn=set_generating_type, inputs=gr.Text(uncond_tab.label, visible=False), outputs=generating_type)
                pc_tab.select(fn=set_generating_type, inputs=gr.Text(pc_tab.label, visible=False), outputs=generating_type)
                sketch_tab.select(fn=set_generating_type, inputs=gr.Text(sketch_tab.label, visible=False), outputs=generating_type)
                svr_tab.select(fn=set_generating_type, inputs=gr.Text(svr_tab.label, visible=False), outputs=generating_type)
                mvr_tab.select(fn=set_generating_type, inputs=gr.Text(mvr_tab.label, visible=False), outputs=generating_type)
                text_tab.select(fn=set_generating_type, inputs=gr.Text(text_tab.label, visible=False), outputs=generating_type)
                    
        # Output demonstration 
        with gr.Column() as output_col:        
            with gr.Tabs():
                with gr.Tab("Solid") as solid_tab:
                    model_solid.render()
                with gr.Tab("Wireframe") as wireframe_tab:
                    model_wireframe.render()
                with gr.Tab("Download") as download_tab:
                    step_file.render()
                    download_files.render()
                    
            
            
            model_index = gr.Number(value=0, visible=False)
            with gr.Row() as switch_row:
                last_button = gr.Button("Last")
                next_button = gr.Button("Next")
                
            last_button.click(
                fn=switch_model,
                inputs=[user_state, generating_type, model_index, gr.Number(-1, visible=False)],
                outputs=[model_index, model_wireframe, model_solid, download_files])
            next_button.click(
                fn=switch_model,
                inputs=[user_state, generating_type, model_index, gr.Number(1, visible=False)],
                outputs=[model_index, model_wireframe, model_solid, download_files])
                
    # Examples
    @gr.render(inputs=[generating_type], triggers=[generating_type.change, inference.load])
    def show_examples(generate_mode):
        if generate_mode == "Unconditional":
            pass
        
        elif generate_mode == "Point Cloud":
            pc_samples=[
                            [Path("app/examples/pc_examples") / sample_number / "pc.png"] for sample_number in os.listdir("app/examples/pc_examples") if sample_number != "take_photo.py"
                        ]
            with gr.Row():
                def dummy_pc_func(pic_path):
                    return Path(pic_path[0]).with_suffix(".ply").as_posix()
                for i in range(len(pc_samples)):
                    with gr.Column(min_width=100):
                        dummy_image = gr.Image(type="filepath", format="png", visible=False)
                        point_cloud_data = gr.Dataset(
                            label=f"Example{i+1}",
                            components=[dummy_image],
                            samples=[pc_samples[i]],
                            layout="table"
                        )
                        point_cloud_data.click(dummy_pc_func, inputs=point_cloud_data, outputs=pc_input_components)
                        
        elif generate_mode == "Text":
            text_data = gr.Dataset(
                components=text_input_components,
                samples=[
                    ["The object is a rectangular prism with two protruding L-shaped sections on opposite sides."],
                    ["This design creates a rectangular plate with rounded edges. The plate measures about 0.3214 units in length, 0.75 units in width, and 0.0429 units in height. The rounded edges give the plate a smooth, aesthetically pleasing appearance."],
                    ["The U-shaped bracket has a flat top and a curved bottom. The design begins by creating a new coordinate system with specific Euler angles and a translation vector. A two-dimensional sketch is then drawn, forming a complex shape with multiple lines and arcs. This sketch is scaled down, rotated, and translated to align with the coordinate system. The sketch is extruded to create a three-dimensional model. The final dimensions of the bracket are approximately 0.7 units in length, 0.75 units in width, and 0.19 units in height. The bracket is designed to integrate seamlessly with other components, providing a sturdy and functional structure."]
                    ],
                layout='table',
                label="Examples",
                headers=["Prompt"]
            )
            def dummy_func(text):
                return gr.Text(text[0])
            text_data.click(fn=dummy_func, inputs=text_data, outputs=text_input_components)
            
        elif generate_mode == "Sketch":
            with gr.Row():
                for i in range(10):
                    with gr.Column(min_width=100):
                        example = gr.Examples(
                            inputs=sketch_input_components,
                            examples=[
                                [f"app/examples/sketch_examples/{i + 1}.png"]
                                ],
                            label=f"Example{i+1}"
                            )
                        
        elif generate_mode == "SVR":
            with gr.Row():
                for i in range(12):
                    with gr.Column(min_width=100):
                        example = gr.Examples(
                            inputs=svr_input_components,
                            examples=[
                                [f"app/examples/svr_examples/{i + 1}.png"]
                                ],
                            label=f"Example{i+1}"
                            )
                       
        elif generate_mode == "MVR":
            with gr.Row():
                for i in range(4):
                    file_num = ["00053073", "00033625", "00052220", "00039769"]
                    with gr.Column():
                        example = gr.Examples(
                            inputs=mvr_input_components,
                            examples=[
                                [f"app/examples/mvr_examples/{file_num[i]}_img0.png", f"app/examples/mvr_examples/{file_num[i]}_img1.png", f"app/examples/mvr_examples/{file_num[i]}_img2.png", f"app/examples/mvr_examples/{file_num[i]}_img3.png"], 
                                ],
                            label=f"Example{i+1}"
                        )
    gr.Markdown(
        value=
        """
        <h2>Citation</h2>
        
        If our work is helpful for your research or applications, please cite us via:
        <br>
        ```
        @article{HolaBRep25,
        title={HoLa: B-Rep Generation using a Holistic Latent Representation},
        author={Yilin Liu and Duoteng Xu and Xinyao Yu and Xiang Xu and Daniel Cohen-Or and Hao Zhang and Hui Huang},
        journal={ACM Transactions on Graphics (Proceedings of SIGGRAPH)},
        volume={44},
        number={4},
        year={2025},
        }
        ```
        """,
        height=300,
        )
                

if __name__ == "__main__":
    inference.launch(allowed_paths=['/data/outputs'], server_port=7860)