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


# Declarations for pre-rendering
model_solid = gr.Model3D(label=f'Solid1', value='empty.obj', key="Solid")
model_wireframe = gr.Model3D(label=f'Wireframe1', value='empty.obj', key="Wireframe")
step_file = gr.File(label=f'Step', file_count='single', file_types=['.step'], interactive=False, visible=False)
download_files = gr.Files(label=f"Models1", value=["app/examples/empty_examples/sample.stl", "app/examples/empty_examples/sample.ply", "app/examples/empty_examples/sample.step"], interactive=False, key="Downloads")

input_tab = gr.Tabs()

generating_type = gr.Text("Unconditional",visible=False)


# Main body
with gr.Blocks(js=force_light, theme=theme, css=custom_css) as inference:

    gr.Markdown(
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
            </sp>
            <p class='title3' style='font-size: 20px; text-align: center;'>
            (Visual Computing Research Center, Shenzhen University)
            </sp>
        </h1>
        """
    )

    gr.Markdown(
            """
            # <h2>What is HoLa-BRep</h2>
            HoLa-BRep contains a BRep VAE to encode a BRep model's topological and geometric information into a unified, holistic latent space and a latent diffusion model to generate holistic latent from multiple modalities. It can turn point clouds, single-view images, multi-view images, 2D sketches, or text prompts into solid BRep models. 
            # <h2>How to use it</h2>
            + Please refer to the example below for more details. You can also select the desired **modality** below and upload your own data. 
            + We generate **4** plausible BRep models for each input and visualize them in the 3D viewer. 
            + Feel free to explore the generated BRep models by rotating, zooming, and panning the 3D viewer, or **download** either the wireframe, surface mesh, or solid BRep model as OBJ or STEP files.
            """
        )
    
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
                    uncond_layout.get_note()
                    uncond_input_components = uncond_layout.get_input_components()
                    
                    uncond_button = gr.Button("Generate")
                    uncond_button.click(
                        fn=UncondGeneratingMethod().generate(), 
                        inputs=[*uncond_input_components, user_state],
                        outputs=[model_wireframe, model_solid, step_file, download_files, user_state]
                    )
                    
                with gr.Tab("Point Cloud") as pc_tab:
                    pc_layout = PCLayout()
                    pc_layout.get_note()
                    pc_input_components = pc_layout.get_input_components()
                    
                    
                    pc_button = gr.Button("Generate")
                    pc_button.click(
                        fn=ConditionedGeneratingMethod(PointCloudDirector(), PointCloudProcessor(), DEMO_NUM).generate(), 
                        inputs=[user_state, *pc_input_components],
                        outputs=[user_state, model_wireframe, model_solid, step_file, download_files]
                    )
                    
                with gr.Tab("Sketch") as sketch_tab:
                    sketch_layout = SketchLayout()
                    sketch_layout.get_note()
                    sketch_input_components = sketch_layout.get_input_components()
                    
                    sketch_button = gr.Button("Generate")
                    sketch_button.click(
                        fn=ConditionedGeneratingMethod(SketchDirector(), SingleImageProcessor(), DEMO_NUM).generate(), 
                        inputs=[user_state, *sketch_input_components],
                        outputs=[user_state, model_wireframe, model_solid, step_file, download_files]
                    )
                    
                with gr.Tab("Text") as text_tab:
                    text_layout = TextLayout()
                    text_layout.get_note()
                    text_input_components = text_layout.get_input_components()
                    
                    text_button = gr.Button("Generate")
                    text_button.click(
                        fn=ConditionedGeneratingMethod(TextDirector(), TextProcessor(), DEMO_NUM).generate(), 
                        inputs=[user_state, *text_input_components],
                        outputs=[user_state, model_wireframe, model_solid, step_file, download_files]
                    )
                    
                with gr.Tab("SVR") as svr_tab:
                    svr_layout = SVRLayout()
                    svr_layout.get_note()
                    svr_input_components = svr_layout.get_input_components()
                    
                    svr_button = gr.Button("Generate")
                    svr_button.click(
                        fn=ConditionedGeneratingMethod(SVRDirector(), SingleImageProcessor(), DEMO_NUM).generate(), 
                        inputs=[user_state, *svr_input_components],
                        outputs=[user_state, model_wireframe, model_solid, step_file, download_files]
                    )
                    
                with gr.Tab("MVR") as mvr_tab:
                    mvr_layout = MVRLayout()
                    mvr_layout.get_note()
                    with gr.Accordion("Some MVR input notification:", open=False):
                        gr.Markdown(
                            """
                            - Input 1:
                            - Input 2:
                            - Input 3:
                            - Input 4:
                            """
                            )
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

                
            
    gr.HTML(
        """
        <div style="text-align: center; margin-top: 20px;">
            <a href="https://visitorbadge.io/status?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2FYuXingyao%2FHoLa-BRep">
                <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2FYuXingyao%2FHoLa-BRep&labelColor=%23d9e3f0&countColor=%23555555" />
            </a>
        </div>
        """
    )

if __name__ == "__main__":
    inference.launch(allowed_paths=['/data/outputs'], server_name="0.0.0.0", server_port=7860)