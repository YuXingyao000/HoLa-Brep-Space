import gradio as gr
import os
from app.app_layout import AppLayout, build_layout
from app.generate_method import delegate_generate_method

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
    }
    p.title2 {
        font-size: 20px !important;
    }
    p.title3 {
        font-size: 12px !important;
    }
}
"""

PRESENT_NUM = 4
def get_model(generate_mode: str, present_mode: str, model_index: int, user_state: dict):
    models: dict = user_state[generate_mode]
    if f"Model{model_index + 1}" not in models.keys():
        if present_mode != "Downloads":
            return gr.Model3D(label=f"{present_mode}{model_index + 1}",value='app/empty.obj',display_mode=present_mode.lower(), key=present_mode)
        else:
            return gr.Files(label=f"Models{model_index + 1}", value=["app/sample.stl", "app/sample.ply", "app/app/sample.step"], interactive=False, key=present_mode)
    else:
        if present_mode == "Wireframe":
            return gr.Model3D(label=f"{present_mode}{model_index + 1}",value=models[f'Model{model_index + 1}'][0],display_mode=present_mode.lower(), key=present_mode)
        elif present_mode == "Solid":
            return gr.Model3D(label=f"{present_mode}{model_index + 1}",value=models[f'Model{model_index + 1}'][1],display_mode=present_mode.lower(), key=present_mode)
        else:
            return gr.Files(label=f"Models{model_index + 1}", value=models[f'Model{model_index + 1}'], file_types=['.obj', '.stl', '.step'],interactive=False, key=present_mode)

def get_last_model(generate_mode: str, present_mode: str,  model_index: int, user_state: dict):
    model_index = (model_index - 1) % PRESENT_NUM
    solid = get_model(generate_mode, "Solid", model_index, user_state)
    wire = get_model(generate_mode, "Wireframe", model_index, user_state)
    downloads = get_model(generate_mode, "Downloads", model_index, user_state)
    return model_index, wire, solid, downloads

def get_next_model(generate_mode: str, present_mode: str,  model_index: int, user_state: dict):
    model_index = (model_index + 1) % PRESENT_NUM
    solid = get_model(generate_mode, "Solid", model_index, user_state)
    wire = get_model(generate_mode, "Wireframe", model_index, user_state)
    downloads = get_model(generate_mode, "Downloads", model_index, user_state)
    return model_index, wire, solid, downloads

# generate_mode = gr.Radio(['Unconditional', 'Point Cloud', 'Sketch', 'Text', 'SVR', 'MVR'], type='value', value="Unconditional", label="Choose a generating method")
# present_mode = gr.Radio(['Wireframe', 'Solid', 'Downloads'], type='value', value="Solid", label="Preview:")
# generate_button = gr.Button("Generate")
model_solid = gr.Model3D(label=f'Solid1', value='app/empty.obj', key="Solid")
model_wireframe = gr.Model3D(label=f'Wireframe1', value='app/empty.obj', key="Wireframe")
step_file = gr.File(label=f'Step', file_count='single', file_types=['.step'], interactive=False, visible=False)
download_files = gr.Files(label=f"Models1", value=["app/sample.stl", "app/sample.ply", "app/sample.step"], interactive=False, key="Downloads")

input_tab = gr.Tabs()

generating = gr.Text("Unconditional",visible=False)
presenting = gr.Text("Solid", visible=False)

uncond_tab = gr.Tab("Unconditional") 
pc_tab = gr.Tab("Point Cloud")
sketch_tab = gr.Tab("Sketch")
text_tab = gr.Tab("Text")
svr_tab = gr.Tab("SVR")
mvr_tab = gr.Tab("MVR")


def set_generate_mode(mode):
    return gr.Text(mode, visible=False)
    
def set_present_mode(mode):
    return gr.Text(mode, visible=False)

with gr.Blocks(js=force_light, theme=theme, css=custom_css) as inference:
    # Title
    # gr.HTML(
    #     """
    #     <div class="header-container">
    #       <div>
    #           <p id="HoLa-Icon">👋</p>
    #       </div>
    #       <div>
    #           <h1>HoLa-BRep</h1>
    #           <p>Holistic Latent Representation for B-Rep Generation</p>
    #           <p>(Visual Computing Research Center, Shenzhen University)</p>
    #       </div>
    #     </div>
    #     """
    #     )
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
            <p class='title1' style='font-size: 100px; text-align: center; letter-spacing: 10px;'> 
            HoLa-BRep 
            </p> 
            <p class='title2' style='font-size: 25px; text-align: center;'>
            Holistic Latent Representation for B-Rep Generation
            </sp>
            <p class='title3' style='font-size: 20px; text-align: center;'>
            (Visual Computing Research Center, Shenzhen University)
            </sp>
        </h1>
        """
    )
    
    # with gr.Accordion("👋About HoLa-BRep", open=False):
    gr.Markdown(
            """
            # ❔What is HoLa-BRep
            HoLa-BRep contains a B-rep VAE to encode a B-rep model's topological and geometric information into a unified, holistic latent space and a latent diffusion model to generate holistic latent from multiple modalities. It can turn point clouds, single-view images, multi-view images, 2D sketches, or text prompts into solid B-rep models. 
            # ✨How to use it
            + Please refer to the example below for more details. You can also select the desired **modality** below and upload your own data. 
            + We generate **4** plausible B-rep models for each input and visualize them in the 3D viewer. 
            + Feel free to explore the generated B-rep models by rotating, zooming, and panning the 3D viewer, or **download** either the wireframe, surface mesh, or solid B-rep model as OBJ or STEP files.
            """
        )
    
    user_state = gr.BrowserState(
        {
            "user_id" : None,
            "user_output_dir" : None,
            "Unconditional" : dict(),
            "Point Cloud" : dict(),
            "Sketch" : dict(),
            "Text" : dict(),
            "SVR" : dict(),
            "MVR" : dict(),
        }
        )
    AppLayout.static_state = user_state
    generating.render()
    presenting.render()
    
    
    with gr.Row():
        with gr.Column() as input_col:
            with gr.Tabs() as input_tab:
                with gr.Tab("Unconditional") as uncond_tab: 
                    uncond_layout = build_layout(uncond_tab.label, user_state)
                    uncond_layout.get_note()
                    uncond_input_components = uncond_layout.get_input_components()
                    uncond_button = gr.Button("Generate")
                    uncond_button.click(
                        fn=delegate_generate_method(uncond_tab.label, user_state), 
                        inputs=[*uncond_input_components, user_state],
                        outputs=[model_wireframe, model_solid, step_file, download_files, user_state]
                    )
                    
                with gr.Tab("Point Cloud") as pc_tab:
                    pc_layout = build_layout(pc_tab.label, user_state)
                    pc_layout.get_note()
                    pc_input_components = pc_layout.get_input_components()
                    pc_button = gr.Button("Generate")
                    pc_button.click(
                        fn=delegate_generate_method(pc_tab.label, user_state), 
                        inputs=[*pc_input_components, user_state],
                        outputs=[model_wireframe, model_solid, step_file, download_files, user_state]
                    )
                    
                with gr.Tab("Sketch") as sketch_tab:
                    sketch_layout = build_layout(sketch_tab.label, user_state)
                    sketch_layout.get_note()
                    sketch_input_components = sketch_layout.get_input_components()
                    sketch_button = gr.Button("Generate")
                    sketch_button.click(
                        fn=delegate_generate_method(sketch_tab.label, user_state), 
                        inputs=[*sketch_input_components, user_state],
                        outputs=[model_wireframe, model_solid, step_file, download_files, user_state]
                    )
                    
                with gr.Tab("Text") as text_tab:
                    text_layout = build_layout(text_tab.label, user_state)
                    text_layout.get_note()
                    text_input_components = text_layout.get_input_components()
                    text_button = gr.Button("Generate")
                    text_button.click(
                        fn=delegate_generate_method(text_tab.label, user_state), 
                        inputs=[*text_input_components, user_state],
                        outputs=[model_wireframe, model_solid, step_file, download_files, user_state]
                    )
                    
                with gr.Tab("SVR") as svr_tab:
                    svr_layout = build_layout(sketch_tab.label, user_state)
                    svr_layout.get_note()
                    svr_input_components = svr_layout.get_input_components()
                    svr_button = gr.Button("Generate")
                    svr_button.click(
                        fn=delegate_generate_method(svr_tab.label, user_state), 
                        inputs=[*svr_input_components, user_state],
                        outputs=[model_wireframe, model_solid, step_file, download_files, user_state]
                    )
                    
                with gr.Tab("MVR") as mvr_tab:
                    mvr_layout = build_layout(mvr_tab.label, user_state)
                    mvr_layout.get_note()
                    with gr.Row():
                        mvr_input_components = mvr_layout.get_input_components()
                    mvr_button = gr.Button("Generate")
                    mvr_button.click(
                        fn=delegate_generate_method(mvr_tab.label, user_state), 
                        inputs=[*mvr_input_components, user_state],
                        outputs=[model_wireframe, model_solid, step_file, download_files, user_state]
                    )
                
                uncond_tab.select(fn=set_generate_mode, inputs=gr.Text(uncond_tab.label, visible=False), outputs=generating)
                pc_tab.select(fn=set_generate_mode, inputs=gr.Text(pc_tab.label, visible=False), outputs=generating)
                sketch_tab.select(fn=set_generate_mode, inputs=gr.Text(sketch_tab.label, visible=False), outputs=generating)
                svr_tab.select(fn=set_generate_mode, inputs=gr.Text(svr_tab.label, visible=False), outputs=generating)
                mvr_tab.select(fn=set_generate_mode, inputs=gr.Text(mvr_tab.label, visible=False), outputs=generating)
                text_tab.select(fn=set_generate_mode, inputs=gr.Text(text_tab.label, visible=False), outputs=generating)
                    
            
        with gr.Column() as output_col:        
            with gr.Tabs():
                with gr.Tab("Solid") as solid_tab:
                    model_solid.render()
                with gr.Tab("Wireframe") as wireframe_tab:
                    model_wireframe.render()
                with gr.Tab("Download") as download_tab:
                    gr.Markdown(
                        """
                        # This is a title
                        + Something here
                        
                        + Something here
                        
                        + Something here
                        
                        + Something here
                        
                        + Something here
                        """
                        )
                    step_file.render()
                    download_files.render()
            
            solid_tab.select(fn=set_present_mode, inputs=gr.Text(solid_tab.label, visible=False))
            wireframe_tab.select(fn=set_present_mode, inputs=gr.Text(wireframe_tab.label, visible=False))
            download_tab.select(fn=set_present_mode, inputs=gr.Text(download_tab.label, visible=False))
            
            model_index = gr.Number(value=0, visible=False)
            with gr.Row() as switch_row:
                last_button = gr.Button("Last")
                next_button = gr.Button("Next")
                last_button.click(
                    fn=get_last_model,
                    inputs=[generating, gr.Text(presenting, visible=False), model_index, user_state],
                    outputs=[model_index, model_wireframe, model_solid, download_files])
                next_button.click(
                    fn=get_next_model,
                    inputs=[generating, gr.Text(presenting, visible=False), model_index, user_state],
                    outputs=[model_index, model_wireframe, model_solid, download_files])
            
            # generate_mode.render()
            # @gr.render(inputs=[generate_mode, user_state], triggers=[generate_mode.select, inference.load])
            # def show_input(generate_mode, local_storage):
            #     input_component = []
            #     if generate_mode == 'Text':
            #         gr.Info("We sincerely apologize, but we currently only support English.")

                
            #     layout = build_layout(generate_mode, user_state)
            #     layout.get_note()
                
            #     with gr.Row():
            #         input_component = layout.get_input_components()
                
            #     generate_button.unrender()
            #     generate_button.render()
            #     generate_button.click(
            #             fn=delegate_generate_method(generate_mode, user_state), 
            #             inputs=[*input_component, user_state],
            #             outputs=[model_wireframe, model_solid, step_file, download_files, user_state]
            #             )
    
                    
                    
            # with gr.Row() as switch_row:
            #     last_button = gr.Button("Last")
            #     next_button = gr.Button("Next")
            #     last_button.click(
            #         fn=get_last_model,
            #         inputs=[generate_mode, model_num, user_state],
            #         outputs=[model_wireframe, model_solid, download_files, model_num])
            #     next_button.click(
            #         fn=get_next_model,
            #         inputs=[generate_mode, model_num, user_state],
            #         outputs=[model_wireframe, model_solid, download_files, model_num])
            
        # with gr.Column() as output_col:
        #     present_mode = gr.Radio(["Solid", "Wireframe", "Downloads"], value="Solid", label="Preview")
        #     @gr.render(inputs=[present_mode, user_state], triggers=[inference.load, present_mode.select])
        #     def show_output(present_mode, local_state):
        #     # with gr.Tabs():
        #     #     with gr.Tab("Solid"):
        #     #         model_solid.render()
        #     #     with gr.Tab("Wireframe"):
        #     #         model_wireframe.render()
        #     #     with gr.Tab("Download",):
        #     #         step_file.render()
        #     #         download_files.render()
        #         if present_mode == "Solid":
        #             model_solid.unrender()
        #             model_solid.render()
        #         elif present_mode == "Wireframe":
        #             model_wireframe.unrender()
        #             model_wireframe.render()
        #         elif present_mode == "Downloads":
        #             download_files.unrender()
        #             download_files.render()
                
                    
        #     with gr.Row() as switch_row:
        #         model_num = gr.Number(value=0, visible=False)
        #         last_button = gr.Button("Last")
        #         next_button = gr.Button("Next")
        #         last_button.click(
        #             fn=get_last_model_path,
        #             inputs=[user_state, generate_mode, model_num],
        #             outputs=[model_num, model_wireframe, model_solid, download_files])
        #         next_button.click(
        #             fn=get_next_model_path,
        #             inputs=[user_state, generate_mode, model_num],
        #             outputs=[model_num, model_wireframe, model_solid, download_files])
                
            
        # Output
        # with gr.Column() as output_col:        
        #     with gr.Tabs():
        #         with gr.Tab("Solid"):
        #             model_solid.render()
        #         with gr.Tab("Wireframe"):
        #             model_wireframe.render()
        #         with gr.Tab("Download",):
        #             step_file.render()
        #             download_files.render()
            
        #     with gr.Row() as switch_row:
        #         model_num = gr.Number(value=0, visible=False)
        #         last_button = gr.Button("Last")
        #         next_button = gr.Button("Next")
        #         last_button.click(
        #             fn=get_last_model_path,
        #             inputs=[user_state, generate_mode, model_num],
        #             outputs=[model_num, model_wireframe, model_solid, download_files])
        #         next_button.click(
        #             fn=get_next_model_path,
        #             inputs=[user_state, generate_mode, model_num],
        #             outputs=[model_num, model_wireframe, model_solid, download_files])

    @gr.render(inputs=[generating], triggers=[generating.change, inference.load])
    def show_examples(generate_mode):
        if generate_mode == "Unconditional":
            pass
        elif generate_mode == "Point Cloud":
            example = gr.Examples(
                inputs=pc_input_components,
                examples=[
                    [r"D:\HoLa-Brep\data\to_upload\organized_data\00000061\pc.ply",],
                    [r"D:\HoLa-Brep\data\to_upload\organized_data\00000070\pc.ply",],
                    [r"D:\HoLa-Brep\data\to_upload\organized_data\00000093\pc.ply",],
                    [r"D:\HoLa-Brep\data\to_upload\organized_data\00000168\pc.ply",],
                    [r"D:\HoLa-Brep\data\to_upload\organized_data\00000178\pc.ply",],
                    [r"D:\HoLa-Brep\data\to_upload\organized_data\00000329\pc.ply",],
                ]
            )
        elif generate_mode == "Text":
            example = gr.Examples(
                inputs=text_input_components,
                examples=[
                    ["A ring"],
                    ["A box"],
                    ["A cat"]
                    ]
            )
        elif generate_mode == "Sketch":
            example = gr.Examples(
                inputs=sketch_input_components,
                examples=[
                    ["app/s2.png"], ["app/s3.png"]
                    ]
            )
        elif generate_mode == "SVR":
            with gr.Row(variant="panel"):
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s1.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s2.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s3.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s1.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s2.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s3.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s1.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s2.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s3.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s3.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s3.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s3.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s3.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s3.png"]
                            ],
                        label=""
                        )
                with gr.Column(min_width=100):
                    example = gr.Examples(
                        inputs=svr_input_components,
                        examples=[
                            ["app/s3.png"]
                            ],
                        label=""
                        )
                
        elif generate_mode == "MVR":
            with gr.Row():
                with gr.Column():
                    example = gr.Examples(
                        inputs=mvr_input_components,
                        examples=[
                            ["app/s1.png", "app/s2.png", "app/s3.png"]
                            ]
                    )
                with gr.Column():
                    example = gr.Examples(
                        inputs=mvr_input_components,
                        examples=[
                            ["app/s1.png", "app/s1.png", "app/s1.png"]
                            ]
                    )
                with gr.Column():
                    example = gr.Examples(
                        inputs=mvr_input_components,
                        examples=[
                            ["app/s1.png", "app/s2.png", "app/s3.png"]
                            ]
                    )
                with gr.Column():
                    example = gr.Examples(
                        inputs=mvr_input_components,
                        examples=[
                            ["app/s1.png", "app/s2.png", "app/s3.png"]
                            ]
                    )
                with gr.Column():
                    example = gr.Examples(
                        inputs=mvr_input_components,
                        examples=[
                            ["app/s1.png", "app/app/s2.png", "app/s3.png"]
                            ]
                    )
                
            
    gr.HTML(
        """
        <div style="text-align: center; margin-top: 20px;">
            <a href="https://visitorbadge.io/status?path=http%3A%2F%2F127.0.0.1%3A7860%2F"><img src="https://api.visitorbadge.io/api/visitors?path=http%3A%2F%2F127.0.0.1%3A7860%2F&labelColor=%23d9e3f0&countColor=%23697689" />
            </a>
        </div>
        """
    )

if __name__ == "__main__":
    inference.launch(allowed_paths=[r'D:\HoLa-Brep\data\to_upload\organized_data'])
