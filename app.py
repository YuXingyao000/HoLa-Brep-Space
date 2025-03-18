import gradio as gr
import os
from GenerateTab import GenerateTab, generate_tab, delegate_generate_method

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

def get_last_model_path(state_dict, generate_mode: str, index: int):
    """
    Read Only
    """
    index = (index - 1) % 3
    if not state_dict[generate_mode]:
        return None, None, None, None, None, None
    if f'Model{index}' not in state_dict[generate_mode].keys():
        return None, None, None, None, None, None
    else:
        return index, *state_dict[generate_mode][f'Model{index}'], *state_dict[generate_mode][f'Model{index}']

def get_next_model_path(state_dict, generate_mode: str, index: int):
    """
    Read Only
    """
    index = (index + 1) % 3
    if not state_dict[generate_mode]:
        return None, None, None, None, None, None
    if f'Model{index}' not in state_dict[generate_mode].keys():
        return None, None, None, None, None, None
    else:
        return index, *state_dict[generate_mode][f'Model{index}'], *state_dict[generate_mode][f'Model{index}']
        

with gr.Blocks(js=force_light, theme=theme, css=custom_css) as inference:
    # Title
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
        
    GenerateTab.static_state = user_state
    
    with gr.Row():
        input_component = []
        output_components = []
        model_path1, model_path2, model_path3 = None, None, None
        with gr.Column():
            generate_mode = gr.Radio(['Unconditional', 'Point Cloud', 'Sketch', 'Text'], type='value', label="Choose a generating method")
            @gr.render(inputs=[generate_mode, user_state], triggers=[generate_mode.select])
            def show_input(generate_mode, local_storage):
                if generate_mode == 'Text':
                    with gr.Row():
                        gr.Markdown(
                            """
                            <p style='color: orange; font-weight: bolder'>
                            We sincerely apologize, but we currently only support English.
                            </p>
                            """
                        )
                with gr.Row():
                    gr.Markdown("Some descriptions here")
                with gr.Row():
                    input_component = generate_tab(generate_mode, user_state).get_input_components()
                    for comp in input_component:
                        comp
                        
                with gr.Row():
                    button = gr.Button("Generate")
                    button.click(
                        fn=delegate_generate_method(generate_mode), 
                        inputs=[*input_component, user_state],
                        outputs=[*output_components, user_state]
                        )
                
        with gr.Column():
            # No use
            output_components = [
                gr.Model3D(visible=False), gr.Model3D(visible=False), gr.File(visible=False),  # model 1
                gr.Model3D(visible=False), gr.Model3D(visible=False), gr.File(visible=False),  # model 2
                gr.Model3D(visible=False), gr.Model3D(visible=False), gr.File(visible=False)   # model 3
                ]
            with gr.Column():
                model_num = gr.Number(value=0, visible=False)
                # model_path1, model_path2, model_path3 = get_generated_model_path(local_storage, generate_mode, i + 1)
                with gr.Tabs():
                    with gr.Tab("Solid"):
                        model_solid = gr.Model3D(label=f'Solid')
                    with gr.Tab("Wireframe"):
                        model_wireframe = gr.Model3D(label=f'Wireframe')
                    with gr.Tab("Download"):
                        step_file = gr.File(label=f'Step', file_count='single', file_types=['.step'], interactive=False, visible=False)
                        download_files = gr.File(label="Files", file_count='multiple', file_types=['.obj', '.stl', '.step'], interactive=False)
    
                with gr.Row():
                    last_button = gr.Button("Last")
                    next_button = gr.Button("Next")
                    
                    last_button.click(
                        fn=get_last_model_path,
                        inputs=[user_state, generate_mode, model_num],
                        outputs=[model_num, model_wireframe, model_solid, step_file, download_files])
                    next_button.click(
                        fn=get_next_model_path,
                        inputs=[user_state, generate_mode, model_num],
                        outputs=[model_num, model_wireframe, model_solid, step_file, download_files])
                        
                    # tab.select(fn = get_generated_model_path, inputs=[user_state, generate_mode, gr.Number(i+1, visible=False)], outputs=[model_wireframe, model_solid, step_file, download_files])


    
        

if __name__ == "__main__":
    inference.launch()
