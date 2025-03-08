import gradio as gr
from textual_inversion import TextualInversion

display_choices = ["minecraft concept art", "dragon born", "birb style", "pool rooms", "matrix"]
repo_id_embeds=["sd-concepts-library/minecraft-concept-art::with <minecraft-concept-art> concept",
                "sd-concepts-library/dragonborn::with <dragonborn> concept", 
                "sd-concepts-library/birb-style::in <birb-style> concept", 
                "sd-concepts-library/poolrooms::with <poolrooms> concept", 
                "sd-concepts-library/matrix::in <hatman-matrix> world"
                ]

textualInversion = TextualInversion(pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4", repo_id_embeds=repo_id_embeds)

def generate_image(prompt, selected_concept):
    return textualInversion.generate_image(prompt, display_choices.index(selected_concept))

demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Dropdown(choices=display_choices, label="Select concept", value=display_choices[0])
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Textual Inversion Image Generator",
    description="Generate images using textual inversion concepts",
    examples=[["a flying dog", display_choices[0]]],
    allow_flagging=False
)

# Launch the app
demo.launch()