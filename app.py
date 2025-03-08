import gradio as gr
from textual_inversion import TextualInversion

display_choices = ["cat toy", "dragon born", "birb style", "pool rooms", "minecraft concept art"]
repo_id_embeds=["sd-concepts-library/cat-toy", "sd-concepts-library/dragonborn", "sd-concepts-library/birb-style", "sd-concepts-library/poolrooms", "sd-concepts-library/minecraft-concept-art"]

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
    examples=[["a graffiti in a favela wall", display_choices[0]]],
)

# Launch the app
demo.launch()