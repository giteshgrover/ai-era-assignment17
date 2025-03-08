#@title Import required libraries
import os
import torch

import PIL
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

class TextualInversion:
    def __init__(self, pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4", repo_id_embeds=["sd-concepts-library/matrix::with <hatman-matrix> concept"]):
        #@markdown `pretrained_model_name_or_path` which Stable Diffusion checkpoint you want to use. This should match the one used for training the embeddings.
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        #@title Load your concept here
    #@markdown Enter the `repo_id` for a concept you like (you can find pre-learned concepts in the public [SD Concepts Library](https://huggingface.co/sd-concepts-library))
        self.repo_id_embeds = [x.split("::")[0] for x in repo_id_embeds]
        self.prompts_suffixes = [x.split("::")[1] for x in repo_id_embeds]
       
       # Set device
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if "mps" == self.device: os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

        #@title Load the Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float16
        ).to(self.device)

        self.generators = []
        # self.textInversionPrompts = []
        for index,repo_id in enumerate(self.repo_id_embeds):
            #@title Load the concept into pipeline
            self.pipe.load_textual_inversion(repo_id)
            self.generators.append(torch.Generator(device=self.device).manual_seed(index + 11))
            # self.textInversionPrompts.append(f" with <{repo_id}> on it")
        #@title Run the Stable Diffusion pipeline
        #@markdown Don't forget to use the placeholder token in your prompt

    def generate_image(self, prompt, concept_index):
        # # Get the index of the selected concept
        # concept_index = self.repo_id_embeds.index(selected_concept)
        prompt_to_send =  prompt + " " + self.prompts_suffixes[concept_index]
        
        print(f"Generating image for concept: {self.repo_id_embeds[concept_index]} with prompt: {prompt_to_send} and generator: {self.generators[concept_index].manual_seed}")

        # Generate the image
        result = self.pipe(prompt_to_send, 
            num_images_per_prompt=1,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=self.generators[concept_index]
        ).images[0]

        #  result.save("output.png")

        # num_samples = 1 #@param {type:"number"}
        # num_rows = 1 #@param {type:"number"}


        # all_images = [] 
        # for _ in range(num_rows):
        #     images = pipe(prompt + textInversionPrompts[user_index], num_images_per_prompt=num_samples, num_inference_steps=50, guidance_scale=7.5, generator=generators[user_index]).images
        #     all_images.extend(images)

        # grid = image_grid(all_images, num_samples, num_rows)
        # grid.save("output.png")
        # print(grid)
        # all_images[0].save("output.png")
        
        return result

    def image_grid(self, imgs, rows, cols):
        assert len(imgs) == rows*cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        grid_w, grid_h = grid.size
        
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid
