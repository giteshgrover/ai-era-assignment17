#@title Import required libraries
import os
import torch
import re
from tqdm import tqdm
import PIL
from PIL import Image

from typing import List, Optional, Tuple, Union

from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
# from transformers.modeling_attn_mask_utils import AttentionMaskConverter

class TextualInversion:
    def __init__(self, pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4", repo_id_embeds=["sd-concepts-library/matrix::with <hatman-matrix> concept"]):
        #@markdown `pretrained_model_name_or_path` which Stable Diffusion checkpoint you want to use. This should match the one used for training the embeddings.
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        #@title Load your concept here
    #@markdown Enter the `repo_id` for a concept you like (you can find pre-learned concepts in the public [SD Concepts Library](https://huggingface.co/sd-concepts-library))
        self.repo_id_embeds = [x.split("::")[0].split("/")[-1] for x in repo_id_embeds]
        self.prompts_suffixes = [x.split("::")[1] for x in repo_id_embeds]
       
       # Set device
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if "mps" == self.device: os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

        #@title Load the Stable Diffusion pipeline
        # self.pipe = StableDiffusionPipeline.from_pretrained(
        #     pretrained_model_name_or_path,
        #     torch_dtype=torch.float16
        # ).to(self.device)

        # Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        # The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        # The noise scheduler
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        # To the GPU we go!
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)
        
        # Access the token embedding layers
        # Token Embedding Layer
        self.token_emb_layer = self.text_encoder.text_model.embeddings.token_embedding 
        # Position Embedding Layer
        self.position_ids = self.text_encoder.text_model.embeddings.position_ids
        self.position_emb_layer = self.text_encoder.text_model.embeddings.position_embedding
        
        self.conceptsEmbeddings = []
        for index,repo_id in enumerate(self.repo_id_embeds):
            #@title Load the concept into pipeline
            concept_embed_lib = torch.load("sd-concepts-library/" + self.repo_id_embeds[index] +"_learned_embeds.bin") # load the concept learned embeddings
            print(self.repo_id_embeds[index])
            print(concept_embed_lib.keys())
            if self.repo_id_embeds[index] in concept_embed_lib.keys():
                concept_embed = concept_embed_lib[self.repo_id_embeds[index]] # Read the embedding value using the key i.e. concept_embed_lib['<birb-style>']
            else:
                first_key, concept_embed = next(iter(concept_embed_lib.items())) # Read the first key and the embedding value
            
            self.conceptsEmbeddings.append(concept_embed.to(self.device))
        print(f"len(self.conceptsEmbeddings): {len(self.conceptsEmbeddings)}")

    def _create_4d_causal_attention_mask(
        input_shape: Union[torch.Size, Tuple, List],
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)`

        Args:
            input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
                The input shape should be a tuple that defines `(batch_size, query_length)`.
            dtype (`torch.dtype`):
                The torch dtype the created mask shall have.
            device (`int`):
                The torch device the created mask shall have.
            sliding_window (`int`, *optional*):
                If the model uses windowed attention, a sliding window should be passed.
        """
        attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

        key_value_length = past_key_values_length + input_shape[-1]
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=dtype, device=device
        )

        return attention_mask 

    def get_output_embeds(self, input_embeddings):
        # CLIP's text model uses causal mask, so we prepare it here:
        bsz, seq_len = input_embeddings.shape[:2]
        # causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, dtype=input_embeddings.dtype)
        # causal_attention_mask = self._create_4d_causal_attention_mask(input_shape=(bsz, seq_len), dtype=input_embeddings.dtype, device=self.device)
        causal_attention_mask = self.text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, dtype=input_embeddings.dtype)

        # Getting the output embeddings involves calling the model with passing output_hidden_states=True
        # so that it doesn't just return the pooled final predictions:
        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=None, # We aren't using an attention mask so that can be None
            causal_attention_mask=causal_attention_mask.to(self.device),
            output_attentions=None,
            output_hidden_states=True, # We want the output embs not the final output
            return_dict=None,
        )

        # We're interested in the output hidden state only
        output = encoder_outputs[0]

        # There is a final layer norm we need to pass these through
        output = self.text_encoder.text_model.final_layer_norm(output)

        # And now they're ready!
        return output
    
    def set_timesteps(self, num_inference_steps):
        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.timesteps = self.scheduler.timesteps.to(torch.float32) # minor fix to ensure MPS compatibility, fixed in diffusers PR 3925
        
    def pil_to_latent(self, input_im):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        with torch.no_grad():
            latent = self.vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(self.device)*2-1) # Note scaling
        return 0.18215 * latent.latent_dist.sample()

    def latents_to_pil(self, latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images
    
    def generate_with_embs(self, text_embeddings, generator, max_length,batch_size = 1):
        height = 512                        # default height of Stable Diffusion
        width = 512                         # default width of Stable Diffusion
        num_inference_steps = 50            # Number of denoising steps
        guidance_scale = 7.5                # Scale for classifier-free guidance
        
        uncond_input = self.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prep Scheduler
        self.set_timesteps(num_inference_steps)

        # Prep latents
        latents = torch.randn(
        (batch_size, self.unet.in_channels, height // 8, width // 8),
        generator=generator,
        # device=self.device
        )
        latents = latents.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma

        # Loop
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = self.scheduler.sigmas[i]
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return self.latents_to_pil(latents)

        
    def generate_image(self, prompt, concept_index, background_blur=False):
        # # Get the index of the selected concept
        # concept_index = self.repo_id_embeds.index(selected_concept)
        prompt_to_send =  prompt + " " + self.prompts_suffixes[concept_index]
        print(f"Selected concept_index: {concept_index}.")
        print(f"concept_index: {concept_index} Generating image for concept: {self.repo_id_embeds[concept_index]} with prompt: {prompt_to_send}")
        print(f"Background blur: {background_blur}")
        
        # replace <..> with a placeholder token that can be easily replaced with the embediing after tokenization
        placeholder_text = "gloucestershire " # 33789 is the token id 
        prompt_to_send = re.sub(r'<.*?>', placeholder_text, prompt_to_send)
        print(f"prompt after replacing placeholder token: {prompt_to_send}")
        # Tokenize
        text_input = self.tokenizer(prompt_to_send, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        input_ids = text_input.input_ids.to(self.device)

        # Get token embeddings
        token_embeddings = self.token_emb_layer(input_ids)
        
        # The new embedding - our concept embedding for the special word token
        # replacement_token_embedding = birb_embed['<birb-style>'].to(torch_device)
        replacement_token_embedding = self.conceptsEmbeddings[concept_index].to(self.device)
        print(f"replacement_token_embedding.shape: {replacement_token_embedding.shape} and token_embeddings.shape: {token_embeddings.shape}")
        print(f"torch.where(input_ids[0]==33789): {torch.where(input_ids[0]==33789)}")
        # Replace the placholder token with the concept embedding
        token_embeddings[0, torch.where(input_ids[0]==33789)] = replacement_token_embedding.to(self.device)
        # print(f"If embedding is replaced: {token_embeddings[0, torch.where(input_ids[0]==33789)] == replacement_token_embedding}")
        
        B, T, C = token_embeddings.shape
        # Get the position embeddings
        position_embeddings = self.position_emb_layer(self.position_ids[:, :T])

        # Combine with pos embs
        input_embeddings = token_embeddings + position_embeddings

        #  Feed through to get final output embs
        modified_output_embeddings = self.get_output_embeds(input_embeddings)

        print(f"manual_seed: {concept_index + 11}")
        generator = torch.manual_seed(concept_index + 11)
        # And generate an image with this:
        result = self.generate_with_embs(modified_output_embeddings, generator=generator, max_length=T)[0]
        
        return result
