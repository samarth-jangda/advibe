# The following script is used to carry following functionalities for he background
# 1) Generating the background image of scene using the scene description from the story (Stable Diffusion)
# 2) Generating the depth map of the background image. (MiDAS from facebook)
# 3) Generating the mask of the area where avatar is expected to be present with new pose (MiDAS + region selection)
# 4) Refining the masked region using YOLO model.

import json
import torch
from PIL import Image
from diffusers import DiffusionPipeline

# The following script is used to carry following functionalities for he background
# 1) Generating the background image of scene using the scene description from the story (Stable Diffusion)
# 2) Generating the depth map of the background image. (MiDAS from facebook)
# 3) Generating the mask of the area where avatar is expected to be present with new pose (MiDAS + region selection)
# 4) Refining the masked region using YOLO model.

class BackgroundScene:
    """
    
    """  
    def __init__(self,story_json):
        self.story_data=story_json
        self.background_images=[] # list of path of all the generated background images for all scenes
        self.background_depth_maps=[] #list path of all depth maps images of background images 
    
    def generate_background_image(self):
        """
        The following function is used to generate the background
        images for every scene.
        NOTE: Need to save the generated images for all scenes for
        furthur processing.
        """
        # load the base stable diffusion model
        pipe=DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # load the refiner model
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            torch_dtype=torch.float16,
            use_safetonsors=True,
            low_cpu_mem_usage=True,
            varient="fp16"
        )
        refiner.to("cuda" if torch.cuda.is_available() else "cpu")
        
        seed=42
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        for scene_idx,scene_json in enumerate(self.story_data['Scenes']):
            # model input parameters
            prompt=scene_json['Scene']['background']
            guidance_scale=7.5
            inference_steps=50
            # model inference
            with torch.no_grad():
              image = pipe(
                  prompt=prompt,  # Fixed typo
                  num_inference_steps=inference_steps,
                  guidance_scale=guidance_scale,  # Fixed parameter name
                  output_type='latent',  # Keep 'latent' if refining, otherwise use 'pil'
                  generator=generator
              ).images  # First stage: Generate latent image

              image = refiner(
                  prompt=prompt,  # Fixed typo
                  num_inference_steps=inference_steps,
                  denoising_start=guidance_scale,  # Ensure this parameter exists
                  image=image,
                  generator=generator
              ).images[0]  # Second stage: Refine the image
            
            image.save(f"/home/background_{scene_idx}.png")    
                
    def background_depth_map():
        """
        
        """    
        
    def positional_depth_map():
        """
        
        """    
    
    def refine_depth_map():
        """
        
        """    
        
def background_scene(story_path):
    """
    LINK: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    The following function is used to carry out following steps for
    generating the background scenes.
    1) background_image: generating the background image as per scene description
    2) background_depth_maps: generating the depth maps for background images.
    3) positional_depth_map: generating depth map of position where avatar/character is required
    4) refine_depth_map: refining of depth map by using refining techniques for better quality
    """
    with open(story_path,'r') as story_json_file:
        open_story_json = json.load(story_json_file)
    background_class=BackgroundScene(open_story_json)
    scene_image=background_class.generate_background_image()

background_scene("/home/story_data.json")    
      
    