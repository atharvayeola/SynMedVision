import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class StableDiffusionGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None):
        self.model_id = model_id
        if device:
            self.device = device
        else:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        
        logger.info(f"Loading Stable Diffusion model on {self.device}...")
        
        # Load the pipeline
        # We use float32 for CPU/MPS compatibility in some cases, but float16 is better for GPU memory if supported.
        # For simplicity and compatibility on Mac (MPS), float32 is often safer by default, or float16 if supported.
        # Let's stick to default precision for now to avoid compatibility issues, or use float16 if CUDA.
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            use_safetensors=True
        )
        self.pipe.to(self.device)
        
        # Enable attention slicing for lower memory usage
        self.pipe.enable_attention_slicing()
        
        # Workaround for some MPS issues with warmups
        if self.device == "mps":
            self.pipe.enable_attention_slicing() 

    def generate(self, prompt: str, steps: int = 30, guidance_scale: float = 7.5, seed: Optional[int] = None) -> Image.Image:
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)
            
        logger.info(f"Generating image for prompt: '{prompt}'")
        image = self.pipe(
            prompt, 
            num_inference_steps=steps, 
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        return image
