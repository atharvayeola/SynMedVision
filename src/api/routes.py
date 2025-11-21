from fastapi import APIRouter, HTTPException
from src.api.schemas import GenerationRequest, GenerationResponse
from src.models.diffusion.pipeline import StableDiffusionGenerator
import base64
from io import BytesIO
import logging
import time

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model instance to avoid reloading on every request
# In a real app, this might be managed by a dependency injection system or lifespan event
generator = None

def get_generator():
    global generator
    if generator is None:
        generator = StableDiffusionGenerator()
    return generator

@router.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    try:
        gen = get_generator()
        
        start_time = time.perf_counter()
        image = gen.generate(
            prompt=request.prompt,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return GenerationResponse(status="success", image_base64=img_str, inference_time=duration)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return GenerationResponse(status="error", error=str(e))
