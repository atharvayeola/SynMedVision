from pydantic import BaseModel
from typing import Optional

class GenerationRequest(BaseModel):
    prompt: str
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None

class GenerationResponse(BaseModel):
    status: str
    image_base64: Optional[str] = None
    inference_time: Optional[float] = None
    error: Optional[str] = None
