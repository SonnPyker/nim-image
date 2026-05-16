from pathlib import Path

OPENAI_COMPAT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_IMAGE_API_BASE = "https://ai.api.nvidia.com/v1/genai"
DEFAULT_PROMPT = (
    "Hexagonal map tile, Chinese ink wash painting, high mountain peak, cultivation game asset"
)
DEFAULT_SIZE = "1024x1024"
OUTPUT_DIR = Path("assets/images")
KNOWN_IMAGE_MODELS = [
    "qwen/qwen-image",
    "black-forest-labs/flux.1-dev",
    "black-forest-labs/flux.1-schnell",
    "black-forest-labs/flux.2-klein-4b",
    "stabilityai/stable-diffusion-3-medium",
]
IMAGE_MODEL_HINTS = (
    "image",
    "flux",
    "stable-diffusion",
    "sdxl",
    "playground",
    "recraft",
    "vision-image",
)
IMAGE_SIZES = ["512x512", "768x768", "1024x1024", "1024x1792", "1792x1024"]
