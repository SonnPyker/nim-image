import base64
import mimetypes
import uuid
from datetime import datetime
from pathlib import Path

import requests

from nim_image.config import OUTPUT_DIR


def guess_extension_from_bytes(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def save_generated_image(
    *,
    b64_png: str | None = None,
    image_url: str | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if b64_png:
        image_bytes = base64.b64decode(b64_png)
        suffix = guess_extension_from_bytes(image_bytes)
    elif image_url:
        response = requests.get(image_url, timeout=120)
        response.raise_for_status()
        image_bytes = response.content
        suffix = mimetypes.guess_extension(response.headers.get("content-type", ""))
        if not suffix:
            suffix = guess_extension_from_bytes(image_bytes)
    else:
        raise ValueError("No image data returned by API.")

    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex}{suffix}"
    saved_path = output_dir / filename
    saved_path.write_bytes(image_bytes)
    return saved_path
