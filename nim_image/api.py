import os
from typing import Iterable

import requests
from dotenv import load_dotenv

from nim_image.config import IMAGE_MODEL_HINTS, KNOWN_IMAGE_MODELS, NVIDIA_IMAGE_API_BASE, OPENAI_COMPAT_BASE_URL
from nim_image.files import save_generated_image


def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY missing. Add it to your .env file.")
    return api_key


def get_client():
    from openai import OpenAI

    return OpenAI(base_url=OPENAI_COMPAT_BASE_URL, api_key=get_api_key())


def parse_size(size: str) -> tuple[int, int]:
    width_text, height_text = size.split("x", maxsplit=1)
    return int(width_text), int(height_text)


def build_nvidia_image_request(
    *, api_key: str, model: str, prompt: str, size: str
) -> tuple[str, dict[str, str], dict[str, int | str]]:
    width, height = parse_size(size)
    endpoint = f"{NVIDIA_IMAGE_API_BASE}/{model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "cfg_scale": 5,
        "steps": 50,
        "samples": 1,
    }
    return endpoint, headers, payload


def filter_image_models(models: Iterable[str]) -> list[str]:
    filtered = []
    for model in models:
        lowered = model.lower()
        if any(hint in lowered for hint in IMAGE_MODEL_HINTS):
            filtered.append(model)
    return sorted(set(filtered))


def fetch_image_models() -> list[str]:
    ordered: list[str] = []

    try:
        client = get_client()
        response = client.models.list()
        model_ids = [model.id for model in response.data if getattr(model, "id", None)]
        for model in filter_image_models(model_ids):
            if model not in ordered:
                ordered.append(model)
    except Exception:
        pass

    for model in KNOWN_IMAGE_MODELS:
        if model not in ordered:
            ordered.append(model)

    return ordered


def select_default_model(model_choices: list[str]) -> str:
    for preferred in ("qwen/qwen-image", "black-forest-labs/flux.1-dev"):
        if preferred in model_choices:
            return preferred
    return model_choices[0]


def ordered_model_candidates(selected_model: str, available_models: list[str]) -> list[str]:
    ordered: list[str] = []

    if selected_model:
        ordered.append(selected_model)

    for model in available_models:
        if model not in ordered:
            ordered.append(model)

    for model in KNOWN_IMAGE_MODELS:
        if model not in ordered:
            ordered.append(model)

    return ordered


def _clean_base64(raw_value: str | None) -> str | None:
    if not isinstance(raw_value, str):
        return None
    if raw_value.startswith("data:image") and "," in raw_value:
        return raw_value.split(",", maxsplit=1)[1]
    return raw_value


def extract_image_data(response_json: dict) -> tuple[str | None, str | None]:
    if isinstance(response_json.get("image"), str):
        return _clean_base64(response_json["image"]), None
    if isinstance(response_json.get("base64"), str):
        return _clean_base64(response_json["base64"]), None
    if isinstance(response_json.get("b64_json"), str):
        return _clean_base64(response_json["b64_json"]), None
    if isinstance(response_json.get("url"), str):
        return None, response_json["url"]

    for collection_key in ("artifacts", "images", "output", "data"):
        collection = response_json.get(collection_key)
        if isinstance(collection, list) and collection:
            first_item = collection[0]
            if isinstance(first_item, dict):
                for b64_key in ("base64", "b64_json", "image"):
                    if isinstance(first_item.get(b64_key), str):
                        return _clean_base64(first_item[b64_key]), None
                if isinstance(first_item.get("url"), str):
                    return None, first_item["url"]

    return None, None


def format_nvidia_http_error(error: requests.HTTPError) -> str:
    response = error.response
    if response is None:
        return str(error)

    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            for key in ("detail", "error", "message", "title"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    detail = value.strip()
                    break
            if not detail:
                detail = str(payload)
        else:
            detail = str(payload)
    except Exception:
        detail = (response.text or "").strip()

    if len(detail) > 300:
        detail = detail[:300] + "..."

    status_code = getattr(response, "status_code", "?")
    url = getattr(response, "url", "")
    return f"status={status_code}, url={url}, detail={detail}"


def _request_single_model(prompt: str, size: str, model: str, api_key: str):
    endpoint, headers, payload = build_nvidia_image_request(
        api_key=api_key,
        model=model,
        prompt=prompt,
        size=size,
    )
    response = requests.post(endpoint, headers=headers, json=payload, timeout=180)
    response.raise_for_status()

    response_json = response.json()
    b64_png, image_url = extract_image_data(response_json)

    if not b64_png and not image_url:
        available_keys = ", ".join(sorted(response_json.keys()))
        raise ValueError(
            f"No image data in response. status={response.status_code}, model={model}, keys=[{available_keys}]"
        )

    return save_generated_image(b64_png=b64_png, image_url=image_url)


def generate_nvidia_image_with_fallback(
    prompt: str,
    size: str,
    selected_model: str,
    available_models: list[str],
):
    api_key = get_api_key()
    errors: list[str] = []

    for model in ordered_model_candidates(selected_model, available_models):
        try:
            return _request_single_model(prompt, size, model, api_key)
        except requests.HTTPError as exc:
            errors.append(f"{model} -> {format_nvidia_http_error(exc)}")
        except Exception as exc:
            errors.append(f"{model} -> {exc}")

    joined = " | ".join(errors)
    raise ValueError(f"All model attempts failed: {joined}")
