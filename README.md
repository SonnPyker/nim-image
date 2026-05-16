# NimImage

Python Gradio WebUI for NVIDIA image generation models.

## Features

- Gradio interface for prompt-based image generation
- Auto-discovers image-like models from NVIDIA OpenAI-compatible model list
- Prefers `qwen/qwen-image`, then falls back to other supported image models
- Calls NVIDIA image generation endpoints directly and retries other models if one fails
- Saves generated images locally to `assets/images/`

## Requirements

- Python 3.12+
- NVIDIA API key

## Installation

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

Set your API key in `.env`:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

## Run

```bash
python app.py
```

Gradio prints a local URL. Open it in your browser.

## Test

```bash
python -m unittest tests.test_app_helpers -v
python -m py_compile app.py
```

## Project Structure

```text
.
├── app.py
├── nim_image/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── files.py
│   └── ui.py
├── assets/
│   └── images/
├── tests/
│   └── test_app_helpers.py
├── requirements.txt
└── .env.example
```

## Notes

- Generated images are saved locally and ignored by git.
- `.env` is ignored by git.
- If one model returns a server error, app automatically tries other supported models.
