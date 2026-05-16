import base64
import imghdr
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

from nim_image.api import (
    build_nvidia_image_request,
    extract_image_data,
    filter_image_models,
    format_nvidia_http_error,
)
from nim_image.files import save_generated_image


class AppHelperTests(unittest.TestCase):
    def test_build_nvidia_image_request_maps_model_and_size(self):
        endpoint, headers, payload = build_nvidia_image_request(
            api_key="secret",
            model="black-forest-labs/flux.1-dev",
            prompt="mountain temple",
            size="1024x1792",
        )

        self.assertEqual(
            endpoint,
            "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev",
        )
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual(headers["Accept"], "application/json")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(payload["prompt"], "mountain temple")
        self.assertEqual(payload["width"], 1024)
        self.assertEqual(payload["height"], 1792)
        self.assertEqual(payload["cfg_scale"], 5)
        self.assertEqual(payload["steps"], 50)

    def test_extract_image_data_reads_artifacts_base64(self):
        b64_png, image_url = extract_image_data(
            {
                "artifacts": [
                    {
                        "base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
                        "seed": 123,
                    }
                ]
            }
        )

        self.assertEqual(b64_png, "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ")
        self.assertIsNone(image_url)

    def test_format_nvidia_http_error_uses_json_body_message(self):
        response = Mock()
        response.status_code = 500
        response.url = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev"
        response.text = '{"detail":"backend overloaded"}'
        response.json.return_value = {"detail": "backend overloaded"}

        err = requests.HTTPError("500 Server Error", response=response)
        message = format_nvidia_http_error(err)

        self.assertIn("status=500", message)
        self.assertIn("backend overloaded", message)

    def test_filter_image_models_prefers_image_generation_ids(self):
        models = [
            "qwen/qwen2.5-coder-32b-instruct",
            "black-forest-labs/flux.1-dev",
            "qwen/qwen-image",
            "nvidia/llama-3.1-nemotron",
        ]

        self.assertEqual(
            filter_image_models(models),
            ["black-forest-labs/flux.1-dev", "qwen/qwen-image"],
        )

    def test_save_generated_image_writes_base64_png(self):
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/lNnXYQAAAABJRU5ErkJggg=="
        )
        b64_png = base64.b64encode(png_bytes).decode("ascii")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            saved_path = save_generated_image(b64_png=b64_png, output_dir=output_dir)

            self.assertEqual(saved_path.parent, output_dir)
            self.assertEqual(saved_path.suffix, ".png")
            self.assertTrue(saved_path.exists())
            self.assertEqual(imghdr.what(saved_path), "png")


if __name__ == "__main__":
    unittest.main()
