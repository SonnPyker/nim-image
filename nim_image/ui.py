from nim_image.api import fetch_image_models, generate_nvidia_image_with_fallback, select_default_model
from nim_image.config import DEFAULT_PROMPT, DEFAULT_SIZE, IMAGE_SIZES


def generate_image(prompt: str, size: str, model: str):
    import gradio as gr

    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")

    try:
        model_choices = fetch_image_models()
        saved_path = generate_nvidia_image_with_fallback(prompt, size, model, model_choices)
    except Exception as exc:
        raise gr.Error(f"Image generation failed: {exc}") from exc

    return str(saved_path)


def build_interface():
    import gradio as gr

    model_choices = fetch_image_models()
    default_model = select_default_model(model_choices)

    with gr.Blocks(title="NVIDIA Image Generation WebUI") as demo:
        gr.Markdown("# NVIDIA Image Generation WebUI")

        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder=DEFAULT_PROMPT,
                lines=4,
                value=DEFAULT_PROMPT,
            )

        with gr.Row():
            model = gr.Dropdown(
                choices=model_choices,
                value=default_model,
                label="Model",
            )
            size = gr.Dropdown(
                choices=IMAGE_SIZES,
                value=DEFAULT_SIZE,
                label="Image Size",
            )

        generate_button = gr.Button("Generate", variant="primary")
        output_image = gr.Image(label="Generated Image", type="filepath")

        generate_button.click(
            fn=generate_image,
            inputs=[prompt, size, model],
            outputs=output_image,
        )

    return demo
