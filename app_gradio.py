
# app_gradio.py
# Simple web UI using Gradio that calls the Hugging Face Inference API.
# Great for demos on local machine or Colab (with public link via share=True).

import os
import requests
from io import BytesIO
from PIL import Image
import gradio as gr
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"

def infer(prompt, negative_prompt, steps, guidance, width, height, token):
    token = token or HF_TOKEN
    if not token:
        return None, "Missing token. Provide HF token in the textbox or set HF_TOKEN env var."
    headers = {"Authorization": f"Bearer {token}", "Accept": "image/png"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt or None,
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance),
            "width": int(width),
            "height": int(height)
        }
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        try:
            return None, f"Error {resp.status_code}: {resp.text[:400]}"
        except Exception:
            return None, f"Error {resp.status_code}"
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img, "Success"

with gr.Blocks(title="Stable Diffusion (HF Inference API)") as demo:
    gr.Markdown("# Stable Diffusion (Hugging Face API)\nEnter a prompt and generate an image in seconds.")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value="A futuristic city at sunset, ultra realistic, cinematic lighting")
        negative = gr.Textbox(label="Negative Prompt", value="blurry, low quality, distorted")
    with gr.Row():
        steps = gr.Slider(5, 50, value=30, step=1, label="Steps")
        guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
    with gr.Row():
        width = gr.Dropdown([384, 448, 512, 576, 640, 704, 768], value=512, label="Width")
        height = gr.Dropdown([384, 448, 512, 576, 640, 704, 768], value=512, label="Height")
    token_box = gr.Textbox(label="HF Token (optional if set in env)", type="password")
    btn = gr.Button("Generate")
    out_img = gr.Image(label="Output")
    status = gr.Markdown()

    btn.click(infer, inputs=[prompt, negative, steps, guidance, width, height, token_box], outputs=[out_img, status])

if __name__ == "__main__":
    # For local use: python app_gradio.py
    # For Colab: set share=True to get a public URL
    demo.launch()
