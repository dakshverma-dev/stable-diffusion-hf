
# app.py
# Lightweight Stable Diffusion client using Hugging Face Inference API
# Works locally and in Google Colab. No big model downloads. No GPU required on your machine.
# Author: Daksh Verma (project scaffold prepared by ChatGPT)

import os
import time
import argparse
import requests
from io import BytesIO
from PIL import Image
import datetime
from dotenv import load_dotenv

# Load token from .env if present
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"

def _headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "image/png"
    }

def generate_image(
    prompt: str,
    negative_prompt: str = None,
    steps: int = 30,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    output_dir: str = "generated_images",
    hf_token: str = None,
    retries: int = 6,
    timeout: int = 120
) -> str:
    """Generate an image via Hugging Face Inference API and save to disk.

    Returns: path of saved image.
    """
    token = hf_token or HF_TOKEN
    if not token:
        raise ValueError("Hugging Face token not found. Set HF_TOKEN env var or pass --token.")

    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height
        }
    }

    # Retry if model is loading (503) or transient errors occur
    backoff = 2
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(API_URL, headers=_headers(token), json=payload, timeout=timeout)
            if resp.status_code == 200:
                image = Image.open(BytesIO(resp.content)).convert("RGB")
                ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                safe_snippet = "".join(c for c in prompt[:40] if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
                filename = f"{ts}_{safe_snippet or 'image'}.png"
                out_path = os.path.join(output_dir, filename)
                image.save(out_path)
                print(f"Saved → {out_path}")
                return out_path
            elif resp.status_code == 503:
                # Model is loading
                wait = backoff
                print(f"[Attempt {attempt}/{retries}] Model loading (503). Retrying in {wait}s...")
                time.sleep(wait)
                backoff = min(backoff * 2, 30)
            else:
                # Some models return JSON error bodies; surface them for easier debugging
                try:
                    print("Response body:", resp.text[:500])
                except Exception:
                    pass
                resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            wait = backoff
            print(f"[Attempt {attempt}/{retries}] Network error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            backoff = min(backoff * 2, 30)

    raise RuntimeError("Failed to generate image after multiple retries.")

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion via Hugging Face Inference API")
    parser.add_argument("--prompt", type=str, required=False, help="Text prompt for the image")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Things to avoid in the image")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion steps (higher = slower, more detail)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Prompt adherence (6–12 typical)")
    parser.add_argument("--width", type=int, default=512, help="Image width (multiples of 64)")
    parser.add_argument("--height", type=int, default=512, help="Image height (multiples of 64)")
    parser.add_argument("--outdir", type=str, default="generated_images", help="Output directory")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token override (else use HF_TOKEN env var)")

    args = parser.parse_args()

    prompt = args.prompt or input("Enter your prompt: ").strip()
    path = generate_image(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        output_dir=args.outdir,
        hf_token=args.token
    )
    print(f"Done. Image saved to: {path}")

if __name__ == "__main__":
    main()
