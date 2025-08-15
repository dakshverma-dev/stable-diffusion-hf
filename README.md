
# Stable Diffusion (Hugging Face Inference API)

Fast, lightweight Stable Diffusion client using **Hugging Face's free Inference API**.
- No huge model downloads
- Runs on any machine (the heavy lifting happens on HF GPUs)
- Works locally **and** in **Google Colab**
- GitHub-ready, token kept out of the repo

---

## 1) Get Your Token (once)
1. Create a free account at Hugging Face → Settings → Access Tokens → **New token**
2. Role: **Read**; Name: e.g., `stable-diffusion-project`; Expiration: your choice
3. Copy the token (looks like `hf_xxx...`). Keep it private.

## 2) Run Locally (fastest setup)
```bash
git clone YOUR_REPO_URL
cd stable-diffusion-hf
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env  # then open .env and paste your token
python app.py --prompt "A cinematic neon-lit street, rainy night"
```

Output images are saved in `generated_images/`.

## 3) Run in Google Colab (no installs on your PC)
- Create a new Colab notebook and paste this in the first cell:
```python
import os, requests, datetime
from PIL import Image
from io import BytesIO

HF_TOKEN = input("Paste your HF token: ").strip()
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
headers = {"Authorization": f"Bearer {HF_TOKEN}", "Accept": "image/png"}

def generate(prompt, negative=None, steps=30, guidance=7.5, width=512, height=512):
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height
        }
    }
    r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"/content/{ts}.png"
    img.save(path)
    display(img); print("Saved:", path)

# Example
generate("A futuristic city at sunset, ultra realistic, cinematic lighting", negative="blurry, low quality")
```

## 4) Optional: Simple Web UI
```bash
# Make sure requirements are installed and .env has HF_TOKEN
python app_gradio.py
```
A local web page will open. In Colab, set `demo.launch(share=True)` inside `app_gradio.py` to get a public link.

## 5) Safe-by-Default Repo
- `.gitignore` excludes `.env` (your token) and generated images
- Use `.env.example` as a template for collaborators

---

## Notes
- Typical generation time: ~5–15 seconds (depends on HF queue)
- If you see **503**, the model is starting up; the script retries automatically (in `app.py`).
- Increase `--steps` to improve quality; decrease for speed.
