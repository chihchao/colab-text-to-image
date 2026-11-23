# @title SDXL-Turbo text-to-image
"""Runs inside a Colab kernel via `colab exec -f generate.py --env ...`.

Parameters arrive as environment variables (set with `colab exec --env`):
  T2I_PROMPT    (required) text prompt
  T2I_NEGATIVE  negative prompt (default: empty)
  T2I_N         number of images (default 1)
  T2I_WIDTH     width in px, multiple of 8 (default 512)
  T2I_HEIGHT    height in px, multiple of 8 (default 512)
  T2I_STEPS     inference steps, 1-4 for Turbo (default 1)
  T2I_SEED      integer seed; empty = random
  T2I_TAG       filename prefix (default: timestamp)
  T2I_SCORE     "1" to compute a CLIP prompt-match score per image (default 1)

The loaded pipeline is cached in kernel globals as `_T2I_PIPE`, so later calls
in the same session skip the model load (CLIP is cached as `_T2I_CLIP`).
Results are printed on one line starting with `T2I_OUTPUTS=` as JSON:
  {"model": ..., "images": [{"path", "seed", "pixel_std", "clip_score"}]}
pixel_std near 0 means a blank/black image. clip_score is 100 * cosine
similarity between prompt and image (CLIP ViT-B/32); measured: matching ~30-37, unrelated ~12.
"""
import json
import os
import random
import subprocess
import sys
import time

MODEL_ID = "stabilityai/sdxl-turbo"
CLIP_ID = "openai/clip-vit-base-patch32"
OUT_DIR = "/content/outputs"


def _env_int(name, default):
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _ensure_deps():
    try:
        import diffusers  # noqa: F401
        import accelerate  # noqa: F401
    except ImportError:
        print("[t2i] installing diffusers/transformers/accelerate ...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "diffusers", "transformers", "accelerate"],
            check=True,
        )


def _get_pipe():
    pipe = globals().get("_T2I_PIPE")
    if pipe is not None:
        return pipe
    _ensure_deps()
    import torch
    from diffusers import AutoPipelineForText2Image

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU in this session; create it with --gpu T4.")
    print(f"[t2i] loading {MODEL_ID} (first run downloads ~7 GB) ...", flush=True)
    t0 = time.time()
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[t2i] model ready in {time.time() - t0:.0f}s", flush=True)
    globals()["_T2I_PIPE"] = pipe
    return pipe


def _get_clip():
    cached = globals().get("_T2I_CLIP")
    if cached is not None:
        return cached
    import torch
    from transformers import CLIPModel, CLIPProcessor

    print(f"[t2i] loading {CLIP_ID} for prompt-match scoring ...", flush=True)
    model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    globals()["_T2I_CLIP"] = (model, proc)
    return model, proc


def _clip_score(image, prompt):
    import torch

    model, proc = _get_clip()
    inputs = proc(text=[prompt], images=image, return_tensors="pt",
                  padding=True, truncation=True, max_length=77).to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].half()
    with torch.no_grad():
        out = model(**inputs)  # image_embeds/text_embeds: projected, stable across versions
    img, txt = out.image_embeds.float(), out.text_embeds.float()
    img = img / img.norm(dim=-1, keepdim=True)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    return float((img * txt).sum()) * 100.0


def _main():
    import torch

    prompt = os.environ.get("T2I_PROMPT", "").strip()
    if not prompt:
        raise ValueError("T2I_PROMPT is empty")
    negative = os.environ.get("T2I_NEGATIVE", "").strip() or None
    n = max(1, min(_env_int("T2I_N", 1), 8))
    width = _env_int("T2I_WIDTH", 512) // 8 * 8
    height = _env_int("T2I_HEIGHT", 512) // 8 * 8
    steps = max(1, min(_env_int("T2I_STEPS", 1), 8))
    seed = _env_int("T2I_SEED", random.randint(0, 2**31 - 1))
    tag = os.environ.get("T2I_TAG", "").strip() or time.strftime("%Y%m%d-%H%M%S")
    tag = "".join(c for c in tag if c.isalnum() or c in "-_") or "img"

    score = os.environ.get("T2I_SCORE", "1").strip() != "0"
    pipe = _get_pipe()
    os.makedirs(OUT_DIR, exist_ok=True)
    images = []
    t0 = time.time()
    for i in range(n):
        gen = torch.Generator("cuda").manual_seed(seed + i)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=0.0,  # SDXL-Turbo is trained without CFG
            width=width,
            height=height,
            generator=gen,
        ).images[0]
        path = f"{OUT_DIR}/{tag}_{i + 1}.png"
        image.save(path)
        import numpy as np

        info = {"path": path, "seed": seed + i,
                "pixel_std": round(float(np.asarray(image, dtype=np.float32).std()), 2)}
        if score:
            try:
                info["clip_score"] = round(_clip_score(image, prompt), 2)
            except Exception as e:  # scoring is advisory; never fail generation
                print(f"[t2i] CLIP scoring skipped: {e}", flush=True)
        images.append(info)
        extra = f", CLIP {info['clip_score']}" if "clip_score" in info else ""
        print(f"[t2i] saved {path} (seed {seed + i}{extra})", flush=True)
    print(f"[t2i] {n} image(s) in {time.time() - t0:.1f}s", flush=True)
    print("T2I_OUTPUTS=" + json.dumps({"model": MODEL_ID, "images": images}), flush=True)


_main()
