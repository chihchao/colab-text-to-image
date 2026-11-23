# @title Text-to-image (SDXL-Turbo / Z-Image-Turbo)
"""Runs inside a Colab kernel via `colab exec -f generate.py --env ...`.

Parameters arrive as environment variables (set with `colab exec --env`):
  T2I_MODEL     "sdxl-turbo" (default, T4) or "z-image-turbo" (L4, better quality,
                can render Chinese/English text)
  T2I_PROMPT    (required) text prompt
  T2I_NEGATIVE  negative prompt (default: empty)
  T2I_N         number of images (default 1)
  T2I_WIDTH     width in px, multiple of 8 (default 512)
  T2I_HEIGHT    height in px, multiple of 8 (default 512)
  T2I_STEPS     inference steps (default: model's own, 1 for SDXL-Turbo, 9 for Z-Image)
  T2I_SEED      integer seed; empty = random
  T2I_TAG       filename prefix (default: timestamp)
  T2I_SCORE     "1" to compute a CLIP prompt-match score per image (default 1)
  T2I_REMOVE_BG "1" to also save a transparent-background copy (BiRefNet)

The loaded pipelines are cached in kernel globals (`_T2I_PIPES`), so later calls
in the same session skip the model load (CLIP is cached as `_T2I_CLIP`,
BiRefNet as `_T2I_BG`). Results are printed on one line starting with
`T2I_OUTPUTS=` as JSON:
  {"model": ..., "images": [{"path", "seed", "pixel_std", "clip_score",
                             "nobg_path", "transparent_ratio", "edge_opaque_ratio"}]}
transparent_ratio = share of fully/mostly transparent pixels in the cut-out;
edge_opaque_ratio = share of border pixels that stayed opaque (subject or
leftover background touching the frame).
pixel_std near 0 means a blank/black image. clip_score is 100 * cosine
similarity between prompt and image (CLIP ViT-B/32); measured: matching ~30-37, unrelated ~12.
"""
import json
import os
import random
import subprocess
import sys
import time

MODELS = {
    # SDXL-Turbo: ~7 GB fp16, fits a T4 entirely. No CFG, 1-4 steps.
    "sdxl-turbo": {"id": "stabilityai/sdxl-turbo", "revision": None, "steps": 1},
    # Z-Image-Turbo (Tongyi-MAI, Apache-2.0): 6B DiT + Qwen3-4B text encoder,
    # bf16. Everything on the GPU peaks at ~23 GB on an L4 (24 GB) and runs out
    # of memory once CLIP/BiRefNet are loaded too, so the text encoder (~8 GB)
    # lives on the CPU and visits the GPU only to encode the prompt.
    "z-image-turbo": {"id": "Tongyi-MAI/Z-Image-Turbo",
                      "revision": "f332072aa78be7aecdf3ee76d5c247082da564a6",
                      "steps": 9},
}
CLIP_ID = "openai/clip-vit-base-patch32"
BG_ID = "ZhengPeng7/BiRefNet"  # MIT license
# Pinned: BiRefNet loads its own Python code from the Hub (trust_remote_code),
# so never follow the moving main branch.
BG_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
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


def _get_pipe(model):
    pipes = globals().setdefault("_T2I_PIPES", {})
    if model in pipes:
        return pipes[model]
    _ensure_deps()
    import torch
    import diffusers

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU in this session.")
    spec = MODELS[model]
    t0 = time.time()
    if model == "sdxl-turbo":
        print(f"[t2i] loading {spec['id']} (first run downloads ~7 GB) ...", flush=True)
        pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
            spec["id"], torch_dtype=torch.float16, variant="fp16").to("cuda")
    else:
        if not hasattr(diffusers, "ZImagePipeline"):
            print("[t2i] upgrading diffusers for Z-Image ...", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers"],
                           check=True)
            raise RuntimeError("diffusers was upgraded; run `colab restart-kernel` and retry.")
        free_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if free_gb < 20:
            raise RuntimeError(f"Z-Image-Turbo needs a 24 GB GPU (L4 or better); "
                               f"this one has {free_gb:.0f} GB.")
        print(f"[t2i] loading {spec['id']} (first run downloads ~20 GB, 2-3 min) ...",
              flush=True)
        pipe = diffusers.ZImagePipeline.from_pretrained(
            spec["id"], torch_dtype=torch.bfloat16, revision=spec["revision"])
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        pipe.text_encoder.to("cpu")
    pipe.set_progress_bar_config(disable=True)
    print(f"[t2i] model ready in {time.time() - t0:.0f}s", flush=True)
    pipes[model] = pipe
    return pipe


def _generate_one(model, pipe, prompt, negative, steps, width, height, seed):
    import torch

    gen = torch.Generator("cuda").manual_seed(seed)
    if model == "sdxl-turbo":
        return pipe(prompt=prompt, negative_prompt=negative, num_inference_steps=steps,
                    guidance_scale=0.0,  # SDXL-Turbo is trained without CFG
                    width=width, height=height, generator=gen).images[0]
    # Z-Image-Turbo: encode on the GPU, then send the text encoder back to the CPU.
    te = pipe.text_encoder
    te.to("cuda")
    try:
        with torch.no_grad():
            embeds, _ = pipe.encode_prompt(prompt=prompt, device="cuda",
                                           do_classifier_free_guidance=False)
    finally:
        te.to("cpu")
        torch.cuda.empty_cache()
    return pipe(prompt_embeds=embeds, num_inference_steps=steps,
                guidance_scale=0.0,  # Turbo distillation: no CFG, negative prompt unused
                width=width, height=height, generator=gen).images[0]


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


def _get_bg():
    cached = globals().get("_T2I_BG")
    if cached is not None:
        return cached
    try:
        import kornia  # noqa: F401  (required by BiRefNet's remote code)
    except ImportError:
        print("[t2i] installing kornia ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kornia"], check=True)
    import torch
    from transformers import AutoModelForImageSegmentation

    print(f"[t2i] loading {BG_ID} for background removal ...", flush=True)
    model = AutoModelForImageSegmentation.from_pretrained(
        BG_ID, trust_remote_code=True, revision=BG_REVISION
    ).to("cuda").eval().half()
    globals()["_T2I_BG"] = model
    return model


def _remove_bg(image):
    """Return (RGBA image, stats) with the background made transparent."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    model = _get_bg()
    tf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(image.convert("RGB")).unsqueeze(0).to("cuda").half()
    with torch.no_grad():
        pred = model(x)[-1].sigmoid().float().cpu()[0, 0]
    mask = transforms.functional.to_pil_image(pred).resize(image.size, Image.BILINEAR)
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)

    a = np.asarray(mask, dtype=np.uint8)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    stats = {
        "transparent_ratio": round(float((a < 128).mean()), 3),
        "edge_opaque_ratio": round(float((border >= 128).mean()), 3),
    }
    return rgba, stats


def _main():
    import torch

    model = os.environ.get("T2I_MODEL", "sdxl-turbo").strip() or "sdxl-turbo"
    if model not in MODELS:
        raise ValueError(f"unknown T2I_MODEL {model!r}; choose from {list(MODELS)}")
    prompt = os.environ.get("T2I_PROMPT", "").strip()
    if not prompt:
        raise ValueError("T2I_PROMPT is empty")
    negative = os.environ.get("T2I_NEGATIVE", "").strip() or None
    n = max(1, min(_env_int("T2I_N", 1), 8))
    width = _env_int("T2I_WIDTH", 512) // 8 * 8
    height = _env_int("T2I_HEIGHT", 512) // 8 * 8
    steps = max(1, min(_env_int("T2I_STEPS", MODELS[model]["steps"]), 16))
    seed = _env_int("T2I_SEED", random.randint(0, 2**31 - 1))
    tag = os.environ.get("T2I_TAG", "").strip() or time.strftime("%Y%m%d-%H%M%S")
    tag = "".join(c for c in tag if c.isalnum() or c in "-_") or "img"

    score = os.environ.get("T2I_SCORE", "1").strip() != "0"
    remove_bg = os.environ.get("T2I_REMOVE_BG", "0").strip() == "1"
    pipe = _get_pipe(model)
    os.makedirs(OUT_DIR, exist_ok=True)
    images = []
    t0 = time.time()
    for i in range(n):
        image = _generate_one(model, pipe, prompt, negative, steps, width, height, seed + i)
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
        if remove_bg:
            rgba, bg_stats = _remove_bg(image)
            nobg_path = f"{OUT_DIR}/{tag}_{i + 1}_nobg.png"
            rgba.save(nobg_path)
            info.update(nobg_path=nobg_path, **bg_stats)
        images.append(info)
        extra = f", CLIP {info['clip_score']}" if "clip_score" in info else ""
        if remove_bg:
            extra += f", transparent {info['transparent_ratio']:.0%}"
        print(f"[t2i] saved {path} (seed {seed + i}{extra})", flush=True)
    print(f"[t2i] {n} image(s) in {time.time() - t0:.1f}s", flush=True)
    print("T2I_OUTPUTS=" + json.dumps({"model": MODELS[model]["id"], "images": images}),
          flush=True)


_main()
