---
name: colab-text-to-image
description: Generate images from a text prompt on a Google Colab GPU (SDXL-Turbo on T4 for fast drafts, Z-Image-Turbo on L4 for high quality and Chinese/English text in images) through the official `colab` CLI, save the PNGs locally, optionally remove the background (transparent PNG), and verify them. Use whenever an image/picture/illustration/icon/concept art must be created from text (文字生圖、生成圖片、畫一張圖) — whether the user asks directly or another task (a web page, slide, document, game asset) needs a generated image — or to make variations of a prompt, or when a transparent/cut-out image is needed (去背、透明背景、sprite, sticker, icon). Handles CLI installation and guides Google sign-in.
---

# Colab Text-to-Image (SDXL-Turbo / Z-Image-Turbo)

One entry point, callable from any directory:

```bash
T2I=~/.claude/skills/colab-text-to-image/scripts/t2i.py
python3 $T2I <doctor|login|generate|stop> [options] --json
```

Always pass `--json` when you (an agent) call it: stdout is then exactly one JSON object and progress goes to stderr. No command ever waits for keyboard input except `login`, which refuses to run without a terminal.

## Workflow (follow in order)

### 1. Check the environment — `doctor`
```bash
python3 $T2I doctor --json
```
It checks macOS/Linux, **installs the `colab` CLI automatically if missing** (uv → pipx → pip --user), fixes the TLS certificate setup, and checks Google sign-in. Act on the exit code:

| Exit | Meaning | What to do |
|---|---|---|
| 0 | Ready | Go to step 2. |
| 3 | Not signed in | Guide the user through sign-in (below), then rerun `doctor`. |
| 4 | CLI could not be installed | Show `attempts` and `hint` from the JSON to the user. |
| 8 | Unsupported OS (Windows) | Tell the user; the colab CLI supports only macOS and Linux. |

If `path_hint` is present, mention it once; it does not block anything.

### Guiding Google sign-in (exit 3)
Sign-in needs a human: Google shows an authorization code in the browser that must be pasted into the CLI. You cannot complete it, and you must never type the code or a password for the user.
1. If you have a tool that opens a terminal tab for the user (e.g. `run_in_terminal`), run the command from the JSON `login_steps[0]` there (`colab sessions`, or the full path shown). Otherwise ask the user to run `python3 ~/.claude/skills/colab-text-to-image/scripts/t2i.py login` in their own terminal.
2. Tell the user the steps from `login_steps` in their language. In short: open the printed Google URL → sign in with the account that has Colab → click Allow → copy the code → paste it into the terminal → Enter.
3. Wait for the user to say they are done, then rerun `doctor --json`. Only proceed on exit 0.

### 2. Pick a model
| `-m` | GPU | Default size | Speed (warm session) | Use when |
|---|---|---|---|---|
| `sdxl-turbo` (default) | T4 | 512×512 | ~2 s/image | Drafts, quick iterations, many variations, low compute cost |
| `z-image-turbo` | L4 (24 GB) | 1024×1024 | ~21 s/image | Final assets, detailed or complex scenes, accurate prompt following, **any text in the image (Chinese or English signs, labels, posters)** |

- Licenses: Z-Image-Turbo is Apache-2.0 (commercial use fine). SDXL-Turbo uses Stability AI's own license — if the user needs images for commercial use, prefer `z-image-turbo` or tell them to check that license.
- L4 needs Colab compute units (paid plan or Google AI Pro/Ultra units). If `generate -m z-image-turbo` exits 5, tell the user L4 is not available on their account right now and offer `sdxl-turbo`.
- Each model has its own session (`t2i` on T4, `t2i-z` on L4), so switching models does not reload anything; `stop` without `-s` stops both.
- `z-image-turbo` ignores `--negative` (no CFG): state what you want positively in the prompt.

### 3. Generate — `generate`
```bash
python3 $T2I generate -p "<prompt>" [-m sdxl-turbo|z-image-turbo] [-n 1] [-W ..] [-H ..] [--steps ..] [--seed N] [--negative "<text>"] [--remove-bg] [-o <dir>] [--stop] --json
```
- `sdxl-turbo`: write the prompt in English (translate if needed); short and concrete. Weak at text/letters and exact counts above ~3.
- `z-image-turbo`: English prompts are tested; Chinese prompts are officially supported but untested here, so prefer English with any in-image text quoted in its original language. Longer, detailed descriptions help. For text in the image, put the exact words in quotes, e.g. `a wooden sign that reads "早安咖啡"`.
- `-o`: save next to where the image is needed (e.g. the project's `assets/`); default `./outputs`.
- First call in a new session: ~2 min for `sdxl-turbo`, ~4 min for `z-image-turbo` (VM + model download).
- Sizes/steps: `sdxl-turbo` 256–1024 px (512 native, ≤768 recommended), steps 1–8 (default 1); `z-image-turbo` 512–1280 px (1024 native), steps 1–16 (default 9).
- Use `--stop` for a one-off image; omit it when more images will follow.
- `--remove-bg`: also saves `<name>_nobg.png` with a transparent background (BiRefNet, MIT license; ~10 s extra load the first time, ~1 s per image). Use it for game sprites, stickers, icons, product shots, or anything to be placed over other content. Unless the prompt already mentions a background, the script appends "isolated on a plain white background" (turn off with `--no-bg-hint`). Add "full body, centered" so the subject is not cut off by the frame.

Exit codes: 0 ok · 2 bad arguments · 3 not signed in · 4 CLI missing · 5 could not get a GPU (no quota/entitlement; a temporary "Service Unavailable" is already retried 3 times) · 6 generation/download failed (see `detail`; for CUDA OOM lower size/`-n`) · 7 verification failed.

### 4. Verify — every time, before reporting success
Automatic checks (in JSON, per image under `checks`): file downloaded, valid PNG, exact requested size, not blank (`pixel_std`), and `clip_score` = prompt–image match (CLIP ViT-B/32 × 100). Measured: matching images score ~30–37; an unrelated prompt scored ~12. Below 22 adds a warning; `--min-score X` turns it into a failure (exit 7).

With `--remove-bg`, each image also gets `nobg_path` and `nobg_checks`: cut-out downloaded, same size, has an alpha channel, `transparent_ratio` between 5% and 97% (otherwise the background was not removed or the subject was lost → failure), and `edge_opaque_ratio` (opaque pixels on the border; above 10% warns that the subject touches the frame or background was left).

Then **look at every image yourself** (Read the PNG file) and check it against the user's actual requirements, not just the prompt:
- Subject(s) present and correct; requested count of objects/people
- Requested style, colors, setting, composition/orientation
- No obvious defects that matter for the use (distorted faces/hands, garbled or misspelled text — read any text character by character, cut-off subject)
- With `--remove-bg`, open the `_nobg.png` too: the whole subject is kept (no missing limbs, hair, weapons or props), no leftover background patches or halos. Thin gaps (between an arm and a sword) should be transparent.

If an image fails: regenerate with a different `--seed` and/or a more specific prompt (move the most important element to the front, add the missing detail). Retry at most 2 times, then show the best result and say honestly what does not match. Never claim an image matches without having looked at it.

When reporting, give the local path(s), the seed(s), and show the image(s) to the user (e.g. SendUserFile if available).

### 5. Clean up — `stop`
The session (`t2i` or `t2i-z`) stays running between calls so follow-ups are fast, but an idle VM burns Colab compute units. When the image work is done (or before ending the conversation), run:
```bash
python3 $T2I stop --json
```

## JSON output (generate)
```jsonc
{"ok": true, "prompt": "...", "model": "stabilityai/sdxl-turbo", "gpu": "T4", "session": "t2i",
 "session_reused": true, "session_stopped": false, "seconds": 9.0, "all_passed": true,
 "remove_bg": false,
 "images": [{"path": "/abs/outputs/20261123-203838_1.png", "seed": 7, "width": 512, "height": 512,
             "checks": {"exists": true, "bytes": 369547, "png_valid": true, "size_ok": true,
                        "not_blank": true, "clip_score": 36.08, "clip_ok": true},
             "problems": [], "warnings": [], "passed": true}],
 // with --remove-bg each image also has:
 //   "nobg_path": "/abs/outputs/..._1_nobg.png",
 //   "nobg_checks": {"exists": true, "size_ok": true, "has_alpha": true,
 //                   "transparent_ratio": 0.42, "edge_opaque_ratio": 0.01}
 "next_step": "Look at each image and confirm it shows what was asked ..."}
```
On failure: `{"ok": false, "exit_code": N, "error": "...", ...}` plus `login_steps`, `hint`, `detail` or the per-image `problems` when relevant.

## Troubleshooting
- "Session not found" after a long idle: the backend reclaimed the VM; `generate` recreates it automatically.
- Wedged kernel: `colab restart-kernel -s t2i` (or `-s t2i-z`), then retry.
- "Connection was lost" from the kernel websocket happens occasionally; `generate` retries once automatically.
- Compute balance: `colab usage`.
- Raw `colab exec` (outside this script) failing with `WebSocketConnectionClosedException` / `CERTIFICATE_VERIFY_FAILED` in `~/.config/colab-cli/colab.log`: set `SSL_CERT_FILE` to the CLI's certifi bundle (t2i.py does this itself; `doctor` prints the path).
