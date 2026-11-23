---
name: colab-text-to-image
description: Generate images from a text prompt with SDXL-Turbo running on a Google Colab GPU through the official `colab` CLI, save the PNGs locally, and verify them. Use whenever an image/picture/illustration/icon/concept art must be created from text (文字生圖、生成圖片、畫一張圖) — whether the user asks directly or another task (a web page, slide, document, game asset) needs a generated image — or to make variations of a prompt. Handles CLI installation and guides Google sign-in.
---

# Colab Text-to-Image (SDXL-Turbo)

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

### 2. Generate — `generate`
```bash
python3 $T2I generate -p "<English prompt>" [-n 1] [-W 512] [-H 512] [--steps 1] [--seed N] [--negative "<text>"] [-o <dir>] [--stop] --json
```
- Write the prompt in English (translate if needed): subject, count, setting, style, lighting. SDXL-Turbo follows short concrete prompts best; it is weak at rendering text/letters and exact counts above ~3.
- `-o`: save next to where the image is needed (e.g. the project's `assets/`); default `./outputs`.
- First call in a new session: ~2 min (VM + model download). Later calls in the same session: a few seconds per image.
- `--steps` 1–4 (default 1). 512×512 is native; up to 768 is fine; range allowed 256–1024.
- Use `--stop` for a one-off image; omit it when more images will follow.

Exit codes: 0 ok · 2 bad arguments · 3 not signed in · 4 CLI missing · 5 could not get a GPU (quota; retry later or `--gpu L4`) · 6 generation/download failed (see `detail`; for CUDA OOM lower size/`-n`) · 7 verification failed.

### 3. Verify — every time, before reporting success
Automatic checks (in JSON, per image under `checks`): file downloaded, valid PNG, exact requested size, not blank (`pixel_std`), and `clip_score` = prompt–image match (CLIP ViT-B/32 × 100). Measured: matching images score ~30–37; an unrelated prompt scored ~12. Below 22 adds a warning; `--min-score X` turns it into a failure (exit 7).

Then **look at every image yourself** (Read the PNG file) and check it against the user's actual requirements, not just the prompt:
- Subject(s) present and correct; requested count of objects/people
- Requested style, colors, setting, composition/orientation
- No obvious defects that matter for the use (distorted faces/hands, garbled text, cut-off subject)

If an image fails: regenerate with a different `--seed` and/or a more specific prompt (move the most important element to the front, add the missing detail). Retry at most 2 times, then show the best result and say honestly what does not match. Never claim an image matches without having looked at it.

When reporting, give the local path(s), the seed(s), and show the image(s) to the user (e.g. SendUserFile if available).

### 4. Clean up — `stop`
The session `t2i` stays running between calls so follow-ups are fast, but an idle VM burns Colab compute units. When the image work is done (or before ending the conversation), run:
```bash
python3 $T2I stop --json
```

## JSON output (generate)
```json
{"ok": true, "prompt": "...", "model": "stabilityai/sdxl-turbo", "session": "t2i",
 "session_reused": true, "session_stopped": false, "seconds": 9.0, "all_passed": true,
 "images": [{"path": "/abs/outputs/20261123-203838_1.png", "seed": 7, "width": 512, "height": 512,
             "checks": {"exists": true, "bytes": 369547, "png_valid": true, "size_ok": true,
                        "not_blank": true, "clip_score": 36.08, "clip_ok": true},
             "problems": [], "warnings": [], "passed": true}],
 "next_step": "Look at each image and confirm it shows what was asked ..."}
```
On failure: `{"ok": false, "exit_code": N, "error": "...", ...}` plus `login_steps`, `hint`, `detail` or the per-image `problems` when relevant.

## Troubleshooting
- "Session not found" after a long idle: the backend reclaimed the VM; `generate` recreates it automatically.
- Wedged kernel: `colab restart-kernel -s t2i`, then retry.
- Compute balance: `colab usage`.
- Raw `colab exec` (outside this script) failing with `WebSocketConnectionClosedException` / `CERTIFICATE_VERIFY_FAILED` in `~/.config/colab-cli/colab.log`: set `SSL_CERT_FILE` to the CLI's certifi bundle (t2i.py does this itself; `doctor` prints the path).
