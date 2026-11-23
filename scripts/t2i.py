#!/usr/bin/env python3
"""Text-to-image on a Google Colab GPU via the official `colab` CLI.

Models: sdxl-turbo (default; T4, fast and cheap) and z-image-turbo (L4; much
better quality, follows prompts closely, renders Chinese/English text).

Designed to be called by humans and by AI agents. Standard library only.

Subcommands
  doctor    Check OS, colab CLI (installs it if missing), TLS certs and login.
  login     Interactive Google sign-in for the colab CLI (needs a terminal).
  generate  Generate images, download them, and verify the files.
  stop      Release the Colab VM.

Agent usage: add --json. stdout then carries exactly one JSON object; all
progress goes to stderr. No command ever waits for keyboard input except
`login`, which refuses to run without a terminal.

Exit codes
  0 ok | 2 bad arguments | 3 login required | 4 colab CLI missing/install failed
  5 could not allocate VM | 6 generation/download failed | 7 verification failed
  8 unsupported OS
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATE_PY = SCRIPT_DIR / "generate.py"
# Per-model defaults. Each model gets its own session name so a T4 session is
# never asked to load a model that needs an L4.
MODEL_DEFAULTS = {
    "sdxl-turbo":    {"gpu": "T4", "session": "t2i",   "size": 512,  "steps": 1,
                      "size_range": (256, 1024), "steps_range": (1, 8)},
    "z-image-turbo": {"gpu": "L4", "session": "t2i-z", "size": 1024, "steps": 9,
                      "size_range": (512, 1280), "steps_range": (1, 16)},
}
DEFAULT_MODEL = "sdxl-turbo"
DEFAULT_SESSION = MODEL_DEFAULTS[DEFAULT_MODEL]["session"]
AUTH_MARKER = "To authorize colab-cli"

EXIT_OK, EXIT_USAGE, EXIT_LOGIN, EXIT_CLI = 0, 2, 3, 4
EXIT_VM, EXIT_GEN, EXIT_VERIFY, EXIT_OS = 5, 6, 7, 8

LOGIN_STEPS = [
    "Open a terminal and run: {colab} sessions",
    "Open the Google URL it prints in a browser and sign in with the Google account that has Colab.",
    "Click Allow. Google then shows an authorization code.",
    "Paste the code back into the terminal and press Enter.",
    "Success looks like '[colab] No active sessions found on server.' (or a list of sessions).",
]


def login_steps(colab: str) -> list[str]:
    cmd = "colab" if shutil.which("colab") == colab else colab
    return [s.format(colab=cmd) for s in LOGIN_STEPS]


class T2IError(Exception):
    def __init__(self, code: int, message: str, **extra):
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra


def log(msg: str) -> None:
    print(f"[t2i] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# colab CLI discovery, installation and environment
# --------------------------------------------------------------------------

def _candidate_bin_dirs() -> list[Path]:
    dirs = [Path.home() / ".local" / "bin"]
    uv = shutil.which("uv")
    if uv:
        try:
            out = subprocess.run([uv, "tool", "dir", "--bin"], capture_output=True,
                                 text=True, timeout=20).stdout.strip()
            if out:
                dirs.insert(0, Path(out))
        except (OSError, subprocess.SubprocessError):
            pass
    return dirs


def find_colab() -> str | None:
    found = shutil.which("colab")
    if found:
        return found
    for d in _candidate_bin_dirs():
        p = d / "colab"
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def _find_uv() -> str | None:
    uv = shutil.which("uv")
    if uv:
        return uv
    for p in (Path.home() / ".local/bin/uv", Path.home() / ".cargo/bin/uv",
              Path("/opt/homebrew/bin/uv"), Path("/usr/local/bin/uv")):
        if p.is_file():
            return str(p)
    return None


def install_colab() -> str:
    """Install google-colab-cli with uv, pipx or pip --user. Returns colab path."""
    attempts = []
    uv = _find_uv()
    if uv:
        attempts.append([uv, "tool", "install", "google-colab-cli"])
    if shutil.which("pipx"):
        attempts.append(["pipx", "install", "google-colab-cli"])
    attempts.append([sys.executable, "-m", "pip", "install", "--user", "google-colab-cli"])

    errors = []
    for cmd in attempts:
        log("installing colab CLI: " + " ".join(cmd))
        try:
            r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=sys.stderr,
                               stderr=sys.stderr, timeout=600)
        except (OSError, subprocess.SubprocessError) as e:
            errors.append(f"{cmd[0]}: {e}")
            continue
        if r.returncode == 0:
            path = find_colab()
            if path:
                log(f"colab CLI installed at {path}")
                return path
            errors.append(f"{' '.join(cmd)} succeeded but `colab` is not on PATH")
        else:
            errors.append(f"{' '.join(cmd)} exited {r.returncode}")
    raise T2IError(EXIT_CLI, "Could not install the colab CLI automatically.",
                   attempts=errors,
                   hint="Install manually: uv tool install google-colab-cli "
                        "(get uv from https://docs.astral.sh/uv/), then re-run.")


def ensure_colab(auto_install: bool) -> str:
    if platform.system() not in ("Darwin", "Linux"):
        raise T2IError(EXIT_OS, "The colab CLI supports only macOS and Linux "
                       f"(this is {platform.system()}).")
    path = find_colab()
    if path:
        return path
    if not auto_install:
        raise T2IError(EXIT_CLI, "colab CLI is not installed.",
                       hint="Run: uv tool install google-colab-cli")
    return install_colab()


def colab_env(colab: str) -> dict:
    """Environment for colab calls.

    python.org builds of Python on macOS have no CA store, so the CLI's kernel
    websocket fails with CERTIFICATE_VERIFY_FAILED even though plain HTTPS
    works. Point OpenSSL at the certifi bundle inside the CLI's own env.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONWARNINGS", "ignore")
    if env.get("SSL_CERT_FILE"):
        return env
    try:
        first = Path(colab).read_text(errors="ignore").splitlines()[0]
    except (OSError, IndexError):
        return env
    if first.startswith("#!"):
        interp = first[2:].strip().split()[0]
        try:
            cert = subprocess.run([interp, "-c", "import certifi;print(certifi.where())"],
                                  capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            cert = ""
        if cert and Path(cert).is_file():
            env["SSL_CERT_FILE"] = cert
    return env


def run_colab(colab: str, env: dict, args: list[str], timeout: float = 300,
              stdin_data: str | None = None, echo: bool = True) -> tuple[int, str]:
    """Run colab non-interactively. Output is echoed to stderr and returned."""
    try:
        p = subprocess.run([colab, *args], input=stdin_data if stdin_data is not None else "",
                           capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise T2IError(EXIT_GEN, f"colab {args[0]} timed out after {timeout:.0f}s")
    out = (p.stdout or "") + (p.stderr or "")
    if echo and out.strip():
        sys.stderr.write(out if out.endswith("\n") else out + "\n")
        sys.stderr.flush()
    if AUTH_MARKER in out:
        raise T2IError(EXIT_LOGIN, "The colab CLI is not signed in to Google.",
                       login_steps=login_steps(colab))
    return p.returncode, out


def check_login(colab: str, env: dict) -> None:
    code, out = run_colab(colab, env, ["sessions"], timeout=90, echo=False)
    if code != 0 and "Aborted" in out:
        raise T2IError(EXIT_LOGIN, "The colab CLI is not signed in to Google.",
                       login_steps=login_steps(colab))


# --------------------------------------------------------------------------
# Local verification
# --------------------------------------------------------------------------

PNG_COLOR_RGBA, PNG_COLOR_GRAY_ALPHA = 6, 4


def png_info(path: Path) -> tuple[int, int, int]:
    """Return (width, height, color_type) from the PNG IHDR chunk."""
    with path.open("rb") as f:
        head = f.read(26)
    if len(head) < 26 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError("not a valid PNG file")
    w, h = struct.unpack(">II", head[16:24])
    return w, h, head[25]


def verify_cutout(local: Path, meta: dict, want_w: int, want_h: int) -> dict:
    """Checks for the transparent-background copy."""
    checks: dict = {}
    problems: list[str] = []
    warnings: list[str] = []
    checks["exists"] = local.is_file()
    if not checks["exists"]:
        problems.append("cut-out file was not downloaded")
    else:
        try:
            w, h, color = png_info(local)
            checks["size_ok"] = (w, h) == (want_w, want_h)
            checks["has_alpha"] = color in (PNG_COLOR_RGBA, PNG_COLOR_GRAY_ALPHA)
            if not checks["size_ok"]:
                problems.append(f"cut-out size {w}x{h} != {want_w}x{want_h}")
            if not checks["has_alpha"]:
                problems.append("cut-out PNG has no alpha channel")
        except (OSError, ValueError) as e:
            problems.append(f"cut-out: {e}")

    tr = meta.get("transparent_ratio")
    if tr is not None:
        checks["transparent_ratio"] = tr
        if tr < 0.05:
            problems.append(f"background was not removed (only {tr:.0%} transparent); "
                            "use a plainer background in the prompt")
        elif tr > 0.97:
            problems.append(f"almost everything was removed ({tr:.0%} transparent); "
                            "the subject was probably not detected")
    edge = meta.get("edge_opaque_ratio")
    if edge is not None:
        checks["edge_opaque_ratio"] = edge
        if edge > 0.10:
            warnings.append(f"{edge:.0%} of the border is opaque: the subject may be cut off "
                            "by the frame or background was left; add 'full body, centered' "
                            "or check the cut-out")
    return {"checks": checks, "problems": problems, "warnings": warnings}


def verify_image(local: Path, meta: dict, want_w: int, want_h: int,
                 min_score: float | None) -> dict:
    checks: dict = {}
    problems: list[str] = []
    warnings: list[str] = []

    checks["exists"] = local.is_file()
    size = local.stat().st_size if checks["exists"] else 0
    checks["bytes"] = size
    if not checks["exists"]:
        problems.append("file was not downloaded")
    elif size < 1024:
        problems.append(f"file is suspiciously small ({size} bytes)")
    if checks["exists"]:
        try:
            w, h, _ = png_info(local)
            checks["png_valid"] = True
            checks["size_ok"] = (w, h) == (want_w, want_h)
            if not checks["size_ok"]:
                problems.append(f"size {w}x{h} != requested {want_w}x{want_h}")
        except (OSError, ValueError) as e:
            checks["png_valid"] = False
            problems.append(str(e))

    std = meta.get("pixel_std")
    if std is not None:
        checks["not_blank"] = std >= 8.0
        if not checks["not_blank"]:
            problems.append(f"image is nearly uniform (pixel std {std:.1f}); likely blank/black")

    score = meta.get("clip_score")
    if score is not None:
        threshold = min_score if min_score is not None else 22.0
        checks["clip_score"] = round(score, 2)
        checks["clip_ok"] = score >= threshold
        if not checks["clip_ok"]:
            msg = (f"low prompt match (CLIP score {score:.1f} < {threshold}); "
                   "inspect the image and consider a new seed or a clearer prompt")
            (problems if min_score is not None else warnings).append(msg)

    return {"checks": checks, "problems": problems, "warnings": warnings,
            "passed": not problems}


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

def cmd_doctor(a) -> dict:
    report = {"os": platform.system()}
    colab = ensure_colab(auto_install=not a.no_install)
    report["colab_path"] = colab
    if shutil.which("colab") != colab:
        report["path_hint"] = (f"{Path(colab).parent} is not on PATH; add it to your shell "
                               "profile to call `colab` directly (t2i works either way).")
    env = colab_env(colab)
    report["ssl_cert_file"] = env.get("SSL_CERT_FILE")
    try:
        v = subprocess.run([colab, "version"], capture_output=True, text=True,
                           env=env, timeout=30, stdin=subprocess.DEVNULL).stdout
        report["colab_version"] = (re.search(r"[\d.]+", v) or [None])[0]
    except (OSError, subprocess.SubprocessError):
        report["colab_version"] = None
    check_login(colab, env)
    report["logged_in"] = True
    log(f"ready: colab {report['colab_version']} at {colab}, signed in")
    return report


def cmd_login(a) -> dict:
    colab = ensure_colab(auto_install=not a.no_install)
    env = colab_env(colab)
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise T2IError(EXIT_LOGIN, "Login needs a human at an interactive terminal.",
                       login_steps=login_steps(colab),
                       command=f"{colab} sessions")
    print("\n".join(f"  {i}. {s}" for i, s in enumerate(login_steps(colab)[1:], 1)))
    subprocess.run([colab, "sessions"], env=env)
    check_login(colab, env)
    log("signed in")
    return {"logged_in": True}


def _session_alive(colab: str, env: dict, session: str) -> bool:
    _, out = run_colab(colab, env, ["status", "-s", session], timeout=90, echo=False)
    return "not found" not in out and session in out


def _create_session(colab: str, env: dict, session: str, gpu: str) -> None:
    log(f"creating session '{session}' with GPU {gpu} (about a minute) ...")
    for attempt in range(3):
        code, out = run_colab(colab, env, ["new", "-s", session, "--gpu", gpu], timeout=600)
        if "READY" in out or "Service Unavailable" not in out:
            break
        log(f"{gpu} temporarily unavailable; retrying in 20 s ({attempt + 1}/3)")
        time.sleep(20)
    if code != 0 or "READY" not in out:
        hint = ("No quota for this GPU right now. Retry later or try another --gpu."
                if "400" in out else "See colab output above.")
        raise T2IError(EXIT_VM, f"Could not allocate a {gpu} session.", hint=hint)


def _exec_generate(colab: str, env: dict, a, tag: str) -> dict:
    params = {
        "T2I_MODEL": a.model, "T2I_PROMPT": a.prompt, "T2I_NEGATIVE": a.negative or "", "T2I_N": str(a.num),
        "T2I_WIDTH": str(a.width), "T2I_HEIGHT": str(a.height),
        "T2I_STEPS": str(a.steps), "T2I_SEED": "" if a.seed is None else str(a.seed),
        "T2I_TAG": tag, "T2I_SCORE": "0" if a.no_score else "1",
        "T2I_REMOVE_BG": "1" if a.remove_bg else "0",
    }
    args = ["exec", "-s", a.session, "-f", str(GENERATE_PY), "--timeout", "900"]
    for k, v in params.items():          # values are passed as data, never as code
        args += ["--env", f"{k}={v}"]
    _, out = run_colab(colab, env, args, timeout=1800)
    if "T2I_OUTPUTS=" not in out and "Connection was lost" in out:
        # The kernel websocket drops now and then; the kernel itself survives.
        log("connection to the Colab kernel was lost; retrying once")
        _, out = run_colab(colab, env, args, timeout=1800)
    m = None
    for line in out.splitlines():
        if line.startswith("T2I_OUTPUTS="):
            m = line
    if not m:
        err = next((l for l in reversed(out.splitlines()) if "Error" in l), "")
        raise T2IError(EXIT_GEN, "Generation failed on the Colab VM.",
                       detail=err.strip()[:500])
    return json.loads(m.split("=", 1)[1])


def cmd_generate(a) -> dict:
    d = MODEL_DEFAULTS[a.model]
    a.session = a.session or d["session"]
    a.gpu = a.gpu or d["gpu"]
    a.width = a.width or d["size"]
    a.height = a.height or d["size"]
    a.steps = a.steps or d["steps"]
    if not a.prompt or not a.prompt.strip():
        raise T2IError(EXIT_USAGE, "--prompt is required")
    if not (1 <= a.num <= 8):
        raise T2IError(EXIT_USAGE, "--num must be 1-8")
    lo, hi = d["steps_range"]
    if not (lo <= a.steps <= hi):
        raise T2IError(EXIT_USAGE, f"--steps must be {lo}-{hi} for {a.model}")
    lo, hi = d["size_range"]
    for v in (a.width, a.height):
        if not (lo <= v <= hi):
            raise T2IError(EXIT_USAGE, f"--width/--height must be {lo}-{hi} for {a.model}")
    if a.negative and a.model == "z-image-turbo":
        log("note: z-image-turbo runs without CFG, so --negative has no effect; "
            "describe what you want in the prompt instead")
    want_w, want_h = a.width // 8 * 8, a.height // 8 * 8
    a.prompt = a.prompt.strip()
    if a.remove_bg and not a.no_bg_hint and "background" not in a.prompt.lower():
        # A plain backdrop makes the cut-out much cleaner.
        a.prompt += ", isolated on a plain white background"
        log(f"prompt for cut-out: {a.prompt}")

    colab = ensure_colab(auto_install=not a.no_install)
    env = colab_env(colab)
    check_login(colab, env)

    reused = _session_alive(colab, env, a.session)
    if reused:
        log(f"reusing session '{a.session}'")
    else:
        _create_session(colab, env, a.session, a.gpu)

    tag = time.strftime("%Y%m%d-%H%M%S")
    t0 = time.time()
    try:
        result = _exec_generate(colab, env, a, tag)
    except T2IError as e:
        # The backend may have reclaimed a reused VM between status and exec.
        if e.code == EXIT_GEN and reused and not _session_alive(colab, env, a.session):
            log("session vanished; recreating once")
            _create_session(colab, env, a.session, a.gpu)
            result = _exec_generate(colab, env, a, tag)
        else:
            raise

    out_dir = Path(a.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for meta in result["images"]:
        remote = meta["path"]
        local = out_dir / Path(remote).name
        code, _ = run_colab(colab, env, ["download", "-s", a.session, remote, str(local)],
                            timeout=300)
        v = verify_image(local, meta, want_w, want_h, a.min_score)
        if code != 0:
            v["passed"] = False
            v["problems"].insert(0, f"colab download exited {code}")
        entry = {"path": str(local), "remote_path": remote, "seed": meta["seed"],
                 "width": want_w, "height": want_h, **v}
        print(local, file=sys.stderr if a.json else sys.stdout, flush=True)
        if a.remove_bg:
            if "nobg_path" not in meta:
                entry["passed"] = False
                entry["problems"].append("background removal produced no file")
            else:
                nobg_local = out_dir / Path(meta["nobg_path"]).name
                code, _ = run_colab(colab, env, ["download", "-s", a.session,
                                                 meta["nobg_path"], str(nobg_local)], timeout=300)
                cv = verify_cutout(nobg_local, meta, want_w, want_h)
                if code != 0:
                    cv["problems"].insert(0, f"colab download of cut-out exited {code}")
                entry["nobg_path"] = str(nobg_local)
                entry["nobg_checks"] = cv["checks"]
                entry["problems"] += cv["problems"]
                entry["warnings"] += cv["warnings"]
                entry["passed"] = entry["passed"] and not cv["problems"]
                print(nobg_local, file=sys.stderr if a.json else sys.stdout, flush=True)
        images.append(entry)

    stopped = False
    if a.stop:
        run_colab(colab, env, ["stop", "-s", a.session], timeout=120)
        stopped = True
    else:
        log(f"session '{a.session}' left running (uses compute units while idle); "
            f"release it with: {sys.argv[0]} stop -s {a.session}")

    report = {
        "prompt": a.prompt, "negative": a.negative or "", "model": result.get("model"),
        "gpu": a.gpu,
        "steps": a.steps, "remove_bg": a.remove_bg,
        "session": a.session, "session_reused": reused,
        "session_stopped": stopped, "seconds": round(time.time() - t0, 1),
        "images": images,
        "all_passed": all(i["passed"] for i in images) and len(images) == a.num,
        "next_step": ("Look at each image and confirm it shows what was asked "
                      "(subject, style, count, key details). If not, regenerate with "
                      "a new --seed or a more specific prompt."
                      + (" Also open each *_nobg.png: the whole subject should be kept "
                         "(no missing limbs/hair/props) and no background patches left."
                         if a.remove_bg else "")),
    }
    if len(images) != a.num:
        report["problem"] = f"expected {a.num} images, got {len(images)}"
    if not report["all_passed"]:
        raise T2IError(EXIT_VERIFY, "Some images failed verification.", **report)
    for im in images:
        for w in im["warnings"]:
            log(f"warning ({Path(im['path']).name}): {w}")
    return report


def cmd_stop(a) -> dict:
    colab = ensure_colab(auto_install=False)
    env = colab_env(colab)
    names = [a.session] if a.session else [d["session"] for d in MODEL_DEFAULTS.values()]
    stopped = []
    for name in names:
        _, out = run_colab(colab, env, ["stop", "-s", name], timeout=120)
        if "not found" not in out:
            stopped.append(name)
    if not stopped:
        log("no t2i session was running")
    return {"sessions_checked": names, "stopped": stopped}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="t2i", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true",
                        help="print one JSON object on stdout (for agents)")
    common.add_argument("--no-install", action="store_true",
                        help="do not install the colab CLI if it is missing")
    common.add_argument("-s", "--session", default=None,
                        help="session name (default: t2i for sdxl-turbo, t2i-z for z-image-turbo)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", parents=[common], help="check/install CLI and login")
    sub.add_parser("login", parents=[common], help="interactive Google sign-in")
    sub.add_parser("stop", parents=[common], help="release the Colab VM")

    g = sub.add_parser("generate", parents=[common], help="generate images")
    g.add_argument("-p", "--prompt", required=True)
    g.add_argument("-m", "--model", choices=list(MODEL_DEFAULTS), default=DEFAULT_MODEL,
                   help="sdxl-turbo: T4, fast/cheap (default). z-image-turbo: L4, best "
                        "quality, can draw Chinese/English text")
    g.add_argument("--negative", default="")
    g.add_argument("-n", "--num", type=int, default=1)
    g.add_argument("-W", "--width", type=int, default=None,
                   help="default 512 (sdxl-turbo) / 1024 (z-image-turbo)")
    g.add_argument("-H", "--height", type=int, default=None)
    g.add_argument("--steps", type=int, default=None,
                   help="default 1 (sdxl-turbo) / 9 (z-image-turbo)")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("-o", "--out", default="./outputs")
    g.add_argument("--gpu", default=None, help="default T4 (sdxl-turbo) / L4 (z-image-turbo)")
    g.add_argument("--stop", action="store_true", help="release the VM afterwards")
    g.add_argument("--remove-bg", action="store_true",
                   help="also save a transparent-background copy (*_nobg.png, BiRefNet)")
    g.add_argument("--no-bg-hint", action="store_true",
                   help="with --remove-bg, do not append 'plain white background' to the prompt")
    g.add_argument("--no-score", action="store_true",
                   help="skip the CLIP prompt-match score")
    g.add_argument("--min-score", type=float, default=None,
                   help="fail verification if CLIP score is below this (measured: matching ~30-37, unrelated ~12)")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backward compatibility with the old flag-only interface.
    if argv and argv[0] == "--stop-only":
        argv = ["stop", *argv[1:]]
    elif argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["generate", *argv]

    parser = build_parser()
    try:
        a = parser.parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code else EXIT_OK
    handler = {"doctor": cmd_doctor, "login": cmd_login,
               "generate": cmd_generate, "stop": cmd_stop}[a.cmd]
    try:
        result = handler(a)
        if a.json:
            print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
        return EXIT_OK
    except T2IError as e:
        payload = {"ok": False, "exit_code": e.code, "error": e.message, **e.extra}
        if a.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            log(f"error: {e.message}")
            for k, v in e.extra.items():
                if k == "login_steps":
                    log("to sign in:")
                    for i, s in enumerate(v, 1):
                        print(f"      {i}. {s}", file=sys.stderr)
                elif k in ("hint", "detail", "attempts", "problem"):
                    log(f"{k}: {v}")
                elif k == "images":
                    for im in v:
                        for pr in im["problems"]:
                            log(f"{Path(im['path']).name}: {pr}")
        return e.code
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
