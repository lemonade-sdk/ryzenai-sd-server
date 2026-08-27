#!/usr/bin/env python3
"""
Shared helpers for the per-mode ryzenai-sd-server test scripts
(test_txt2img.py, test_img2img.py, test_variants.py, test_controlnet.py,
test_cli.py).

Config loading, model discovery/auto-download, server launch/stop, image
saving, and tagged logging all live here so each per-mode script only needs
to define its own request bodies and a couple of `run_smoke`/`run_detailed`
callables.
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image


# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_JSON = SCRIPT_DIR / "models.json"
DEFAULT_MODELS_DIR = Path(os.environ.get("RAI_SD_MODELS_DIR", r"C:\Users\mickraus\Work\rai_1.8.0_models"))
OUTPUT_DIR = PROJECT_ROOT / "test_outputs"
DEFAULT_INPUT_IMAGE = SCRIPT_DIR / "img2img_test_input.png"

LEVELS = ("smoke", "detailed")
SMOKE_STEPS = 1


# ─── Config Loading ──────────────────────────────────────────────────────────

def load_config():
    """Load model configurations from models.json."""
    with open(MODELS_JSON) as f:
        return json.load(f)


def get_model_config(config, model_name, args=None, level="smoke"):
    """Get merged config for a model (model-specific values override defaults).

    `level` controls steps:
      - "smoke": steps forced to SMOKE_STEPS (fast plumbing check).
      - "detailed": model's own configured steps (full quality run).

    If `args` is given, --seed/--steps/--guidance/--prompt CLI overrides
    (when set) take precedence over both defaults and the level/model config.
    """
    defaults = config["defaults"].copy()
    model_cfg = config["models"].get(model_name)
    if model_cfg:
        defaults.update(model_cfg)

    if level == "smoke":
        defaults["steps"] = SMOKE_STEPS

    if args is not None:
        if args.seed is not None:
            defaults["seed"] = args.seed
        if args.steps is not None:
            defaults["steps"] = args.steps
        if args.guidance is not None:
            defaults["guidance"] = args.guidance
        if args.prompt is not None:
            defaults["prompt"] = args.prompt

    return defaults


def discover_models(models_dir, capability, config):
    """Find models in directory (or HF cache) that declare `capability`.

    Includes models that are:
    - Present as a flat directory under models_dir, OR
    - Registered in models.json with an hf_repo_id whose HF snapshot already
      exists (downloaded by Lemonade or a prior test run).
    """
    found_names: set = set()

    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                found_names.add(d.name)

    for name, model_cfg in config["models"].items():
        if name in found_names:
            continue
        repo_id = model_cfg.get("hf_repo_id")
        if repo_id and get_hf_snapshot_path(repo_id) is not None:
            found_names.add(name)

    if not found_names:
        print(f"ERROR: No models found in {models_dir} or HF cache")
        sys.exit(1)

    result = []
    for name in sorted(found_names):
        model_cfg = config["models"].get(name)
        if model_cfg is None:
            if capability in ("txt2img", "cli"):
                result.append(name)
        elif capability in model_cfg.get("capabilities", []):
            result.append(name)

    return result


# ─── Model Auto-Download ─────────────────────────────────────────────────────

def _hf_hub_cache() -> Path:
    """Return the HuggingFace hub cache root, matching Lemonade's C++ resolution order.

    Priority: HF_HUB_CACHE env var > HF_HOME/hub env var > ~/.cache/huggingface/hub
    This mirrors lemonade's resolve_hf_cache_dir() in path_utils.cpp.
    """
    if val := os.environ.get("HF_HUB_CACHE"):
        return Path(val)
    if val := os.environ.get("HF_HOME"):
        return Path(val) / "hub"
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except ImportError:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_id_to_cache_name(repo_id: str) -> str:
    """Convert 'org/repo' to 'models--org--repo' (Lemonade / HF hub convention)."""
    return "models--" + repo_id.replace("/", "--")


def get_hf_snapshot_path(hf_repo_id: str) -> Optional[Path]:
    """Return the active HF-cache snapshot directory for a repo, or None if not cached.

    Reads <HF_HUB_CACHE>/models--org--repo/refs/main to find the commit hash,
    then returns snapshots/<hash>/ if it exists.  This is the same lookup
    Lemonade's RyzenAISDOps::resolve_checkpoint_path() performs in C++.
    """
    model_cache = _hf_hub_cache() / _repo_id_to_cache_name(hf_repo_id)
    refs_main = model_cache / "refs" / "main"
    if not refs_main.exists():
        return None
    commit = refs_main.read_text(encoding="utf-8").strip()
    if not commit:
        return None
    snapshot = model_cache / "snapshots" / commit
    return snapshot if snapshot.exists() else None


def ensure_model_present(flat_path: Path, model_cfg: dict) -> Optional[Path]:
    """Resolve the usable model directory, downloading from HF if necessary.

    Resolution order (returns first hit):
      1. Flat local path (e.g. rai_1.8.0_models/<name>/) — backward compat.
      2. HF hub cache snapshot — lets Lemonade and this test share one copy.
      3. Download via snapshot_download() to the HF hub cache (no local_dir),
         so the download is also visible to Lemonade without a second download.

    Returns the usable Path, or None on failure.
    """
    if flat_path.exists():
        return flat_path

    repo_id = model_cfg.get("hf_repo_id")

    if repo_id:
        snapshot = get_hf_snapshot_path(repo_id)
        if snapshot is not None:
            print(f"  Found in HF cache: {snapshot}")
            return snapshot

    if not repo_id:
        print(f"  ERROR: Model not found at {flat_path} and no hf_repo_id defined")
        return None

    print(f"  Model not found locally — downloading {repo_id} from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            ignore_patterns=[".cache/**", "__pycache__/**"],
            resume_download=True,
        )
        resolved = Path(snapshot_path)
        print(f"  Download complete: {resolved}")
        return resolved
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "credentials" in msg.lower() or "token" in msg.lower():
            print(f"  ERROR: Authentication required — run 'huggingface-cli login' first")
        else:
            print(f"  ERROR downloading {repo_id}: {msg}")
        return None


# ─── Server Management ───────────────────────────────────────────────────────

def find_server_exe():
    """Auto-discover ryzenai-sd-server.exe from build directory."""
    candidates = [
        PROJECT_ROOT / "build" / "bin" / "Release" / "ryzenai-sd-server.exe",
        PROJECT_ROOT / "build" / "bin" / "Debug" / "ryzenai-sd-server.exe",
        PROJECT_ROOT / "build" / "bin" / "ryzenai-sd-server",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def server_is_up(url):
    """Check if server responds to /health."""
    try:
        r = requests.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def launch_server(model_path, port, extra_args=None, timeout=300):
    """Start ryzenai-sd-server and wait for /health. Returns subprocess.Popen."""
    exe = find_server_exe()
    if not exe:
        print(f"ERROR: ryzenai-sd-server executable not found in build/bin/Release or build/bin/Debug")
        sys.exit(1)

    cmd = [str(exe), "--server", "--model-path", str(model_path), "--port", str(port)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(exe.parent))

    url = f"http://localhost:{port}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_is_up(url):
            print(f"  Server ready (pid {proc.pid})")
            return proc
        if proc.poll() is not None:
            print(f"  ERROR: Server exited early (code {proc.returncode})")
            return None
        time.sleep(2)

    proc.terminate()
    print(f"  ERROR: Server did not become ready within {timeout}s")
    return None


def stop_server(proc):
    """Terminate server process."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ─── Image Handling ──────────────────────────────────────────────────────────

def save_response_image(response_json, output_path):
    """Decode b64_json from server response and save as PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_list = response_json.get("data", [])
    if not data_list:
        raise ValueError("Response contains no image data")

    raw_bytes = base64.b64decode(data_list[0]["b64_json"])
    fmt = response_json.get("format", "png")

    # The server encodes output via stb_image_write as PNG then base64-encodes
    # that PNG — even when it labels the response format "raw_rgb".
    # Always decode through PIL so we handle PNG, JPEG, and true raw RGB uniformly.
    from io import BytesIO
    try:
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        w = response_json.get("width", 0)
        h = response_json.get("height", 0)
        if w <= 0 or h <= 0:
            raise ValueError(f"Cannot decode image: PIL failed and no valid dimensions in response (format={fmt})")
        expected = h * w * 3
        arr = np.frombuffer(raw_bytes[:expected], dtype=np.uint8).reshape(h, w, 3)
        img = Image.fromarray(arr, "RGB")

    img.save(str(output_path), "PNG")
    return output_path


def resolve_shared_input_image(args) -> Path:
    """Resolve the ONE source image used for every model in an img2img/variants
    run, so outputs are directly comparable across models (each model just
    resizes this same source to its own resolution).

    Priority:
      1. --input-image CLI override (must exist on disk).
      2. The cached default at test/img2img_test_input.png, reused across runs
         if already present.
      3. A freshly generated, deterministically-seeded synthetic image, saved
         to the cache path so this run (and future ones) reuse the same image.
    """
    if args.input_image:
        path = Path(args.input_image)
        if not path.exists():
            raise FileNotFoundError(f"--input-image not found: {path}")
        return path

    if DEFAULT_INPUT_IMAGE.exists():
        return DEFAULT_INPUT_IMAGE

    print(f"  No shared input image found — generating a synthetic one at {DEFAULT_INPUT_IMAGE}")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(DEFAULT_INPUT_IMAGE))
    return DEFAULT_INPUT_IMAGE


# ─── Tagged Logging ──────────────────────────────────────────────────────────

def make_logger(mode, model, level, out_dir):
    """Return a `log(msg)` function that prints AND persists to a per-model
    log file, every line tagged `[mode][model][level]`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{mode}_{model}_{level}.log"
    tag = f"[{mode}][{model}][{level}]"
    fh = open(log_path, "a", encoding="utf-8")

    def log(msg=""):
        line = f"{tag} {msg}" if msg else tag
        print(line)
        fh.write(line + "\n")
        fh.flush()

    log.path = log_path
    log.close = fh.close
    return log


# ─── Shared Argparse ─────────────────────────────────────────────────────────

def build_common_parser(description: str) -> argparse.ArgumentParser:
    """Build an argparse parser with the flags shared by all per-mode scripts.

    Every flag is optional with a sensible default — each script must be
    runnable with just --model-path (or --all-models) / --url and nothing
    else.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-path", help="Path to a specific model directory")
    group.add_argument("--all-models", action="store_true", help="Test all discovered models")

    parser.add_argument("--url", help="Server URL (skip auto-launch, test against a running server)")
    parser.add_argument("--port", type=int, default=8080, help="Port for auto-launched server (default: 8080)")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Directory containing model folders")
    parser.add_argument("--model-name", help="Model name (used with --url when server is already running)")

    parser.add_argument("--level", choices=LEVELS, default="smoke",
                         help="smoke: 1 step, fast plumbing check (default). "
                              "detailed: model's full configured steps/guidance, representative image.")

    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument("--steps", type=int, help="Override inference steps")
    parser.add_argument("--guidance", type=float, help="Override guidance scale")
    parser.add_argument("--prompt", help="Override prompt")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and show what would be tested without launching servers or requiring model files.",
    )

    return parser


def validate_model_selection(parser, args):
    """Shared validation: require --url, --model-path, or --all-models unless --dry-run."""
    if not args.dry_run and not args.url and not args.model_path and not args.all_models:
        parser.error("Provide --url, --model-path, or --all-models")


def resolve_model_names(args, config, capability):
    """Determine which model names to test, per the standard CLI conventions."""
    if args.all_models:
        return discover_models(Path(args.models_dir), capability, config)
    if args.model_path:
        return [Path(args.model_path).name]
    return [args.model_name or "unknown"]


# ─── Dry Run ─────────────────────────────────────────────────────────────────

def _dry_run(mode, args, config, capability):
    """Print what would be tested without launching servers or requiring model files."""
    level = args.level
    model_names = resolve_model_names(args, config, capability)

    print(f"Dry run — mode={mode}  level={level}  models-dir={args.models_dir}")
    print(f"Would test {len(model_names)} model(s):\n")

    for model_name in model_names:
        model_cfg = get_model_config(config, model_name, args, level=level)
        flat_path = Path(args.models_dir) / model_name if args.all_models else Path(args.model_path or "")
        repo_id = model_cfg.get("hf_repo_id")

        if flat_path.exists():
            location = f"local: {flat_path}"
        elif repo_id and get_hf_snapshot_path(repo_id) is not None:
            location = f"HF cache: {get_hf_snapshot_path(repo_id)}"
        elif repo_id:
            location = f"NOT FOUND (would download {repo_id})"
        else:
            location = "NOT FOUND (no hf_repo_id)"

        print(f"  {model_name}")
        print(f"    size={model_cfg['width']}x{model_cfg['height']}  steps={model_cfg['steps']}  "
              f"guidance={model_cfg['guidance']}  seed={model_cfg.get('seed', 42)}")
        print(f"    prompt: {model_cfg['prompt']!r}")
        print(f"    location: {location}")
        print()


# ─── Per-Model Runner Loop ───────────────────────────────────────────────────

def run_per_model_loop(mode, args, config, capability, test_fn, needs_server=True):
    """Shared driver used by every per-mode script.

    `test_fn(log, url, model_name, model_cfg, model_path, out_dir, level) ->
    (success: bool, detail: str)` performs the actual request(s) for one
    model. This loop handles model discovery, download, server
    launch/stop (skipped entirely when `needs_server=False`, e.g. the CLI
    mode which runs the executable directly), tagged logging, per-model
    try/except (never aborts the whole run), and the final report.json +
    exit code.
    """
    port = args.port
    url = args.url or f"http://localhost:{port}"
    level = args.level

    if args.dry_run:
        _dry_run(mode, args, config, capability)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / mode / ts

    model_names = resolve_model_names(args, config, capability)

    results = []
    for model_name in model_names:
        model_cfg = get_model_config(config, model_name, args, level=level)
        flat_path = Path(args.models_dir) / model_name if args.all_models else Path(args.model_path or "")

        log = make_logger(mode, model_name, level, out_dir)
        log(f"size={model_cfg['width']}x{model_cfg['height']} steps={model_cfg['steps']} "
            f"guidance={model_cfg['guidance']} seed={model_cfg.get('seed', 42)}")

        proc = None
        try:
            model_path = ensure_model_present(flat_path, model_cfg)
            if model_path is None:
                success, detail = False, "Model not found and could not be downloaded"
            elif not needs_server:
                success, detail = test_fn(log, url, model_name, model_cfg, model_path, out_dir, level)
            else:
                if args.all_models or (args.model_path and not server_is_up(url)):
                    proc = launch_server(model_path, port)
                    if proc is None:
                        success, detail = False, "Server failed to start"
                    else:
                        success, detail = test_fn(log, url, model_name, model_cfg, model_path, out_dir, level)
                else:
                    success, detail = test_fn(log, url, model_name, model_cfg, model_path, out_dir, level)
        except Exception as e:
            success, detail = False, f"Exception: {e}"
        finally:
            stop_server(proc)

        status = "PASS" if success else "FAIL"
        log(f"{status} — {detail}")
        log.close()
        results.append({"model": model_name, "mode": mode, "level": level, "success": success, "detail": detail})

    passed = sum(1 for r in results if r["success"])
    print(f"\n{'='*72}")
    print(f"  SUMMARY — {mode} [{level}]")
    print(f"{'='*72}")
    for r in results:
        icon = "PASS" if r["success"] else "FAIL"
        print(f"  [{icon}]  {r['model']}  {r['detail']}")
    print(f"\n  {passed}/{len(results)} passed")
    print(f"  Output: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": ts, "mode": mode, "level": level, "results": results}
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    sys.exit(0 if results and passed == len(results) else 1)
