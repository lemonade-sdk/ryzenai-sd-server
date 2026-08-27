#!/usr/bin/env python3
"""
Unified test runner for ryzenai-sd-server.

Supports txt2img, img2img, variations, controlnet, and CLI modes.
Model parameters are data-driven from models.json — no per-model if/else logic.

Usage:
    # Test a single model (server must be running):
    python test_server.py txt2img --url http://localhost:8080

    # Auto-launch server for a specific model:
    python test_server.py txt2img --model-path C:/models/stable-diffusion-turbo-amdnpu-onnx

    # Test all models (auto-launches server per model):
    python test_server.py txt2img --all-models

    # img2img (needs VAE encoder). All models use the SAME source input image
    # (test_server/img2img_test_input.png by default, or --input-image), each
    # resized to that model's own resolution — so results are comparable:
    python test_server.py img2img --all-models
    python test_server.py img2img --all-models --input-image C:/photos/reference.png

    # variations: same idea, but hits /v1/images/variations WITHOUT a prompt
    # (true OpenAI-style unprompted image variation, vs. img2img's prompt-
    # guided mode):
    python test_server.py variations --all-models

    # ControlNet (only runs on models that declare controlnet capability):
    python test_server.py controlnet --all-models
    python test_server.py controlnet --model-path C:/models/sd3-medium --types canny depth

    # CLI mode (no server, runs executable directly):
    python test_server.py cli --all-models
    python test_server.py cli --model-path C:/models/stable-diffusion-turbo-amdnpu-onnx
"""

import argparse
import base64
import json
import re
import subprocess
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from PIL import Image


# ─── Paths ───────────────────────────────────────────────────────────────────

import os

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_JSON = SCRIPT_DIR / "models.json"
DEFAULT_MODELS_DIR = Path(os.environ.get("RAI_SD_MODELS_DIR", r"C:\Users\mickraus\Work\rai_1.8.0_models"))
OUTPUT_DIR = PROJECT_ROOT / "test_outputs"


# ─── Config Loading ──────────────────────────────────────────────────────────

def load_config():
    """Load model configurations from models.json."""
    with open(MODELS_JSON) as f:
        return json.load(f)


def get_model_config(config, model_name, args=None):
    """Get merged config for a model (model-specific values override defaults).

    If `args` is given, --seed/--steps/--guidance/--prompt CLI overrides
    (when set) take precedence over both defaults and the model's own config.
    """
    defaults = config["defaults"].copy()
    model_cfg = config["models"].get(model_name)
    if model_cfg:
        defaults.update(model_cfg)
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


def discover_models(models_dir, mode, config):
    """Find models in directory (or HF cache) that support the given mode.

    Includes models that are:
    - Present as a flat directory under models_dir, OR
    - Registered in models.json with an hf_repo_id whose HF snapshot already exists
      (downloaded by Lemonade or a prior test run).
    """
    found_names: set = set()

    # Flat directories under models_dir
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                found_names.add(d.name)

    # Models registered in models.json that are in the HF cache
    for name, model_cfg in config["models"].items():
        if name in found_names:
            continue
        repo_id = model_cfg.get("hf_repo_id")
        if repo_id and get_hf_snapshot_path(repo_id) is not None:
            found_names.add(name)

    if not found_names:
        print(f"ERROR: No models found in {models_dir} or HF cache")
        sys.exit(1)

    # Filter to models that support this mode.
    # "variations" reuses the "img2img" capability tag in models.json — both
    # need the same vae_encoder; they only differ in whether a prompt is sent.
    required_capability = "img2img" if mode == "variations" else mode
    result = []
    for name in sorted(found_names):
        model_cfg = config["models"].get(name)
        if model_cfg is None:
            if mode in ("txt2img", "cli"):
                result.append(name)
        elif required_capability in model_cfg.get("capabilities", []):
            result.append(name)

    return result


# ─── Model Auto-Download ─────────────────────────────────────────────────────

def _hf_hub_cache() -> Path:
    """Return the HuggingFace hub cache root, matching Lemonade's C++ resolution order.

    Priority: HF_HUB_CACHE env var > HF_HOME/hub env var > ~/.cache/huggingface/hub
    This mirrors lemonade's resolve_hf_cache_dir() in path_utils.cpp.
    """
    import os
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
    # 1. Flat local path
    if flat_path.exists():
        return flat_path

    repo_id = model_cfg.get("hf_repo_id")

    # 2. HF cache (already downloaded, possibly by Lemonade)
    if repo_id:
        snapshot = get_hf_snapshot_path(repo_id)
        if snapshot is not None:
            print(f"  Found in HF cache: {snapshot}")
            return snapshot

    # 3. Download
    if not repo_id:
        print(f"  ERROR: Model not found at {flat_path} and no hf_repo_id defined")
        return None

    print(f"  Model not found locally — downloading {repo_id} from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
        # No local_dir → downloads to HF hub cache, same location Lemonade reads.
        # snapshot_download returns the snapshot directory path directly.
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
    """Start sd_npu_server and wait for /health. Returns subprocess.Popen."""
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
        # Last resort: treat as packed raw RGB using dimensions from the response
        w = response_json.get("width", 0)
        h = response_json.get("height", 0)
        if w <= 0 or h <= 0:
            raise ValueError(f"Cannot decode image: PIL failed and no valid dimensions in response (format={fmt})")
        expected = h * w * 3
        arr = np.frombuffer(raw_bytes[:expected], dtype=np.uint8).reshape(h, w, 3)
        img = Image.fromarray(arr, "RGB")

    img.save(str(output_path), "PNG")
    return output_path


# ─── Shared img2img/variations Input Image ──────────────────────────────────

DEFAULT_INPUT_IMAGE = SCRIPT_DIR / "img2img_test_input.png"


def resolve_shared_input_image(args) -> Path:
    """Resolve the ONE source image used for every model in an img2img/variations
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


# ─── Test Modes ──────────────────────────────────────────────────────────────

def test_txt2img(url, model_cfg, output_path):
    """Run a txt2img generation test. Returns (success, detail)."""
    w, h = model_cfg["width"], model_cfg["height"]
    prompt = model_cfg["prompt"]
    steps = model_cfg["steps"]
    guidance = model_cfg["guidance"]
    seed = model_cfg.get("seed", 42)

    # Build tagged prompt with extra args
    extra = json.dumps({"steps": steps, "cfg_scale": guidance, "seed": seed})
    tagged_prompt = f"{prompt} <sd_cpp_extra_args>{extra}</sd_cpp_extra_args>"

    body = {
        "prompt": tagged_prompt,
        "n": 1,
        "size": f"{w}x{h}",
        "response_format": "b64_json",
    }

    try:
        resp = requests.post(f"{url}/v1/images/generations", json=body, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        save_response_image(result, output_path)
        return True, str(output_path)
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {url}"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return False, str(e)


def test_img2img(url, model_cfg, model_path, output_path, ref_image_path):
    """Run a prompt-guided img2img generation test (POSTs the model's prompt
    alongside the image to /v1/images/variations). Returns (success, detail).

    `ref_image_path` is the ONE shared source image used for every model in
    this run (see resolve_shared_input_image) — resized here to this model's
    own resolution so different-resolution models remain comparable.
    """
    w, h = model_cfg["width"], model_cfg["height"]
    vae_encoder = model_cfg.get("vae_encoder")
    if not vae_encoder:
        return False, "No vae_encoder defined for this model"

    vae_path = Path(model_path) / vae_encoder
    if not vae_path.exists():
        return False, f"VAE encoder not found: {vae_path}"

    # Load, resize, encode as raw RGB base64
    ref_img = Image.open(ref_image_path).convert("RGB").resize((w, h), Image.LANCZOS)
    image_b64 = base64.b64encode(np.array(ref_img).tobytes()).decode()

    try:
        resp = requests.post(
            f"{url}/v1/images/variations",
            files={"image[]": ("image.bin", image_b64.encode())},
            data={
                "size": f"{w}x{h}",
                "n": "1",
                "prompt": model_cfg["prompt"],
                "num_inference_steps": str(model_cfg["steps"]),
                "guidance_scale": str(model_cfg["guidance"]),
                "strength": str(model_cfg.get("strength", 0.5)),
                "seed": str(model_cfg.get("seed", 42)),
            },
            timeout=600,
        )
        resp.raise_for_status()
        result = resp.json()
        save_response_image(result, output_path)
        return True, str(output_path)
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {url}"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return False, str(e)


def test_variations(url, model_cfg, model_path, output_path, ref_image_path):
    """Run a true OpenAI-style image-variations test: same endpoint as
    img2img (/v1/images/variations) but with NO prompt sent, so the server
    falls back to its unprompted "image variation" default. Returns
    (success, detail).

    `ref_image_path` is the ONE shared source image used for every model in
    this run (see resolve_shared_input_image).
    """
    w, h = model_cfg["width"], model_cfg["height"]
    vae_encoder = model_cfg.get("vae_encoder")
    if not vae_encoder:
        return False, "No vae_encoder defined for this model"

    vae_path = Path(model_path) / vae_encoder
    if not vae_path.exists():
        return False, f"VAE encoder not found: {vae_path}"

    ref_img = Image.open(ref_image_path).convert("RGB").resize((w, h), Image.LANCZOS)
    image_b64 = base64.b64encode(np.array(ref_img).tobytes()).decode()

    try:
        resp = requests.post(
            f"{url}/v1/images/variations",
            files={"image[]": ("image.bin", image_b64.encode())},
            data={
                # Deliberately NO "prompt" field — true unprompted variations.
                "size": f"{w}x{h}",
                "n": "1",
                "num_inference_steps": str(model_cfg["steps"]),
                "guidance_scale": str(model_cfg["guidance"]),
                "strength": str(model_cfg.get("strength", 0.5)),
                "seed": str(model_cfg.get("seed", 42)),
            },
            timeout=600,
        )
        resp.raise_for_status()
        result = resp.json()
        save_response_image(result, output_path)
        return True, str(output_path)
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {url}"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return False, str(e)


def test_controlnet(url, model_cfg, cn_type, cn_config, output_path):
    """Run a controlnet generation test. Returns (success, detail)."""
    w, h = model_cfg["width"], model_cfg["height"]
    image_path = SCRIPT_DIR / cn_config["image"]
    if not image_path.exists():
        return False, f"Control image not found: {image_path}"

    # Encode control image as raw RGB base64
    img = Image.open(image_path).convert("RGB").resize((w, h), Image.LANCZOS)
    control_b64 = base64.b64encode(np.array(img).tobytes()).decode()

    # Build tagged prompt
    extra = json.dumps({
        "steps": cn_config["steps"],
        "cfg_scale": cn_config["guidance"],
        "seed": cn_config["seed"],
    })
    prompt = f"{cn_config['prompt']} <sd_cpp_extra_args>{extra}</sd_cpp_extra_args>"

    try:
        resp = requests.post(
            f"{url}/v1/images/edits",
            files={"image[]": ("control.rgb", control_b64.encode("ascii"), "text/plain")},
            data={"prompt": prompt, "size": f"{w}x{h}", "n": "1"},
            timeout=600,
        )
        resp.raise_for_status()
        result = resp.json()
        save_response_image(result, output_path)
        return True, str(output_path)
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {url}"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return False, str(e)


def test_cli(model_cfg, model_path, output_path):
    """Run a CLI (one-shot, no server) test. Returns (success, detail)."""
    exe = find_server_exe()
    if not exe:
        return False, "ryzenai-sd-server executable not found"

    w, h = model_cfg["width"], model_cfg["height"]
    cmd = [
        str(exe),
        "--model-path", str(model_path),
        "--width", str(w),
        "--height", str(h),
        "--num-inference-steps", str(model_cfg["steps"]),
        "--guidance-scale", str(model_cfg["guidance"]),
        "--prompt", model_cfg["prompt"],
        "--output", str(output_path),
        "--seed", str(model_cfg.get("seed", 42)),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and Path(output_path).exists():
            # Extract timing if available
            gen_match = re.search(r"\[TIMING\] Total generation: ([\d.]+) seconds", result.stdout)
            detail = str(output_path)
            if gen_match:
                detail += f" ({float(gen_match.group(1)):.1f}s)"
            return True, detail
        else:
            error = result.stderr[:300] if result.stderr else f"exit code {result.returncode}"
            return False, error
    except subprocess.TimeoutExpired:
        return False, "Timeout (5 min)"
    except Exception as e:
        return False, str(e)


# ─── Main Runner ─────────────────────────────────────────────────────────────

def _dry_run(args, config, mode):
    """Print what would be tested without launching servers or requiring model files."""
    if args.all_models:
        models_dir = Path(args.models_dir)
        model_names = discover_models(models_dir, mode, config)
    elif args.model_path:
        model_names = [Path(args.model_path).name]
    else:
        model_names = [args.model_name or "unknown"]

    print(f"Dry run — mode={mode}  models-dir={args.models_dir}")
    print(f"Would test {len(model_names)} model(s):\n")

    for model_name in model_names:
        model_cfg = get_model_config(config, model_name, args)
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

        if mode == "controlnet":
            cn_types = args.types or model_cfg.get("controlnet_types", [])
            if not cn_types:
                print(f"    controlnet: NONE declared")
            for cn_type in cn_types:
                cn_cfg = config["controlnet"].get(cn_type)
                if not cn_cfg:
                    print(f"    controlnet[{cn_type}]: UNKNOWN TYPE")
                    continue
                image = args.control_image or cn_cfg["image"]
                image_path = SCRIPT_DIR / image
                exists = "OK" if image_path.exists() else "MISSING"
                print(f"    controlnet[{cn_type}]: guidance={cn_cfg['guidance']} steps={cn_cfg['steps']} "
                      f"conditioning_scale={cn_cfg['conditioning_scale']}  image={image} ({exists})")
        print()


def run_tests(args):
    """Execute tests based on parsed arguments."""
    config = load_config()
    mode = args.mode
    port = args.port
    url = args.url or f"http://localhost:{port}"

    if args.dry_run:
        _dry_run(args, config, mode)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / mode / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to test
    if args.all_models:
        models_dir = Path(args.models_dir)
        model_names = discover_models(models_dir, mode, config)
        print(f"Discovered {len(model_names)} models for '{mode}' mode:")
        for m in model_names:
            print(f"  - {m}")
    elif args.model_path:
        model_names = [Path(args.model_path).name]
    else:
        # Assume server already running, use a placeholder name
        model_names = [args.model_name or "unknown"]

    # img2img/variations: resolve ONE shared source image up front, before the
    # per-model loop, so every model tests against the exact same picture.
    ref_image_path = None
    if mode in ("img2img", "variations"):
        ref_image_path = resolve_shared_input_image(args)
        print(f"Using shared input image for all models: {ref_image_path}")

    results = []

    for model_name in model_names:
        model_cfg = get_model_config(config, model_name, args)
        # flat_path is used as the first lookup; the resolved path may differ (HF cache)
        flat_path = Path(args.models_dir) / model_name if args.all_models else Path(args.model_path or "")

        print(f"\n{'='*72}")
        print(f"  {model_name}  [{mode}]")
        print(f"  {model_cfg['width']}x{model_cfg['height']}  steps={model_cfg['steps']}  guidance={model_cfg['guidance']}")
        print(f"{'='*72}")

        # Resolve the actual model path (flat dir, HF cache snapshot, or download)
        model_path = ensure_model_present(flat_path, model_cfg)
        if model_path is None:
            results.append({"model": model_name, "mode": mode, "success": False, "detail": "Model not found and could not be downloaded"})
            continue

        # For server modes: launch server if needed
        proc = None
        if mode != "cli":
            if args.all_models or (args.model_path and not server_is_up(url)):
                proc = launch_server(model_path, port)
                if proc is None:
                    results.append({"model": model_name, "mode": mode, "success": False, "detail": "Server failed to start"})
                    print(f"  [FAIL] Server failed to start — skipping")
                    continue

        try:
            if mode == "txt2img":
                output = out_dir / f"{model_name}.png"
                success, detail = test_txt2img(url, model_cfg, output)

            elif mode == "img2img":
                output = out_dir / f"{model_name}.png"
                success, detail = test_img2img(url, model_cfg, model_path, output, ref_image_path)

            elif mode == "variations":
                output = out_dir / f"{model_name}.png"
                success, detail = test_variations(url, model_cfg, model_path, output, ref_image_path)

            elif mode == "controlnet":
                # Run all controlnet types for this model
                cn_types = args.types or model_cfg.get("controlnet_types", [])
                if not cn_types:
                    success, detail = False, "No controlnet types defined"
                else:
                    cn_results = []
                    for cn_type in cn_types:
                        cn_cfg = config["controlnet"].get(cn_type)
                        if not cn_cfg:
                            cn_results.append((cn_type, False, f"Unknown type: {cn_type}"))
                            continue
                        # Allow CLI override of control image
                        if args.control_image:
                            cn_cfg = cn_cfg.copy()
                            cn_cfg["image"] = args.control_image
                        # Need to relaunch server with -C flag for each type
                        if proc:
                            stop_server(proc)
                        proc = launch_server(model_path, port, extra_args=["-C", cn_type])
                        cn_out = out_dir / f"{model_name}_{cn_type}.png"
                        s, d = test_controlnet(url, model_cfg, cn_type, cn_cfg, cn_out)
                        cn_results.append((cn_type, s, d))
                        print(f"    {cn_type}: {'PASS' if s else 'FAIL'} — {d}")

                    passed = sum(1 for _, s, _ in cn_results if s)
                    success = passed == len(cn_results)
                    detail = f"{passed}/{len(cn_results)} types passed"

            elif mode == "cli":
                output = out_dir / f"{model_name}.png"
                success, detail = test_cli(model_cfg, model_path, output)

            else:
                success, detail = False, f"Unknown mode: {mode}"

        finally:
            stop_server(proc)

        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {detail}")
        results.append({"model": model_name, "mode": mode, "success": success, "detail": detail})

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SUMMARY — {mode}")
    print(f"{'='*72}")
    passed = sum(1 for r in results if r["success"])
    for r in results:
        icon = "PASS" if r["success"] else "FAIL"
        print(f"  [{icon}]  {r['model']}  {r['detail']}")
    print(f"\n  {passed}/{len(results)} passed")
    print(f"  Output: {out_dir}")

    # Save report
    report = {"timestamp": ts, "mode": mode, "results": results}
    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    sys.exit(0 if passed == len(results) else 1)


def main():
    parser = argparse.ArgumentParser(
        description="""Unified sd_npu_server test runner.

Supports txt2img, img2img, variations, controlnet, and CLI modes.
Model parameters are data-driven from models.json — no per-model if/else logic.

Modes:
  txt2img      Generate an image from a text prompt via the server API.
  img2img      Prompt-guided image-to-image via the server API (requires VAE
               encoder). All models tested use the SAME source input image
               (see --input-image), each resized to its own resolution.
  variations   Unprompted image variations (same endpoint as img2img, but NO
               prompt sent) — true OpenAI-style /v1/images/variations. Also
               uses the SAME shared source image across all models.
  controlnet   Generate with ControlNet guidance (canny, pose, tile, depth).
  cli          Run the executable directly (one-shot, no server).

Model selection (pick one):
  --url URL          Test against a server that is already running.
  --model-path PATH  Auto-launch the server with this model, then test.
  --all-models       Discover all models in --models-dir and test each one.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test against a running server:
  python test_server.py txt2img --url http://localhost:8080

  # Auto-launch server for a specific model:
  python test_server.py txt2img --model-path C:/models/stable-diffusion-turbo-amdnpu-onnx

  # Test all models (auto-launches server per model):
  python test_server.py txt2img --all-models

  # img2img for all models that have a VAE encoder (all use the same source
  # image, so results are directly comparable):
  python test_server.py img2img --all-models
  python test_server.py img2img --all-models --input-image C:/photos/reference.png

  # Unprompted image variations (no text prompt sent):
  python test_server.py variations --all-models

  # ControlNet — only specific types:
  python test_server.py controlnet --all-models --types canny depth

  # CLI mode (no server, runs executable directly):
  python test_server.py cli --model-path C:/models/stable-diffusion-turbo-amdnpu-onnx
  python test_server.py cli --all-models

Notes:
  - Model configs (resolution, steps, guidance, capabilities) live in models.json.
  - To add a new model, add an entry to models.json — no code changes needed.
  - Unknown models get safe defaults for txt2img and cli modes.
  - ControlNet only runs on models that declare "controlnet" in capabilities.
""",
    )

    parser.add_argument(
        "mode",
        choices=["txt2img", "img2img", "variations", "controlnet", "cli"],
        help="Test mode to run (required). One of: txt2img, img2img, variations, controlnet, cli. "
             "Example: python test_server.py txt2img --all-models",
    )

    # Model selection (mutually exclusive approaches)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-path", help="Path to a specific model directory")
    group.add_argument("--all-models", action="store_true", help="Test all discovered models")

    # Connection / paths
    parser.add_argument("--url", help="Server URL (skip auto-launch, test against running server)")
    parser.add_argument("--port", type=int, default=8080, help="Port for auto-launched server (default: 8080)")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Directory containing model folders")
    parser.add_argument("--model-name", help="Model name (used with --url when server is already running)")

    # Overrides
    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument("--steps", type=int, help="Override inference steps")
    parser.add_argument("--guidance", type=float, help="Override guidance scale")
    parser.add_argument("--prompt", help="Override prompt")

    # ControlNet-specific
    parser.add_argument("--types", nargs="+", help="ControlNet types to test (default: all for model)")
    parser.add_argument("--control-image", help="Override control image path (applies to all types being tested)")

    # img2img/variations-specific
    parser.add_argument(
        "--input-image",
        help="Source image for img2img/variations modes (applies to ALL models tested, each resized to its "
             "own resolution). Default: test/img2img_test_input.png (auto-generated once if missing).",
    )

    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and show what would be tested without launching servers or requiring model files.",
    )

    args = parser.parse_args()

    # Validate args
    if not args.dry_run:
        if args.mode != "cli" and not args.url and not args.model_path and not args.all_models:
            parser.error(f"For '{args.mode}' mode, provide --url, --model-path, or --all-models")
        if args.mode == "cli" and not args.model_path and not args.all_models:
            parser.error("For 'cli' mode, provide --model-path or --all-models")

    run_tests(args)


if __name__ == "__main__":
    main()
