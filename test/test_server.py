#!/usr/bin/env python3
"""
Unified test runner for ryzenai-sd-server.

Supports txt2img, img2img, controlnet, and CLI modes.
Model parameters are data-driven from models.json — no per-model if/else logic.

Usage:
    # Test a single model (server must be running):
    python run_tests.py txt2img --url http://localhost:8080

    # Auto-launch server for a specific model:
    python run_tests.py txt2img --model-path C:/models/stable-diffusion-turbo-amdnpu-onnx

    # Test all models (auto-launches server per model):
    python run_tests.py txt2img --all-models

    # img2img (needs VAE encoder):
    python run_tests.py img2img --all-models

    # ControlNet (only runs on models that declare controlnet capability):
    python run_tests.py controlnet --all-models
    python run_tests.py controlnet --model-path C:/models/sd3-medium --types canny depth

    # CLI mode (no server, runs executable directly):
    python run_tests.py cli --all-models
    python run_tests.py cli --model-path C:/models/stable-diffusion-turbo-amdnpu-onnx
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_JSON = SCRIPT_DIR / "models.json"
DEFAULT_MODELS_DIR = Path(r"C:\Users\mickraus\sd_models\1.7.1_models")
OUTPUT_DIR = PROJECT_ROOT / "test_outputs"


# ─── Config Loading ──────────────────────────────────────────────────────────

def load_config():
    """Load model configurations from models.json."""
    with open(MODELS_JSON) as f:
        return json.load(f)


def get_model_config(config, model_name):
    """Get merged config for a model (model-specific values override defaults)."""
    defaults = config["defaults"].copy()
    model_cfg = config["models"].get(model_name)
    if model_cfg:
        defaults.update(model_cfg)
    return defaults


def discover_models(models_dir, mode, config):
    """Find models in directory that support the given mode."""
    if not models_dir.exists():
        print(f"ERROR: Models directory not found: {models_dir}")
        sys.exit(1)

    available = sorted(d.name for d in models_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not available:
        print(f"ERROR: No model directories found in {models_dir}")
        sys.exit(1)

    # Filter to models that support this mode
    result = []
    for name in available:
        model_cfg = config["models"].get(name)
        if model_cfg is None:
            # Unknown model — include it for txt2img/cli (safe defaults), skip for specialized modes
            if mode in ("txt2img", "cli"):
                result.append(name)
        elif mode in model_cfg.get("capabilities", []):
            result.append(name)

    return result


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
            sys.exit(1)
        time.sleep(2)

    proc.terminate()
    print(f"  ERROR: Server did not become ready within {timeout}s")
    sys.exit(1)


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


def test_img2img(url, model_cfg, model_path, output_path):
    """Run an img2img generation test. Returns (success, detail)."""
    w, h = model_cfg["width"], model_cfg["height"]
    vae_encoder = model_cfg.get("vae_encoder")
    if not vae_encoder:
        return False, "No vae_encoder defined for this model"

    vae_path = Path(model_path) / vae_encoder
    if not vae_path.exists():
        return False, f"VAE encoder not found: {vae_path}"

    # Use the test input image (or generate a synthetic one)
    input_image = SCRIPT_DIR / "img2img_test_input.png"
    if not input_image.exists():
        # Create a simple synthetic input
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(input_image))

    # Load, resize, encode as raw RGB base64
    ref_img = Image.open(input_image).convert("RGB").resize((w, h), Image.LANCZOS)
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

    results = []

    for model_name in model_names:
        model_cfg = get_model_config(config, model_name)
        model_path = Path(args.models_dir) / model_name if args.all_models else Path(args.model_path or "")

        print(f"\n{'='*72}")
        print(f"  {model_name}  [{mode}]")
        print(f"  {model_cfg['width']}x{model_cfg['height']}  steps={model_cfg['steps']}  guidance={model_cfg['guidance']}")
        print(f"{'='*72}")

        # For server modes: launch server if needed
        proc = None
        if mode != "cli":
            if args.all_models or (args.model_path and not server_is_up(url)):
                proc = launch_server(model_path, port)

        try:
            if mode == "txt2img":
                output = out_dir / f"{model_name}.png"
                success, detail = test_txt2img(url, model_cfg, output)

            elif mode == "img2img":
                output = out_dir / f"{model_name}.png"
                success, detail = test_img2img(url, model_cfg, model_path, output)

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

Supports txt2img, img2img, controlnet, and CLI modes.
Model parameters are data-driven from models.json — no per-model if/else logic.

Modes:
  txt2img      Generate an image from a text prompt via the server API.
  img2img      Transform an input image via the server API (requires VAE encoder).
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
  python run_tests.py txt2img --url http://localhost:8080

  # Auto-launch server for a specific model:
  python run_tests.py txt2img --model-path C:/models/stable-diffusion-turbo-amdnpu-onnx

  # Test all models (auto-launches server per model):
  python run_tests.py txt2img --all-models

  # img2img for all models that have a VAE encoder:
  python run_tests.py img2img --all-models

  # ControlNet — only specific types:
  python run_tests.py controlnet --all-models --types canny depth

  # CLI mode (no server, runs executable directly):
  python run_tests.py cli --model-path C:/models/stable-diffusion-turbo-amdnpu-onnx
  python run_tests.py cli --all-models

Notes:
  - Model configs (resolution, steps, guidance, capabilities) live in models.json.
  - To add a new model, add an entry to models.json — no code changes needed.
  - Unknown models get safe defaults for txt2img and cli modes.
  - ControlNet only runs on models that declare "controlnet" in capabilities.
""",
    )

    parser.add_argument(
        "mode",
        choices=["txt2img", "img2img", "controlnet", "cli"],
        help="Test mode to run (required). One of: txt2img, img2img, controlnet, cli. "
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
