#!/usr/bin/env python3
"""
CLI test for ryzenai-sd-server: one-shot invocation of the executable
directly (no server, no HTTP).

Usage:
    # Test a single model:
    python test_cli.py --model-path C:/models/sdxl-turbo-amdnpu-onnx

    # Full quality run instead of the default fast smoke check:
    python test_cli.py --model-path C:/models/sdxl-turbo-amdnpu-onnx --level detailed

    # Test all discovered models:
    python test_cli.py --all-models
"""

import re
import subprocess
from pathlib import Path

import sd_test_common as common

MODE = "cli"


def run_test(log, url, model_name, model_cfg, model_path, out_dir, level):
    """Run a CLI (one-shot, no server) test. Returns (success, detail)."""
    exe = common.find_server_exe()
    if not exe:
        return False, "ryzenai-sd-server executable not found"

    w, h = model_cfg["width"], model_cfg["height"]
    steps = model_cfg["steps"]
    guidance = model_cfg["guidance"]
    seed = model_cfg.get("seed", 42)
    output_path = out_dir / f"{MODE}_{model_name}_{level}.png"

    cmd = [
        str(exe),
        "--model-path", str(model_path),
        "--width", str(w),
        "--height", str(h),
        "--num-inference-steps", str(steps),
        "--guidance-scale", str(guidance),
        "--prompt", model_cfg["prompt"],
        "--output", str(output_path),
        "--seed", str(seed),
    ]
    log(f"exec size={w}x{h} steps={steps} guidance={guidance} seed={seed}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and output_path.exists():
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


def main():
    parser = common.build_common_parser(
        "CLI test — one-shot invocation of the executable directly (no server)."
    )
    args = parser.parse_args()

    if not args.dry_run and not args.model_path and not args.all_models:
        parser.error("Provide --model-path or --all-models")

    config = common.load_config()
    common.run_per_model_loop(MODE, args, config, capability="cli", test_fn=run_test, needs_server=False)


if __name__ == "__main__":
    main()
