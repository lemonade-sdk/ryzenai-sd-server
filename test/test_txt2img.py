#!/usr/bin/env python3
"""
txt2img test for ryzenai-sd-server: generate an image from a text prompt via
the /v1/images/generations endpoint.

Usage:
    # Test a single model (auto-launches server):
    python test_txt2img.py --model-path C:/models/sdxl-turbo-amdnpu-onnx

    # Full quality run instead of the default fast smoke check:
    python test_txt2img.py --model-path C:/models/sdxl-turbo-amdnpu-onnx --level detailed

    # Test all discovered models:
    python test_txt2img.py --all-models

    # Against a server that's already running:
    python test_txt2img.py --url http://localhost:8080 --model-name sdxl-turbo-amdnpu-onnx
"""

import json

import requests

import sd_test_common as common

MODE = "txt2img"


def run_test(log, url, model_name, model_cfg, model_path, out_dir, level):
    """Run a txt2img generation test. Returns (success, detail)."""
    w, h = model_cfg["width"], model_cfg["height"]
    prompt = model_cfg["prompt"]
    steps = model_cfg["steps"]
    guidance = model_cfg["guidance"]
    seed = model_cfg.get("seed", 42)

    extra = json.dumps({"steps": steps, "cfg_scale": guidance, "seed": seed})
    tagged_prompt = f"{prompt} <sd_cpp_extra_args>{extra}</sd_cpp_extra_args>"

    body = {
        "prompt": tagged_prompt,
        "n": 1,
        "size": f"{w}x{h}",
        "response_format": "b64_json",
    }
    log(f"POST /v1/images/generations size={w}x{h} steps={steps} guidance={guidance} seed={seed}")

    output_path = out_dir / f"{MODE}_{model_name}_{level}.png"
    try:
        resp = requests.post(f"{url}/v1/images/generations", json=body, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        common.save_response_image(result, output_path)
        return True, str(output_path)
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {url}"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return False, str(e)


def main():
    parser = common.build_common_parser(
        "txt2img test — generate an image from a text prompt via /v1/images/generations."
    )
    args = parser.parse_args()
    common.validate_model_selection(parser, args)

    config = common.load_config()
    common.run_per_model_loop(MODE, args, config, capability="txt2img", test_fn=run_test)


if __name__ == "__main__":
    main()
