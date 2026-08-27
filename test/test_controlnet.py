#!/usr/bin/env python3
"""
ControlNet test for ryzenai-sd-server: generate with ControlNet guidance
(canny, pose, tile, depth) via /v1/images/edits. Only runs on models that
declare "controlnet" in their models.json capabilities.

Each ControlNet type requires relaunching the server with a `-C <type>` flag,
so (unlike the other per-mode scripts) this one manages its own server
lifecycle per type instead of using the shared single-launch runner loop.

Usage:
    # Test a single model, all its declared controlnet types:
    python test_controlnet.py --model-path C:/models/stable-diffusion-3-medium-amdnpu-onnx

    # Only specific types:
    python test_controlnet.py --model-path C:/models/sd3-medium --types canny depth

    # Full quality run instead of the default fast smoke check:
    python test_controlnet.py --model-path C:/models/sd3-medium --level detailed

    # Test all discovered controlnet-capable models:
    python test_controlnet.py --all-models
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

import sd_test_common as common

MODE = "controlnet"


def run_type(log, url, model_cfg, cn_type, cn_config, output_path):
    """Run a single ControlNet type generation. Returns (success, detail)."""
    w, h = model_cfg["width"], model_cfg["height"]
    image_path = common.SCRIPT_DIR / cn_config["image"]
    if not image_path.exists():
        return False, f"Control image not found: {image_path}"

    img = Image.open(image_path).convert("RGB").resize((w, h), Image.LANCZOS)
    control_b64 = base64.b64encode(np.array(img).tobytes()).decode()

    extra = json.dumps({
        "steps": cn_config["steps"],
        "cfg_scale": cn_config["guidance"],
        "seed": cn_config["seed"],
    })
    prompt = f"{cn_config['prompt']} <sd_cpp_extra_args>{extra}</sd_cpp_extra_args>"
    log(f"[{cn_type}] POST /v1/images/edits size={w}x{h} steps={cn_config['steps']} "
        f"guidance={cn_config['guidance']} seed={cn_config['seed']}")

    try:
        resp = requests.post(
            f"{url}/v1/images/edits",
            files={"image[]": ("control.rgb", control_b64.encode("ascii"), "text/plain")},
            data={"prompt": prompt, "size": f"{w}x{h}", "n": "1"},
            timeout=600,
        )
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


def get_cn_config(config, cn_type, cn_config_override, level):
    """Merge models.json's controlnet[type] config with the requested level."""
    cn_cfg = config["controlnet"].get(cn_type)
    if not cn_cfg:
        return None
    cn_cfg = cn_cfg.copy()
    if cn_config_override:
        cn_cfg["image"] = cn_config_override
    if level == "smoke":
        cn_cfg["steps"] = common.SMOKE_STEPS
    return cn_cfg


def test_model(args, config, model_name, out_dir):
    """Run all requested ControlNet types for one model. Returns list of result dicts."""
    level = args.level
    port = args.port
    url = args.url or f"http://localhost:{port}"

    model_cfg = common.get_model_config(config, model_name, args, level=level)
    flat_path = Path(args.models_dir) / model_name if args.all_models else Path(args.model_path or "")

    results = []
    log = common.make_logger(MODE, model_name, level, out_dir)

    model_path = common.ensure_model_present(flat_path, model_cfg)
    if model_path is None:
        log("FAIL — Model not found and could not be downloaded")
        log.close()
        return [{"model": model_name, "mode": MODE, "level": level, "success": False,
                  "detail": "Model not found and could not be downloaded"}]

    cn_types = args.types or model_cfg.get("controlnet_types", [])
    if not cn_types:
        log("FAIL — No controlnet types defined")
        log.close()
        return [{"model": model_name, "mode": MODE, "level": level, "success": False,
                  "detail": "No controlnet types defined"}]

    proc = None
    try:
        for cn_type in cn_types:
            cn_cfg = get_cn_config(config, cn_type, args.control_image, level)
            if not cn_cfg:
                log(f"[{cn_type}] FAIL — Unknown controlnet type")
                results.append({"model": model_name, "mode": MODE, "level": level, "success": False,
                                  "detail": f"Unknown type: {cn_type}"})
                continue

            if proc:
                common.stop_server(proc)
            proc = common.launch_server(model_path, port, extra_args=["-C", cn_type])
            if proc is None:
                log(f"[{cn_type}] FAIL — Server failed to start")
                results.append({"model": model_name, "mode": MODE, "level": level, "success": False,
                                  "detail": "Server failed to start"})
                continue

            output_path = out_dir / f"{MODE}_{model_name}_{cn_type}_{level}.png"
            success, detail = run_type(log, url, model_cfg, cn_type, cn_cfg, output_path)
            log(f"[{cn_type}] {'PASS' if success else 'FAIL'} — {detail}")
            results.append({"model": model_name, "mode": MODE, "level": level, "success": success,
                              "detail": f"[{cn_type}] {detail}"})
    except Exception as e:
        log(f"FAIL — Exception: {e}")
        results.append({"model": model_name, "mode": MODE, "level": level, "success": False,
                          "detail": f"Exception: {e}"})
    finally:
        common.stop_server(proc)
        log.close()

    return results


def main():
    parser = common.build_common_parser(
        "ControlNet test — generate with ControlNet guidance via /v1/images/edits."
    )
    parser.add_argument("--types", nargs="+", help="ControlNet types to test (default: all declared for model)")
    parser.add_argument("--control-image", help="Override control image path (applies to all types tested)")
    args = parser.parse_args()
    common.validate_model_selection(parser, args)

    config = common.load_config()

    if args.dry_run:
        common.run_per_model_loop(MODE, args, config, capability="controlnet", test_fn=None)
        return

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = common.OUTPUT_DIR / MODE / ts

    model_names = common.resolve_model_names(args, config, "controlnet")

    all_results = []
    for model_name in model_names:
        all_results.extend(test_model(args, config, model_name, out_dir))

    passed = sum(1 for r in all_results if r["success"])
    print(f"\n{'='*72}")
    print(f"  SUMMARY — {MODE} [{args.level}]")
    print(f"{'='*72}")
    for r in all_results:
        icon = "PASS" if r["success"] else "FAIL"
        print(f"  [{icon}]  {r['model']}  {r['detail']}")
    print(f"\n  {passed}/{len(all_results)} passed")
    print(f"  Output: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": ts, "mode": MODE, "level": args.level, "results": all_results}
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    sys.exit(0 if all_results and passed == len(all_results) else 1)


if __name__ == "__main__":
    main()
