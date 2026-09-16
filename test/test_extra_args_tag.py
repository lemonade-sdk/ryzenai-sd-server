#!/usr/bin/env python3
"""
sd_cpp_extra_args tag test for ryzenai-sd-server: verifies steps/cfg_scale/
seed/strength embedded ONLY in the <sd_cpp_extra_args>{...}</sd_cpp_extra_args>
prompt tag (no plain multipart fields at all) are correctly parsed and applied
on both /v1/images/edits and /v1/images/variations. This is the exact
convention Lemonade uses end-to-end (see RyzenAISDServer::image_edits() /
image_variations() in the lemonade repo) -- it never sends plain
num_inference_steps/guidance_scale/seed/strength fields, only the tag.

Every model tested uses the SAME shared source image (see --input-image),
each resized to its own resolution, so results are directly comparable
across models.

Usage:
    # Test a single model (auto-launches server):
    python test_extra_args_tag.py --model-path C:/models/sd-turbo-amdnpu-onnx

    # Full quality run instead of the default fast smoke check:
    python test_extra_args_tag.py --model-path C:/models/sd-turbo-amdnpu-onnx --level detailed

    # Test all discovered models against a custom source image:
    python test_extra_args_tag.py --all-models --input-image C:/photos/reference.png
"""

import base64
import json
from functools import partial
from pathlib import Path

import numpy as np
import requests
from PIL import Image

import sd_test_common as common

MODE = "extra_args_tag"


def _tagged_prompt(prompt, steps, cfg_scale, seed, strength):
    """Build a prompt with every generation parameter embedded in the tag and
    NOT sent as any plain multipart field, matching Lemonade's calling
    convention exactly."""
    extra_args = {
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "strength": strength,
    }
    return f"{prompt} <sd_cpp_extra_args>{json.dumps(extra_args)}</sd_cpp_extra_args>"


def _post_tagged(url, endpoint, image_b64, w, h, prompt, mask_b64=None):
    """POST to `endpoint` with prompt/size/n only -- no steps/cfg_scale/seed/
    strength plain fields, so the tag is the only channel that can supply
    them."""
    files = {"image[]": ("image.bin", image_b64.encode())}
    if mask_b64 is not None:
        files["mask"] = ("mask.bin", mask_b64.encode())
    return requests.post(
        f"{url}{endpoint}",
        files=files,
        data={
            "size": f"{w}x{h}",
            "n": "1",
            "prompt": prompt,
        },
        timeout=600,
    )


def run_test(ref_image_path, log, url, model_name, model_cfg, model_path, out_dir, level):
    """Run both /v1/images/edits and /v1/images/variations with every
    parameter carried exclusively by the <sd_cpp_extra_args> tag. Returns
    (success, detail)."""
    w, h = model_cfg["width"], model_cfg["height"]
    vae_encoder = model_cfg.get("vae_encoder")
    if not vae_encoder:
        return False, "No vae_encoder defined for this model"

    vae_path = Path(model_path) / vae_encoder
    if not vae_path.exists():
        return False, f"VAE encoder not found: {vae_path}"

    steps = model_cfg["steps"]
    guidance = model_cfg["guidance"]
    strength = model_cfg.get("strength", 0.5)
    seed = model_cfg.get("seed", 42)

    ref_img = Image.open(ref_image_path).convert("RGB").resize((w, h), Image.LANCZOS)
    image_b64 = base64.b64encode(np.array(ref_img).tobytes()).decode()
    prompt = _tagged_prompt(model_cfg["prompt"], steps, guidance, seed, strength)

    log(f"POST /v1/images/edits (tag-only) size={w}x{h} steps={steps} guidance={guidance} "
        f"strength={strength} seed={seed}")
    try:
        resp = _post_tagged(url, "/v1/images/edits", image_b64, w, h, prompt)
        resp.raise_for_status()
        result = resp.json()
        common.save_response_image(result, out_dir / f"{MODE}_edits_{model_name}_{level}.png")
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {url}"
    except requests.exceptions.HTTPError as e:
        return False, f"edits: HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return False, f"edits: {e}"

    log(f"POST /v1/images/variations (tag-only) size={w}x{h} steps={steps} guidance={guidance} "
        f"strength={strength} seed={seed}")
    try:
        resp = _post_tagged(url, "/v1/images/variations", image_b64, w, h, prompt)
        resp.raise_for_status()
        result = resp.json()
        output_path = out_dir / f"{MODE}_variations_{model_name}_{level}.png"
        common.save_response_image(result, output_path)
        return True, str(output_path)
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {url}"
    except requests.exceptions.HTTPError as e:
        return False, f"variations: HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return False, f"variations: {e}"


def main():
    parser = common.build_common_parser(
        "sd_cpp_extra_args tag test — steps/cfg_scale/seed/strength carried "
        "exclusively by the prompt tag (no plain fields) on both "
        "/v1/images/edits and /v1/images/variations, matching Lemonade's "
        "calling convention."
    )
    parser.add_argument(
        "--input-image",
        help="Source image for this run (applies to ALL models tested, each resized to its "
             "own resolution). Default: test/img2img_test_input.png (auto-generated once if missing).",
    )
    args = parser.parse_args()
    common.validate_model_selection(parser, args)

    config = common.load_config()

    if args.dry_run:
        common.run_per_model_loop(MODE, args, config, capability="img2img", test_fn=None)
        return

    ref_image_path = common.resolve_shared_input_image(args)
    print(f"Using shared input image for all models: {ref_image_path}")

    common.run_per_model_loop(
        MODE, args, config, capability="img2img",
        test_fn=partial(run_test, ref_image_path),
    )


if __name__ == "__main__":
    main()
