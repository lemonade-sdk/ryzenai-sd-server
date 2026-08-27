#!/usr/bin/env python3
"""
variants test for ryzenai-sd-server: true OpenAI-style unprompted image
variations. Same endpoint as img2img (/v1/images/variations) but with NO
prompt sent, so the server falls back to its unprompted "image variation"
default. Every model tested uses the SAME shared source image (see
--input-image), each resized to its own resolution.

Usage:
    # Test a single model (auto-launches server):
    python test_variants.py --model-path C:/models/sd-turbo-amdnpu-onnx

    # Full quality run instead of the default fast smoke check:
    python test_variants.py --model-path C:/models/sd-turbo-amdnpu-onnx --level detailed

    # Test all discovered models against a custom source image:
    python test_variants.py --all-models --input-image C:/photos/reference.png
"""

import base64
from functools import partial
from pathlib import Path

import numpy as np
import requests
from PIL import Image

import sd_test_common as common

MODE = "variants"


def run_test(ref_image_path, log, url, model_name, model_cfg, model_path, out_dir, level):
    """Run an unprompted image-variations test. Returns (success, detail)."""
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
    log(f"POST /v1/images/variations (unprompted) size={w}x{h} steps={steps} guidance={guidance} "
        f"strength={strength} seed={seed}")

    ref_img = Image.open(ref_image_path).convert("RGB").resize((w, h), Image.LANCZOS)
    image_b64 = base64.b64encode(np.array(ref_img).tobytes()).decode()

    output_path = out_dir / f"{MODE}_{model_name}_{level}.png"
    try:
        resp = requests.post(
            f"{url}/v1/images/variations",
            files={"image[]": ("image.bin", image_b64.encode())},
            data={
                # Deliberately NO "prompt" field — true unprompted variations.
                "size": f"{w}x{h}",
                "n": "1",
                "num_inference_steps": str(steps),
                "guidance_scale": str(guidance),
                "strength": str(strength),
                "seed": str(seed),
            },
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


def main():
    parser = common.build_common_parser(
        "variants test — unprompted image variations via /v1/images/variations (no prompt sent)."
    )
    parser.add_argument(
        "--input-image",
        help="Source image for this run (applies to ALL models tested, each resized to its "
             "own resolution). Default: test/img2img_test_input.png (auto-generated once if missing).",
    )
    args = parser.parse_args()
    common.validate_model_selection(parser, args)

    # "variants" reuses the "img2img" capability tag in models.json — both
    # need the same vae_encoder; they only differ in whether a prompt is sent.
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
