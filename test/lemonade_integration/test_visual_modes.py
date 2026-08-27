#!/usr/bin/env python3
"""
Visual-mode sweep for the ryzenai-sd <-> lemond integration: txt2img across
all 13 models, prompt-guided img2img with a strength sweep, unprompted
variants, and a controlnet placeholder (blocked -- see below). Saves every
generated PNG to ./output/ with a descriptive filename for manual review, in
addition to automated sanity checks (valid PNG, expected dimensions,
non-blank via pixel stddev).

Usage:
    python test_visual_modes.py
    python test_visual_modes.py --modes txt2img,img2img
    python test_visual_modes.py --models SD-Turbo-NPU,SDXL-Turbo-NPU

Controlnet is NOT exercised here: `SD3-Medium-NPU` is registered in lemonade
and its underlying ryzenai-sd-server.exe supports ControlNet (Canny, Pose,
Tile, Depth), but lemonade's own C++ bridge (server.cpp / ryzenaisd_server.cpp)
has no request parameter wired up yet to select a ControlNet type or pass a
control image through to the subprocess. See
test/lemonade_integration/README.md.
"""

import argparse
import sys
from pathlib import Path

from lemonade_client import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    EXPECTED_MODELS,
    LemonadeClient,
    assert_not_blank,
    assert_valid_png,
    decode_b64_images,
    make_test_input_png,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRENGTH_SWEEP = (0.3, 0.6, 0.9)
IMG2IMG_MODELS = ("SD-1.5-NPU", "SDXL-Base-NPU")
VARIANT_MODELS = ("SD-Turbo-NPU", "SDXL-Turbo-NPU")
ALL_MODES = ("txt2img", "img2img", "variants", "controlnet")


def check(label, condition, details=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {details}" if details else ""))
    return condition


def save(png_bytes, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    path.write_bytes(png_bytes)
    return path


def run_txt2img(client, models):
    failures = 0
    for model in models:
        expected = EXPECTED_MODELS.get(model, {})
        r = client.generate(model, f"a scenic vista, digital art ({model})", n=1)
        if not check(f"txt2img[{model}]: request succeeds", r.ok,
                     f"{r.status_code}: {r.text[:300]}"):
            failures += 1
            continue
        imgs = decode_b64_images(r.json())
        if not check(f"txt2img[{model}]: response has 1 image", len(imgs) == 1):
            failures += 1
            continue
        path = save(imgs[0], f"txt2img_{model}.png")
        img = assert_valid_png(imgs[0], expected.get("width"), expected.get("height"))
        stddev = assert_not_blank(img)
        print(f"    saved {path} ({img.width}x{img.height}, stddev={stddev:.2f})")
    return failures


def run_img2img(client, models):
    failures = 0
    input_png = make_test_input_png()
    for model in models:
        for strength in STRENGTH_SWEEP:
            r = client.variations(model, input_png,
                                  prompt=f"a watercolor painting ({model}, strength={strength})",
                                  strength=strength)
            label = f"img2img[{model}, strength={strength}]"
            if not check(f"{label}: request succeeds", r.ok, f"{r.status_code}: {r.text[:300]}"):
                failures += 1
                continue
            imgs = decode_b64_images(r.json())
            if not check(f"{label}: response has 1 image", len(imgs) == 1):
                failures += 1
                continue
            path = save(imgs[0], f"img2img_{model}_strength{strength}.png")
            img = assert_valid_png(imgs[0])
            stddev = assert_not_blank(img)
            print(f"    saved {path} ({img.width}x{img.height}, stddev={stddev:.2f})")
    return failures


def run_variants(client, models):
    failures = 0
    input_png = make_test_input_png()
    for model in models:
        r = client.variations(model, input_png)  # no prompt -> unprompted variation
        label = f"variants[{model}]"
        if not check(f"{label}: request succeeds", r.ok, f"{r.status_code}: {r.text[:300]}"):
            failures += 1
            continue
        imgs = decode_b64_images(r.json())
        if not check(f"{label}: response has 1 image", len(imgs) == 1):
            failures += 1
            continue
        path = save(imgs[0], f"variants_{model}.png")
        img = assert_valid_png(imgs[0])
        stddev = assert_not_blank(img)
        print(f"    saved {path} ({img.width}x{img.height}, stddev={stddev:.2f})")
    return failures


def run_controlnet():
    print("[SKIP] controlnet: SD3-Medium-NPU is registered in lemonade and its "
          "ryzenai-sd-server.exe supports ControlNet, but lemonade's C++ bridge "
          "(server.cpp/ryzenaisd_server.cpp) does not yet forward a ControlNet type or "
          "control image to the subprocess -- there is no request parameter for it. "
          "That needs to be added to lemonade before this can be exercised. "
          "See test/lemonade_integration/README.md for details.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--modes", default=",".join(ALL_MODES),
                         help=f"comma-separated subset of {ALL_MODES}")
    parser.add_argument("--models", default=None,
                         help="comma-separated model override for txt2img (default: all 13)")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - set(ALL_MODES)
    if unknown:
        print(f"Unknown mode(s): {unknown}. Valid modes: {ALL_MODES}", file=sys.stderr)
        sys.exit(2)

    client = LemonadeClient(host=args.host, port=args.port)
    try:
        client.health()
    except Exception as e:
        print(f"Cannot reach lemond at {client.base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    txt2img_models = (args.models.split(",") if args.models else list(EXPECTED_MODELS))

    failures = 0
    if "txt2img" in modes:
        failures += run_txt2img(client, txt2img_models)
    if "img2img" in modes:
        failures += run_img2img(client, IMG2IMG_MODELS)
    if "variants" in modes:
        failures += run_variants(client, VARIANT_MODELS)
    if "controlnet" in modes:
        failures += run_controlnet()

    print()
    print(f"Saved outputs to {OUTPUT_DIR} for manual review.")
    if failures:
        print(f"{failures} check(s) FAILED")
        sys.exit(1)
    print("All checks passed")


if __name__ == "__main__":
    main()
