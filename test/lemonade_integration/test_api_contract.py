#!/usr/bin/env python3
"""
API-contract checks for the ryzenai-sd <-> lemond integration: quad-prefix
parity, seed determinism, the n>1 regression (the bug that originally
triggered this test suite), error handling, and size resolution. Talks only
to a running `lemond` over HTTP.

Usage:
    python test_api_contract.py
    python test_api_contract.py --model SD-Turbo-NPU
"""

import argparse
import sys

import requests

from lemonade_client import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_SMOKE_MODEL,
    EXPECTED_MODELS,
    LemonadeClient,
    assert_not_blank,
    assert_valid_png,
    decode_b64_images,
    images_differ,
)

QUAD_PREFIXES = ["/api/v0", "/api/v1", "/v0", "/v1"]


def check(label, condition, details=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {details}" if details else ""))
    return condition


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_SMOKE_MODEL,
                         help="fast ryzenai-sd model to use for all contract checks")
    args = parser.parse_args()

    client = LemonadeClient(host=args.host, port=args.port)
    failures = 0

    try:
        client.health()
    except Exception as e:
        print(f"Cannot reach lemond at {client.base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    resp = client.load(args.model, timeout=300)
    if not check(f"POST /load {args.model}", resp.ok, f"{resp.status_code}: {resp.text[:300]}"):
        sys.exit(1)

    expected_wh = EXPECTED_MODELS.get(args.model, {})

    # --- 1. Quad-prefix parity -------------------------------------------------
    for prefix in QUAD_PREFIXES:
        url = f"http://{args.host}:{args.port}{prefix}/health"
        try:
            r = requests.get(url, timeout=15)
            failures += not check(f"GET {prefix}/health", r.status_code == 200,
                                  f"status={r.status_code}")
        except Exception as e:
            failures += not check(f"GET {prefix}/health", False, str(e))

    gen_bodies = {}
    for prefix in ("/api/v1", "/v1"):
        url = f"http://{args.host}:{args.port}{prefix}/images/generations"
        body = {"model": args.model, "prompt": "a red apple on a table", "seed": 123, "n": 1}
        r = requests.post(url, json=body, timeout=300)
        ok = r.ok
        failures += not check(f"POST {prefix}/images/generations", ok, f"status={r.status_code}: {r.text[:300]}")
        if ok:
            gen_bodies[prefix] = r.json()

    if "/api/v1" in gen_bodies and "/v1" in gen_bodies:
        imgs_a = decode_b64_images(gen_bodies["/api/v1"])
        imgs_b = decode_b64_images(gen_bodies["/v1"])
        failures += not check(
            "/api/v1 and /v1 both return an image for the same request",
            len(imgs_a) == 1 and len(imgs_b) == 1,
        )

    # --- 2. Seed determinism ----------------------------------------------------
    seed_prompt = "a blue bicycle leaning against a brick wall"
    r1 = client.generate(args.model, seed_prompt, seed=42, n=1)
    r2 = client.generate(args.model, seed_prompt, seed=42, n=1)
    if check("Two generate() calls with seed=42 both succeed", r1.ok and r2.ok,
             f"{r1.status_code}, {r2.status_code}"):
        imgs1 = decode_b64_images(r1.json())
        imgs2 = decode_b64_images(r2.json())
        if imgs1 and imgs2:
            failures += not check(
                "Same seed produces identical output (determinism)",
                not images_differ(imgs1[0], imgs2[0]),
            )
    else:
        failures += 1

    # --- 3. n>1 regression (the original bug) -----------------------------------
    r_n3 = client.generate(args.model, "a green forest with sunlight", n=3)
    if check("POST /images/generations with n=3 succeeds", r_n3.ok,
             f"{r_n3.status_code}: {r_n3.text[:300]}"):
        imgs3 = decode_b64_images(r_n3.json())
        failures += not check("n=3 returns exactly 3 images", len(imgs3) == 3,
                              f"got {len(imgs3)}")
        if len(imgs3) == 3:
            pairwise_distinct = (
                images_differ(imgs3[0], imgs3[1])
                and images_differ(imgs3[0], imgs3[2])
                and images_differ(imgs3[1], imgs3[2])
            )
            failures += not check("n=3 images are pairwise distinct (not duplicated)",
                                  pairwise_distinct)
            for img_bytes in imgs3:
                img = assert_valid_png(img_bytes)
                assert_not_blank(img)
    else:
        failures += 1

    # --- 4. Negative / error cases -----------------------------------------------
    r = requests.post(f"{client.base_url}/images/generations", json={"model": args.model}, timeout=30)
    failures += not check("Missing 'prompt' -> 400", r.status_code == 400, f"got {r.status_code}")

    r = requests.post(f"{client.base_url}/images/generations", json={"prompt": "no model here"}, timeout=30)
    failures += not check("Missing 'model' -> 400", r.status_code == 400, f"got {r.status_code}")

    r = client.generate("Nonexistent-Model-XYZ", "a test prompt")
    failures += not check("Invalid model name -> error response", not r.ok,
                          f"got {r.status_code}")

    r = client.generate(args.model, "n too low", n=0)
    failures += not check("n=0 -> 400", r.status_code == 400, f"got {r.status_code}")

    r = client.generate(args.model, "n too high", n=11)
    failures += not check("n=11 -> 400", r.status_code == 400, f"got {r.status_code}")

    # --- 5. Size resolution -------------------------------------------------------
    r_default_size = client.generate(args.model, "default size check")
    if check("Default-size generation succeeds", r_default_size.ok, f"{r_default_size.status_code}"):
        imgs = decode_b64_images(r_default_size.json())
        if imgs:
            img = assert_valid_png(
                imgs[0],
                expected_width=expected_wh.get("width"),
                expected_height=expected_wh.get("height"),
            )
            failures += not check(
                "Default size matches model's declared image_defaults",
                img.width == expected_wh.get("width") and img.height == expected_wh.get("height"),
                f"got {img.width}x{img.height}, expected "
                f"{expected_wh.get('width')}x{expected_wh.get('height')}",
            )
    else:
        failures += 1

    override_w, override_h = 256, 256
    r_override = client.generate(args.model, "explicit size override check",
                                 size=f"{override_w}x{override_h}")
    if check("Explicit size override request succeeds", r_override.ok,
             f"{r_override.status_code}: {r_override.text[:300]}"):
        imgs = decode_b64_images(r_override.json())
        if imgs:
            img = assert_valid_png(imgs[0])
            failures += not check(
                f"Explicit size={override_w}x{override_h} is honored",
                img.width == override_w and img.height == override_h,
                f"got {img.width}x{img.height}",
            )
    else:
        failures += 1

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        sys.exit(1)
    print("All checks passed")


if __name__ == "__main__":
    main()
