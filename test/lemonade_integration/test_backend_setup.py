#!/usr/bin/env python3
"""
Backend registration / lifecycle checks for the ryzenai-sd <-> lemond
integration. Talks only to a running `lemond` over HTTP -- no dependency on
lemonade's internal test harness.

Usage:
    python test_backend_setup.py
    python test_backend_setup.py --host 127.0.0.1 --port 13305
    python test_backend_setup.py --npu-exclusive-model Llama-3.2-1B-Instruct-Hybrid

Checks performed:
    1. GET /health is reachable.
    2. GET /models?show_all=true lists all 7 expected ryzenai-sd models with
       recipe == "ryzenai-sd".
    3. POST /load successfully loads a ryzenai-sd model.
    4. Hot-swap: loading a second ryzenai-sd model reports the same
       ryzenai-sd-server.exe subprocess PID (via GET /health's
       `all_models_loaded[].pid`) instead of spawning a new one -- confirms
       the hot-swap path (`/v1/internal/load`) is taken rather than a full
       subprocess restart.
    5. (optional, needs --npu-exclusive-model) NPU exclusivity: loading a
       non-FLM NPU-exclusive recipe (e.g. ryzenai-llm, whispercpp on NPU)
       evicts the ryzenai-sd model.

Exits non-zero on any failed check.
"""

import argparse
import sys
import time

from lemonade_client import DEFAULT_HOST, DEFAULT_PORT, EXPECTED_MODELS, LemonadeClient


def check(label, condition, details=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {details}" if details else ""))
    return condition


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default="SD-Turbo-NPU",
                         help="ryzenai-sd model to load for the load/hot-swap checks")
    parser.add_argument("--second-model", default="SDXL-Turbo-NPU",
                         help="second ryzenai-sd model, used for the hot-swap PID check")
    parser.add_argument("--npu-exclusive-model", default=None,
                         help="a non-ryzenai-sd NPU-exclusive model name (e.g. a "
                              "ryzenai-llm or whispercpp/npu model) to test NPU eviction. "
                              "Skipped if not provided.")
    parser.add_argument("--load-timeout", type=int, default=300)
    args = parser.parse_args()

    client = LemonadeClient(host=args.host, port=args.port)
    failures = 0

    # 1. Health
    try:
        health = client.health()
        failures += not check("GET /health reachable", True, str(health.get("status")))
    except Exception as e:
        failures += not check("GET /health reachable", False, str(e))
        print("Cannot continue without a reachable server.", file=sys.stderr)
        sys.exit(1)

    # 2. Model registry
    try:
        registered = {m["id"]: m for m in client.ryzenai_sd_models()}
    except Exception as e:
        failures += not check("GET /models?show_all=true succeeded", False, str(e))
        sys.exit(1)

    missing = [m for m in EXPECTED_MODELS if m not in registered]
    failures += not check(
        "All 7 expected ryzenai-sd models registered",
        not missing,
        f"missing: {missing}" if missing else f"found: {sorted(registered)}",
    )
    for model_id, info in registered.items():
        failures += not check(
            f"{model_id}.recipe == 'ryzenai-sd'", info.get("recipe") == "ryzenai-sd",
            f"got recipe={info.get('recipe')!r}",
        )

    # 3. Load
    resp = client.load(args.model, timeout=args.load_timeout)
    failures += not check(f"POST /load {args.model}", resp.ok, f"{resp.status_code}: {resp.text[:300]}")

    loaded_entry = client.find_loaded(args.model)
    failures += not check(f"{args.model} appears in /health all_models_loaded", loaded_entry is not None)
    first_pid = loaded_entry.get("pid") if loaded_entry else None

    # 4. Hot-swap: load a second model, confirm subprocess PID is unchanged.
    resp2 = client.load(args.second_model, timeout=args.load_timeout)
    failures += not check(f"POST /load {args.second_model} (hot-swap)", resp2.ok,
                          f"{resp2.status_code}: {resp2.text[:300]}")

    second_entry = client.find_loaded(args.second_model)
    failures += not check(f"{args.second_model} appears in /health all_models_loaded",
                          second_entry is not None)
    second_pid = second_entry.get("pid") if second_entry else None

    if first_pid is not None and second_pid is not None:
        failures += not check(
            "Hot-swap reused the same ryzenai-sd-server.exe process (PID unchanged)",
            first_pid == second_pid,
            f"pid before={first_pid}, after={second_pid}",
        )
    else:
        print("[SKIP] Hot-swap PID comparison (pid missing from one of the /health entries)")

    # 5. NPU exclusivity (optional)
    if args.npu_exclusive_model:
        resp3 = client.load(args.npu_exclusive_model, timeout=args.load_timeout)
        failures += not check(
            f"POST /load {args.npu_exclusive_model} (should evict ryzenai-sd)",
            resp3.ok, f"{resp3.status_code}: {resp3.text[:300]}",
        )
        time.sleep(1)  # let eviction settle
        still_loaded = client.find_loaded(args.second_model) is not None
        failures += not check(
            f"{args.second_model} evicted by NPU-exclusive load",
            not still_loaded,
            "ryzenai-sd model still reports loaded" if still_loaded else "",
        )
    else:
        print("[SKIP] NPU exclusivity check (pass --npu-exclusive-model to enable)")

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        sys.exit(1)
    print("All checks passed")


if __name__ == "__main__":
    main()
