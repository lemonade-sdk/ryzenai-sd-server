# lemonade_integration — external Lemonade Server integration tests

Standalone integration tests for the ryzenai-sd backend **as consumed through
a running `lemond` (Lemonade Server)**, as opposed to the sibling `test/`
scripts in this repo (`test_txt2img.py`, `test_img2img.py`, etc.) which talk
directly to `ryzenai-sd-server.exe`'s own HTTP API.

This suite intentionally has **no dependency on lemonade's internal test
harness** (`test/utils/*` in the lemonade repo) — it lives here because this
repo is the one under your ownership/commit access, and it only talks to
`lemond` over plain HTTP (`requests`).

## Why this exists

lemonade's own `test/server_sd_npu.py` covers basic ryzenai-sd plumbing, but
as of this writing it doesn't exercise: `n>1` (the bug that originally
prompted this investigation), img2img `strength`, size-resolution edge
cases, quad-prefix API parity, or a full visual sweep across all 13 NPU
models. This suite fills that gap end-to-end, from a black-box HTTP client's
point of view.

## Setup

```bash
pip install -r requirements.txt
```

Start `lemond` separately (e.g. via `LemonadeServer.exe`, the tray app, or
`lemond.exe <cache_dir> --port 13305`), then point these scripts at it:

```bash
set LEMONADE_HOST=127.0.0.1
set LEMONADE_PORT=13305
```

Both default to `127.0.0.1:13305` if unset (matching the convention already
used by `../validate_ryzenaisd.py`).

## Scripts

| Script | What it tests |
|--------|----------------|
| `test_backend_setup.py` | Model registration (`GET /models` lists all 13 ryzenai-sd models), `POST /load`, hot-swap (same subprocess PID across loads via `GET /health`), optional NPU-exclusivity eviction. |
| `test_api_contract.py` | Quad-prefix parity (`/api/v0`, `/api/v1`, `/v0`, `/v1`), seed determinism, **the `n>1` regression test**, missing-field/invalid-model/out-of-range-`n` error handling, default and explicit `size` resolution. |
| `test_visual_modes.py` | txt2img across all 13 models, prompt-guided img2img with a `strength` sweep (0.3/0.6/0.9) on `SD-1.5-NPU`/`SDXL-Base-NPU`, unprompted variants, and a controlnet placeholder (see below). Saves every PNG to `output/` for manual review, plus automated valid-PNG/dimension/non-blank checks. |

`lemonade_client.py` is the shared helper module: a thin `LemonadeClient`
HTTP wrapper (`/health`, `/models`, `/load`, `/images/generations`,
`/images/edits`, `/images/variations`), the expected-model registry
(`EXPECTED_MODELS`), and PNG sanity-check helpers (`assert_valid_png`,
`assert_not_blank`, `images_differ`, `decode_b64_images`).

Each script is standalone (`python test_*.py --help` for options) — none
require pytest or any lemonade-repo import.

## Known gap: ControlNet

`test_visual_modes.py --modes controlnet` currently just prints a skip
notice. `SD3-Medium-NPU` is already registered in lemonade and its
`ryzenai-sd-server.exe` supports ControlNet (Canny, Pose, Tile, Depth)
directly \u2014 no new model registry entry is needed. However, lemonade's own
C++ bridge (`server.cpp` / `ryzenaisd_server.cpp`) has **no request
parameter wired up** to select a ControlNet type or pass a control image
through to the subprocess (confirmed via source search \u2014 no
`controlnet`/`control_type`/`control_image` references anywhere in
lemonade's C++ code). That wiring needs to be added to lemonade first
(similar to the `strength` passthrough added earlier this session), then
`run_controlnet()` can be updated to exercise it via `/images/edits` against
`SD3-Medium-NPU`. The distinct `SD-1.5-ControlNet-Canny-NPU`
(`amd/sd1.5-controlnet-canny-amdnpu-onnx`) is now registered separately (see
below) and is unrelated to this capability.

## RAI 1.7 -> RAI 1.8 checkpoint migration (resolved)

lemonade's ryzenai-sd registry originally targeted **RyzenAI SDK 1.7**
checkpoints (e.g. `amd/stable-diffusion-turbo-amdnpu-onnx`). This repo's
merged `ryzenai-sd-integration` branch — and its built `ryzenai-sd-server.exe`
— is hard-configured for **RyzenAI SDK 1.8** exclusively (verified via
`CMakeLists.txt`, `config.json`, `src/main.cpp`, `src/onnx_model.cpp`, and
`.github/workflows/build_and_release.yml`, all of which reference
`RyzenAI/1.8.0` paths with zero RAI 1.7 references anywhere in the repo).

lemonade's `server_models.json` has been updated to use the RAI 1.8
checkpoint ids confirmed via this repo's CI `modelMap`
(`build_and_release.yml`):

| Model | Old (RAI 1.7) checkpoint | New (RAI 1.8) checkpoint |
|-------|---------------------------|---------------------------|
| `SD-Turbo-NPU` | `amd/stable-diffusion-turbo-amdnpu-onnx` | `amd/sd-turbo-amdnpu-onnx` |
| `SDXL-Turbo-NPU` | `amd/stable-diffusion-sdxl-turbo-amdnpu-onnx` | `amd/sdxl-turbo-amdnpu-onnx` |
| `SDXL-Base-NPU` | `amd/stable-diffusion-sdxl-base-amdnpu-onnx` | `amd/sdxl-base-amdnpu-onnx` |
| `Segmind-Vega-NPU` | `amd/stable-diffusion-segmind-vega-amdnpu-onnx` | `amd/segmind-vega-amdnpu-onnx` |

`SD-1.5-NPU`, `SD3-Medium-NPU`, and `SD3.5-Medium-NPU` already matched the CI
modelMap exactly and were left unchanged. `docs/tools/gen_backend_boilerplate.py`
was re-run in lemonade to regenerate `defaults.json`, `backends-reference.md`,
and `docs/guide/configuration/README.md` from the corrected registry.

`SD-1.5-NPU`, `SD3-Medium-NPU`, and `SD3.5-Medium-NPU` already matched the CI
modelMap exactly and were left unchanged. All checkpoint ids use the `-onnx`
naming convention (confirmed directly by the model author) — a bare
`amd/*-amdnpu` repo (no `-onnx` suffix) also exists publicly on HF for
several of these models, but that is a different/interim repo, not the one
ryzenai-sd-server expects. `docs/tools/gen_backend_boilerplate.py` was
re-run in lemonade to regenerate `defaults.json`, `backends-reference.md`,
and `docs/guide/configuration/README.md` from the corrected registry.

## All 13 models now registered in lemonade

This repo's own `test/models.json` (post-merge with
`feature/flux2-img2img-and-sd3-inpainting`) covers **13** models; lemonade's
`server_models.json` now registers all of them, using the model author's own
HF collection (https://huggingface.co/collections/makn87amd/ryzen-ai-180-sd-server)
and direct confirmation of the `-onnx` naming convention as the source of
truth for the 4 that previously had no confirmed `hf_repo_id` anywhere in
this repo:

- `SSD-1B-NPU` → `amd/SSD-1B-amdnpu-onnx`
- `Playground-v2.5-NPU` → `amd/playground-v2.5-1024px-aesthetic-amdnpu-onnx`
- `Dreamshaper-XL-Lightning-NPU` → `amd/dreamshaper-xl-lightning-amdnpu-onnx`
- `SD-1.5-ControlNet-Canny-NPU` → `amd/sd1.5-controlnet-canny-amdnpu-onnx`
  (a standalone ControlNet-canny checkpoint, distinct from `SD3-Medium-NPU`'s
  built-in ControlNet capability described above)

**Open question:** `FLUX.2-Klein-NPU`'s checkpoint is registered as
`amd/FLUX.2-klein-amdnpu-onnx` (matching the CI modelMap), but a live HF
search found a differently-named public repo `amd/FLUX.2-klein-4B-amdnpu`
(no `-onnx` suffix, with a `4B` size qualifier). Unconfirmed whether the
real `-onnx` checkpoint also includes `4B` in its name
(`amd/FLUX.2-klein-4B-amdnpu-onnx`) — needs verification before relying on
this model in production.

## Known gap: CI coverage

lemonade's `.github/workflows/cpp_server_build_test_release.yml` has no
build/install step for `ryzenai-sd-server.exe` — the self-hosted `xdna2`
runner must already have a pre-built binary on disk (pointed to by
`config.json`'s `ryzenaisd.npu_bin`), so CI silently keeps testing whatever
binary happens to be there rather than picking up new commits to this repo.
This suite doesn't fix that; it's flagged here as a follow-up to raise
separately (requires lemonade repo write access).
