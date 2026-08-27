# ryzenai-sd-server Tests

## Quick Start

```bash
pip install -r requirements.txt

# Each test is its own script. Every script works standalone with just a
# model target — --level defaults to "smoke" (fast, 1-step plumbing check).

python test_txt2img.py --model-path C:\path\to\sd-turbo-amdnpu-onnx
python test_img2img.py --model-path C:\path\to\sd-turbo-amdnpu-onnx
python test_variants.py --model-path C:\path\to\sd-turbo-amdnpu-onnx
python test_controlnet.py --model-path C:\path\to\stable-diffusion-3-medium-amdnpu-onnx
python test_cli.py --model-path C:\path\to\sd-turbo-amdnpu-onnx

# Or test all discovered models for a given script:
python test_txt2img.py --all-models
```

## Table of Contents

| Script | What it tests | Usage examples |
|--------|----------------|-----------------|
| `test_txt2img.py` | Text-to-image via `/v1/images/generations` | [Jump to usage](#test_txt2imgpy) |
| `test_img2img.py` | Prompt-guided image-to-image via `/v1/images/variations` | [Jump to usage](#test_img2imgpy) |
| `test_variants.py` | Unprompted image variations via `/v1/images/variations` | [Jump to usage](#test_variantspy) |
| `test_controlnet.py` | ControlNet-guided generation via `/v1/images/edits` | [Jump to usage](#test_controlnetpy) |
| `test_cli.py` | One-shot executable invocation, no server | [Jump to usage](#test_clipy) |

## Scripts

| Script | Endpoint | Prompt sent? | Needs VAE encoder? |
|--------|----------|:---:|:---:|
| `test_txt2img.py` | `/v1/images/generations` | yes | no |
| `test_img2img.py` | `/v1/images/variations` | yes (prompt-guided) | yes |
| `test_variants.py` | `/v1/images/variations` | no (true unprompted OpenAI-style variation) | yes |
| `test_controlnet.py` | `/v1/images/edits` | yes | no |
| `test_cli.py` | n/a — runs the executable directly, no server | yes | n/a |

`test_img2img.py` and `test_variants.py` hit the same server endpoint and
only differ in whether a `prompt` field is sent. Both use models.json's
`"img2img"` capability tag.

All five scripts share a common helper module, `sd_test_common.py` — config
loading, model discovery/auto-download, server launch/stop, tagged logging,
and the per-model runner loop live there. Each script itself only defines its
own request body and endpoint.

## Target selection (pick one, same on every script)

- `--url URL` — test against a server that's already running (no auto-launch)
- `--model-path PATH` — auto-launch the server for this one model
- `--all-models` — discover and test every model in `--models-dir`

## `--level` (smoke / detailed)

Every script accepts `--level {smoke,detailed}`, default `smoke`:

- **`smoke`** (default) — steps forced to 1, everything else at model
  defaults. Fast plumbing check: confirms the endpoint is wired correctly
  end-to-end without producing a representative image.
- **`detailed`** — full model defaults from `models.json` (steps, guidance,
  size, prompt). Produces a representative image; slower.

```bash
python test_txt2img.py --model-path C:\models\sd-turbo-amdnpu-onnx --level detailed
```

## Examples

```bash
# txt2img — single model, auto-launch, smoke check
python test_txt2img.py --model-path C:\models\sd-turbo-amdnpu-onnx

# txt2img — all models, full quality
python test_txt2img.py --all-models --level detailed

# img2img (prompt-guided) — all models that have a VAE encoder
python test_img2img.py --all-models
python test_img2img.py --all-models --input-image C:\photos\reference.png

# variants — unprompted image variations, same endpoint as img2img but no prompt
python test_variants.py --all-models

# ControlNet — specific types only (only runs on models declaring "controlnet")
python test_controlnet.py --all-models --types canny depth

# CLI mode — no server, runs the executable directly
python test_cli.py --model-path C:\models\sd-turbo-amdnpu-onnx
python test_cli.py --all-models

# Dry run — validate config and show what would be tested, no server/model files needed
python test_txt2img.py --all-models --dry-run
```

<a id="test_txt2imgpy"></a>
### `test_txt2img.py`

Generates an image straight from a text prompt via `/v1/images/generations`.
This is the simplest and fastest script — no VAE encoder, no source image,
no server-relaunch — so it's the best first check when validating a new
model or a server build.

**Common use cases:**

```bash
# Quick plumbing check for one model right after building/downloading it
python test_txt2img.py --model-path C:\models\sd-turbo-amdnpu-onnx

# Produce a representative, full-quality sample image for one model
python test_txt2img.py --model-path C:\models\sd-turbo-amdnpu-onnx --level detailed

# Smoke-test every discovered model in one pass (e.g. after a server change)
python test_txt2img.py --all-models

# Test against a server you already have running
python test_txt2img.py --url http://localhost:8080 --model-name sdxl-turbo-amdnpu-onnx

# Validate config/model discovery without needing model files or a server
python test_txt2img.py --all-models --dry-run
```

<a id="test_img2imgpy"></a>
### `test_img2img.py`

Prompt-guided image-to-image via `/v1/images/variations` (requires the
model to have a VAE encoder). Every model in a run is tested against the
SAME shared source image, each resized to that model's own resolution, so
outputs are directly comparable side by side.

**Common use cases:**

```bash
# Quick plumbing check for one img2img-capable model
python test_img2img.py --model-path C:\models\sd-turbo-amdnpu-onnx

# Full-quality img2img sample for one model
python test_img2img.py --model-path C:\models\sd-turbo-amdnpu-onnx --level detailed

# Compare img2img output across every capable model using the same source image
python test_img2img.py --all-models

# Use your own reference photo instead of the auto-generated default
python test_img2img.py --all-models --input-image C:\photos\reference.png
```

<a id="test_variantspy"></a>
### `test_variants.py`

True OpenAI-style unprompted image variations — same endpoint as
`test_img2img.py` (`/v1/images/variations`) but with NO prompt sent, so the
server falls back to its unprompted variation behavior. Useful for
verifying that omitting the prompt is handled correctly and doesn't crash
or silently reuse a stale prompt.

**Common use cases:**

```bash
# Quick plumbing check that unprompted variation requests are handled
python test_variants.py --model-path C:\models\sd-turbo-amdnpu-onnx

# Full-quality unprompted variation sample for one model
python test_variants.py --model-path C:\models\sd-turbo-amdnpu-onnx --level detailed

# Sweep every img2img-capable model with the same shared source image
python test_variants.py --all-models
```

<a id="test_controlnetpy"></a>
### `test_controlnet.py`

Generates with ControlNet guidance (canny, pose, tile, depth) via
`/v1/images/edits`. Only runs against models that declare `"controlnet"` in
their `models.json` capabilities; the server is relaunched per ControlNet
type since each type requires its own `-C <type>` startup flag.

**Common use cases:**

```bash
# Run every ControlNet type declared for one model
python test_controlnet.py --model-path C:\models\stable-diffusion-3-medium-amdnpu-onnx

# Only test specific ControlNet types
python test_controlnet.py --model-path C:\models\sd3-medium --types canny depth

# Full-quality ControlNet samples instead of the fast smoke check
python test_controlnet.py --model-path C:\models\sd3-medium --level detailed

# Sweep every controlnet-capable model that's been discovered
python test_controlnet.py --all-models
```

<a id="test_clipy"></a>
### `test_cli.py`

Runs the `ryzenai-sd-server` executable directly, one-shot, with no server
and no HTTP — the fastest way to sanity-check a build or a model without
standing up an API server at all.

**Common use cases:**

```bash
# Verify the CLI path works for a freshly built executable/model
python test_cli.py --model-path C:\models\sd-turbo-amdnpu-onnx

# Full-quality CLI generation sample
python test_cli.py --model-path C:\models\sd-turbo-amdnpu-onnx --level detailed

# Smoke-test the CLI path across every discovered model
python test_cli.py --all-models
```

## Options (shared across all scripts)

| Flag | Purpose |
|------|---------|
| `--level {smoke,detailed}` | See above. Default `smoke`. |
| `--port PORT` | Port for auto-launched server (default: 8080) |
| `--models-dir DIR` | Where to look for models (default: `$RAI_SD_MODELS_DIR` env var, or `C:\Users\mickraus\Work\rai_1.8.0_models`) |
| `--model-name NAME` | Model name to use for config lookup (used with `--url` when server is already running) |
| `--seed N` | Override seed |
| `--steps N` | Override inference steps (overrides `--level`'s own step count) |
| `--guidance F` | Override guidance scale |
| `--prompt TEXT` | Override prompt |
| `--dry-run` | Validate config and show what would be tested, without launching servers or requiring model files |

Script-specific extras:

| Flag | Script(s) | Purpose |
|------|-----------|---------|
| `--input-image PATH` | `test_img2img.py`, `test_variants.py` | Source image (default: `test/img2img_test_input.png`, auto-generated once if missing). The SAME resolved image is used for every model in a run — each just resizes it to its own resolution — so outputs are directly comparable across models. |
| `--types T [T...]` | `test_controlnet.py` | ControlNet types to test (default: the model's declared `controlnet_types`) |
| `--control-image PATH` | `test_controlnet.py` | Override control image path (applies to all types being tested) |

## Model Discovery & Auto-Download

With `--all-models`, models are discovered from two sources and merged:

1. **Flat directories** under `--models-dir` (e.g. `rai_1.8.0_models\sd-turbo-amdnpu-onnx\`).
2. **HuggingFace-cache-only models** — any entry in `models.json` with an
   `hf_repo_id` whose snapshot already exists in the HF hub cache
   (resolved the same way Lemonade resolves it: `$HF_HUB_CACHE` >
   `$HF_HOME/hub` > `~/.cache/huggingface/hub`, reading `refs/main` for the
   active commit).

If a model isn't found in either place but declares an `hf_repo_id`, it is
downloaded via `snapshot_download()` straight into the HF hub cache (no
`--model-path` needed) so the same copy is reusable by Lemonade.

Only models that declare the relevant capability in their `"capabilities"`
list are included (`txt2img`/`cli` also run against unknown/undeclared
models with safe defaults).

## Output & Logging

Each run writes to `test_outputs/<script-mode>/<timestamp>/`:

- `<mode>_<model>_<level>.png` (ControlNet: `<mode>_<model>_<type>_<level>.png`) —
  the decoded generation output.
- `<mode>_<model>_<level>.log` — per-model tagged log
  (`[mode][model][level] ...`) with every request logged and a final
  PASS/FAIL line.
- `report.json` — `{"timestamp", "mode", "level", "results": [{"model", "mode", "level", "success", "detail"}, ...]}`
  for every model (and, for ControlNet, every type) tested.

Console output is tagged the same way as the log files. On any model
failure (exception, HTTP error, missing model), the failure is logged and
the run **continues to the next model** — it never aborts the whole run.
The process exits non-zero if anything failed.

## Model Configuration

Model parameters (resolution, steps, guidance, prompts, capabilities,
`hf_repo_id`, `vae_encoder`, `controlnet_types`) are defined in `models.json`.
To add a new model, add an entry there — no code changes needed.

Models declare which capabilities they support via `"capabilities"`. Unknown
models get safe defaults for `txt2img` and `cli`.

The `"controlnet"` section of `models.json` defines per-type presets
(`guidance`, `steps`, `conditioning_scale`, control `image`) for each
ControlNet type exercised by `test_controlnet.py`.

## Lemonade Server Integration Tests

`lemonade_integration/` is a separate, standalone suite that tests this
repo's backend **as consumed through a running Lemonade Server (`lemond`)**,
rather than talking to `ryzenai-sd-server.exe`'s own HTTP API directly. It
lives here — instead of in the lemonade repo — so it stays under this repo's
ownership/commit access without needing to land any files upstream. It has
its own `README.md`, `requirements.txt`, and `lemonade_client.py` helper; see
[`lemonade_integration/README.md`](lemonade_integration/README.md) for setup
and usage.

## Known Limitations

- **Mask-aware ControlNet (InPainting/OutPainting/Removal) is not covered.**
  `test_controlnet.py` only ever sends the `image[]` field, never a `mask`
  field, and `models.json`'s `"controlnet"` section only has presets for
  `canny`/`pose`/`tile`/`depth`. This path is currently only verified by
  manual CLI runs, not by this harness.
- **No `negative_prompt` testing** — never wired from client to pipeline by
  the server for any endpoint, so it's excluded here too.
- **CLI mode only exercises plain txt2img** — it doesn't drive
  `--control-image`/`--control-mask`/`--strength`/img2img through the CLI
  argument path, even though the executable supports all of that.
- **No multi-image (`n > 1`) requests, error-path testing (bad prompt, bad
  image size, unknown ControlNet type), or concurrency testing.**
- **No parameter-passthrough verification** (i.e. asserting the server
  actually received the exact steps/guidance/seed/strength that was sent).
  This was considered and deliberately left out to keep the test scripts
  simple — it would require regex-parsing server stdout, which couples
  tests to log wording and adds fragility disproportionate to the value.
  If ever needed, treat it as a separate, explicitly-scoped follow-up.

## Structure

```
test/
├── sd_test_common.py     # Shared helpers: config, discovery, server mgmt,
│                          # tagged logging, per-model runner loop
├── test_txt2img.py        # /v1/images/generations
├── test_img2img.py        # /v1/images/variations (prompt-guided)
├── test_variants.py       # /v1/images/variations (unprompted)
├── test_controlnet.py     # /v1/images/edits
├── test_cli.py            # executable, no server
├── models.json            # Model configs (data-driven, no if/else)
├── requirements.txt
├── img2img_test_input.png  # Shared source image for img2img/variants (auto-generated if missing)
├── controlnet_images/     # Control images for ControlNet tests
└── lemonade_integration/  # Standalone tests against a running Lemonade Server
                           # (see its own README.md)
```
