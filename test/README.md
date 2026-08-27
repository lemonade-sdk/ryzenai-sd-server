# sd_npu_server Tests

## Quick Start

```bash
pip install -r requirements.txt

# Test against a running server:
python test_server.py txt2img --url http://localhost:8080

# Auto-launch server for one model:
python test_server.py txt2img --model-path C:\path\to\sd-turbo-amdnpu-onnx

# Test all models:
python test_server.py txt2img --all-models
```

## Usage

`test_server.py` is the single entry point. Pick a mode and a model target:

```bash
python test_server.py <mode> <target>
```

**Modes:**

| Mode | Endpoint | Prompt sent? | Needs VAE encoder? |
|------|----------|:---:|:---:|
| `txt2img` | `/v1/images/generations` | yes | no |
| `img2img` | `/v1/images/variations` | yes (prompt-guided) | yes |
| `variations` | `/v1/images/variations` | no (true unprompted OpenAI-style variation) | yes |
| `controlnet` | `/v1/images/edits` | yes | no |
| `cli` | n/a — runs the executable directly, no server | yes | n/a |

`img2img` and `variations` hit the same server endpoint and only differ in
whether a `prompt` field is sent. Both use models.json's `"img2img"`
capability tag.

**Targets** (pick one):
- `--url URL` — test against a running server (no auto-launch)
- `--model-path PATH` — auto-launch server for this model
- `--all-models` — discover and test all models in `--models-dir`

## Examples

```bash
# txt2img — single model, auto-launch
python test_server.py txt2img --model-path C:\models\sd-turbo-amdnpu-onnx

# txt2img — all models
python test_server.py txt2img --all-models

# img2img (prompt-guided) — all models that have a VAE encoder
python test_server.py img2img --all-models
python test_server.py img2img --all-models --input-image C:\photos\reference.png

# variations — unprompted image variations, same endpoint as img2img but no prompt
python test_server.py variations --all-models

# ControlNet — specific types only (only runs on models declaring "controlnet")
python test_server.py controlnet --all-models --types canny depth

# CLI mode — no server, runs the executable directly (basic txt2img generation only)
python test_server.py cli --model-path C:\models\sd-turbo-amdnpu-onnx
python test_server.py cli --all-models

# Dry run — validate config and show what would be tested, no server/model files needed
python test_server.py txt2img --all-models --dry-run
```

## Options

| Flag | Purpose |
|------|---------|
| `--port PORT` | Port for auto-launched server (default: 8080) |
| `--models-dir DIR` | Where to look for models (default: `$RAI_SD_MODELS_DIR` env var, or `C:\Users\mickraus\Work\rai_1.8.0_models`) |
| `--model-name NAME` | Model name to use for config lookup (used with `--url` when server is already running) |
| `--seed N` | Override seed |
| `--steps N` | Override inference steps |
| `--guidance F` | Override guidance scale |
| `--prompt TEXT` | Override prompt |
| `--types T [T...]` | ControlNet types to test (controlnet mode only; default is the model's declared `controlnet_types`) |
| `--control-image PATH` | Override control image path (controlnet mode only; applies to all types being tested) |
| `--input-image PATH` | Source image for `img2img`/`variations` (default: `test/img2img_test_input.png`, auto-generated once if missing). The SAME resolved image is used for every model in a run — each just resizes it to its own resolution — so outputs are directly comparable across models. |
| `--dry-run` | Validate config and show what would be tested, without launching servers or requiring model files |

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

Only models that declare the mode being tested in their `"capabilities"` list
are included (`txt2img`/`cli` also run against unknown/undeclared models with
safe defaults).

## Output

Each run writes to `test_outputs/<mode>/<timestamp>/`:
- `<model_name>.png` (or `<model_name>_<controlnet_type>.png` for controlnet mode) — the decoded generation output.
- `report.json` — `{"timestamp", "mode", "results": [{"model", "mode", "success", "detail"}, ...]}` for every model tested.

The process exits non-zero if any model in the run failed.

## Model Configuration

Model parameters (resolution, steps, guidance, prompts, capabilities,
`hf_repo_id`, `vae_encoder`, `controlnet_types`) are defined in `models.json`.
To add a new model, add an entry there — no code changes needed.

Models declare which modes they support via `"capabilities"`. Unknown models
get safe defaults for `txt2img` and `cli`.

The `"controlnet"` section of `models.json` defines per-type presets
(`guidance`, `steps`, `conditioning_scale`, control `image`) for each
ControlNet type exercised by `controlnet` mode.

## Known Limitations

- **Mask-aware ControlNet (InPainting/OutPainting/Removal) is not covered.**
  `test_controlnet()` only ever sends the `image[]` field, never a `mask`
  field, and `models.json`'s `"controlnet"` section only has presets for
  `canny`/`pose`/`tile`/`depth`. This path is currently only verified by
  manual CLI runs, not by this harness.
- **No `negative_prompt` testing** — never sent in any request body.
- **CLI mode only exercises plain txt2img** — it doesn't drive
  `--control-image`/`--control-mask`/`--strength`/img2img through the CLI
  argument path, even though the executable supports all of that.
- **No multi-image (`n > 1`) requests, error-path testing (bad prompt, bad
  image size, unknown ControlNet type), or concurrency testing.**

## Structure

```
test/
├── test_server.py        # Unified test runner
├── models.json           # Model configs (data-driven, no if/else)
├── requirements.txt
├── img2img_test_input.png  # Shared source image for img2img/variations (auto-generated if missing)
└── controlnet_images/    # Control images for ControlNet tests
```
