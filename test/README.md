# sd_npu_server Tests

## Quick Start

```bash
pip install -r requirements.txt

# Test against a running server:
python run_tests.py txt2img --url http://localhost:8080

# Auto-launch server for one model:
python run_tests.py txt2img --model-path C:\path\to\stable-diffusion-turbo-amdnpu-onnx

# Test all models:
python run_tests.py txt2img --all-models
```

## Usage

`run_tests.py` is the single entry point. Pick a mode and a model target:

```bash
python run_tests.py <mode> <target>
```

**Modes:** `txt2img`, `img2img`, `controlnet`, `cli`

**Targets** (pick one):
- `--url URL` — test against a running server (no auto-launch)
- `--model-path PATH` — auto-launch server for this model
- `--all-models` — discover and test all models in `--models-dir`

## Examples

```bash
# txt2img — single model, auto-launch
python run_tests.py txt2img --model-path C:\models\stable-diffusion-turbo-amdnpu-onnx

# txt2img — all models
python run_tests.py txt2img --all-models

# img2img — all models that have a VAE encoder
python run_tests.py img2img --all-models

# ControlNet — specific types only
python run_tests.py controlnet --all-models --types canny depth

# CLI mode — no server, runs the executable directly
python run_tests.py cli --model-path C:\models\stable-diffusion-turbo-amdnpu-onnx
python run_tests.py cli --all-models
```

## Options

| Flag | Purpose |
|------|---------|
| `--port PORT` | Port for auto-launched server (default: 8080) |
| `--models-dir DIR` | Where to look for models (default: `C:\Users\mickraus\sd_models\1.7.1_models`) |
| `--seed N` | Override seed |
| `--steps N` | Override inference steps |
| `--guidance F` | Override guidance scale |
| `--prompt TEXT` | Override prompt |
| `--types T [T...]` | ControlNet types to test (controlnet mode only) |

## Model Configuration

Model parameters (resolution, steps, guidance, prompts, capabilities) are defined in `models.json`. To add a new model, add an entry there — no code changes needed.

Models declare which modes they support via `"capabilities"`. Unknown models get safe defaults for `txt2img` and `cli`.

## Structure

```
tests/
├── run_tests.py          # Unified test runner
├── models.json           # Model configs (data-driven, no if/else)
├── requirements.txt
├── img2img_test_input.png
└── controlnet_images/    # Control images for ControlNet tests
```
