# Ryzen AI SD Server

A lightweight Stable Diffusion inference server for AMD Ryzen AI NPUs using
ONNX Runtime, exposing an OpenAI-compatible REST API.

## Overview

This server loads a Stable Diffusion model from an ONNX model directory,
auto-detects its variant, and runs the full txt2img / img2img / ControlNet
pipeline on AMD Ryzen AI NPU hardware.

Key Features:

- **OpenAI API Compatible**: `/v1/images/generations`, `/v1/images/edits`, `/v1/images/variations`
- **Multi-Architecture**: SD 1.5, SDXL, SD3, SD3.5 with auto-detection
- **ControlNet Support**: Canny, Pose, Tile, Depth, and more
- **CLI & Server Modes**: One-shot generation or persistent HTTP server
- **Minimal Dependencies**: Single executable + DLLs
- **Self-Registering Variants**: Add new model architectures without changing core pipeline

### Supported Models

| Model | Variant | Default Resolution | Steps | Guidance |
|---|---|---|---|---|
| SD 1.5 | `sd15` | 512×512 | 20 | 7.5 |
| SD Turbo | `sd15` (turbo) | 512×512 | 1 | 0.0 |
| SDXL Base 1.0 | `sdxl` | 1024×1024 | 20 | 7.5 |
| SDXL Turbo | `sdxl` (turbo) | 512×512 | 1 | 0.0 |
| SD3 Medium | `sd3` | 1024×1024 | 28 | 7.0 |
| SD3.5 Medium | `sd35` | 1024×1024 | 28 | 4.5 |

The variant is **auto-detected** from the model directory structure.

## Building from Source

### Prerequisites

Windows Requirements:

- Windows 11 (64-bit)
- Visual Studio 2022
- CMake 3.20 or higher
- ONNX Runtime library (`onnxruntime.lib` / `onnxruntime.dll`)
  - Typically from Ryzen AI SDK at `C:\Program Files\RyzenAI\1.7.1\onnxruntime\lib`
  - ONNX Runtime headers are already vendored in the repo — no SDK needed for compilation

Hardware Requirements:

- AMD Ryzen AI processor (for NPU execution)
- Minimum 16GB RAM (32GB recommended)

### Build Steps (Windows)

```cmd
# Clone the repository
git clone https://github.com/lemonade-sdk/ryzenai-sd-server.git
cd ryzenai-sd-server

# Create and enter build directory
mkdir build
cd build

# Configure with CMake (uses default ORT lib path)
cmake .. -G "Visual Studio 17 2022" -A x64

# Build
cmake --build . --config Release
```

### Build Steps (Linux)

Linux Requirements:

- Ubuntu 22.04+ or equivalent
- GCC 9+ or Clang 10+
- CMake 3.20 or higher
- ONNX Runtime shared library (`libonnxruntime.so`)

```bash
# Clone the repository
git clone https://github.com/lemonade-sdk/ryzenai-sd-server.git
cd ryzenai-sd-server

# Create and enter build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build .
```

### Build Output

Windows: The executable and required DLLs will be created at:

```
build\bin\Release\ryzenai-sd-server.exe
```

Linux: The executable and required shared libraries will be created at:

```
build/bin/ryzenai-sd-server
```

ONNX Runtime DLLs are automatically copied from the lib directory to the
output directory during build.

### Custom ONNX Runtime Path

If the ONNX Runtime library is in a non-default location:

```cmd
# Windows — point at the directory containing onnxruntime.lib
cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DONNXRUNTIME_LIB_DIR="C:\custom\path\onnxruntime\lib"
cmake --build . --config Release
```

```bash
# Linux
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_LIB_DIR=/custom/path/onnxruntime/lib
cmake --build .
```

### Deploy Extra Runtime DLLs

If you have additional DLLs to ship alongside the executable (e.g., RyzenAI
custom ops, DynamicDispatch runtime):

```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DRUNTIME_DLLS_DIR="C:\Program Files\RyzenAI\1.7.1\deployment"
cmake --build . --config Release
```

To deploy a full directory tree (preserving subdirectory structure):

```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DLIB_DIRECTORY="C:\Program Files\RyzenAI\1.7.1\GenAI-SD\lib"
cmake --build . --config Release
```

## Code Structure

```
ryzenai-sd-server/
├── CMakeLists.txt              # Build configuration
│
├── src/                        # Source files
│   ├── main.cpp                # Entry point, CLI arg parsing
│   ├── http_server.cpp         # HTTP server (cpp-httplib)
│   ├── sd_pipeline.cpp         # Core pipeline: encode → denoise → decode
│   ├── onnx_model.cpp          # ONNX Runtime session wrapper
│   ├── scheduler.cpp           # Noise schedulers (Euler, PNDM, etc.)
│   ├── clip_tokenizer.cpp      # CLIP text tokenizer
│   ├── config_loader.cpp       # Configuration file loader
│   ├── controlnet_runner.cpp   # ControlNet model runner
│   ├── variant_registry.cpp    # Variant registration system
│   └── variants/               # Per-architecture implementations
│       ├── sd15_variant.cpp    # SD 1.5 registration + detection
│       ├── sdxl_variant.cpp    # SDXL registration + detection
│       ├── sd3_variant.cpp     # SD3/SD3.5 registration + detection
│       └── *_text_encoder.cpp / *_denoiser.cpp / *_vae_decoder.cpp
│
├── include/sd/                 # Headers
│   ├── sd_types.h              # Common data structures
│   ├── sd_model.h              # Model configuration
│   ├── sd_pipeline.h           # Pipeline interface
│   ├── onnx_model.h            # ONNX session wrapper
│   ├── scheduler.h             # Scheduler interface
│   ├── clip_tokenizer.h        # Tokenizer interface
│   ├── http_server.h           # HTTP server interface
│   ├── config_loader.h         # Config loader interface
│   ├── controlnet_runner.h     # ControlNet interface
│   ├── variant_factory.h       # Variant factory types
│   ├── variant_registry.h      # Variant registry
│   ├── i_text_encoder.h        # Text encoder interface
│   ├── i_denoiser.h            # Denoiser interface
│   ├── i_vae_decoder.h         # VAE decoder interface
│   └── variants/               # Variant-specific headers
│       ├── sd15_*.h
│       ├── sdxl_*.h
│       └── sd3_*.h
│
├── include/onnxruntime/        # Vendored ONNX Runtime headers (no SDK needed)
│   ├── onnxruntime_cxx_api.h
│   ├── onnxruntime_c_api.h
│   └── ...
│
├── test/                       # Test scripts
│   ├── test_server.py          # Server test runner
│   ├── models.json             # Model test configurations
│   └── requirements.txt        # Python test dependencies
│
└── external/                   # Header-only dependencies
    └── cpp-httplib/            # HTTP server (auto-downloaded)
```

## Architecture Overview

### Design Principles

1. **Simplicity**: Single executable with no external runtime dependencies beyond ONNX Runtime
2. **RAII**: Resource management follows C++ best practices with smart pointers
3. **Thread Safety**: Pipeline guarded by `std::mutex`, configs cloned per-request
4. **Convention Over Configuration**: Variant auto-detected from model directory layout
5. **Extensible**: Self-registering variant system — add models without changing core code

### Component Layers

```
┌─────────────────────────────────────────────────┐
│         HTTP Server (cpp-httplib)               │
│         OpenAI API Endpoints                    │
├─────────────────────────────────────────────────┤
│         SD Pipeline                             │
│         encode → denoise → decode               │
├─────────────────────────────────────────────────┤
│         Variant System                          │
│         SD1.5 / SDXL / SD3 / SD3.5             │
├─────────────────────────────────────────────────┤
│         ONNX Runtime + RyzenAI Custom Ops       │
│         NPU / CPU Execution                     │
└─────────────────────────────────────────────────┘
```

### Dependencies

Vendored in the repo (no download/install needed):

- **ONNX Runtime headers** (1.23.3) - Vendored in `include/onnxruntime/`
- **cpp-httplib** (v0.26.0) - HTTP server, auto-downloaded by CMake [MIT License]

Required externally (link library + runtime DLLs only):

- **ONNX Runtime** - `onnxruntime.lib` / `onnxruntime.dll` (from RyzenAI SDK or standalone)

## Usage

### CLI Mode (one-shot generation)

```cmd
# Generate with auto-detected variant
ryzenai-sd-server.exe -m C:\path\to\onnx\model -p "A sunset over mountains"

# With custom parameters
ryzenai-sd-server.exe -m C:\models\sdxl -p "A serene lake" -n 20 -g 7.5 -W 1024 -H 1024
```

### Server Mode (HTTP API)

```cmd
# Start the server
ryzenai-sd-server.exe --server -m C:\path\to\onnx\model --port 8080
```

### Command-Line Arguments

- `-m, --model-path PATH` - Path to ONNX model directory (required)
- `--server` - Start HTTP server instead of single generation
- `--port PORT` - Server port (default: 8080)
- `-p, --prompt TEXT` - Generation prompt
- `-n, --num-inference-steps N` - Denoising steps (auto per variant)
- `-g, --guidance-scale FLOAT` - CFG scale (auto per variant)
- `-W WIDTH` / `-H HEIGHT` - Image dimensions
- `-s, --seed INT` - Random seed (default: 0)
- `-v, --variant NAME` - Model variant: sd15, sdxl, sd3, sd35 (auto-detected)
- `-C, --controlnet TYPE` - ControlNet type: Canny, Pose, Tile, Depth, etc.
- `--force-cpu` - Force CPU execution provider
- `-h, --help` - Show help message

## API Endpoints

The server implements OpenAI-compatible API endpoints.

### Health Check

```
GET /health
```

Returns server status.

### Image Generation

- `POST /v1/images/generations` - Text-to-image generation
- `POST /v1/images/edits` - Image-to-image + ControlNet
- `POST /v1/images/variations` - Image variations

All endpoints return responses in OpenAI Images API format.

## Testing

```bash
cd test
pip install -r requirements.txt

# Test against a running server
python test_server.py txt2img --url http://localhost:8080

# Auto-launch server for one model
python test_server.py txt2img --model-path C:\path\to\model

# Test all models
python test_server.py txt2img --all-models
```

## Integration with Lemonade Server

This server is designed to be used as a backend for
[Lemonade Server](https://github.com/lemonade-sdk/lemonade). When running
Lemonade Server, the `ryzenai-sd-server` executable is automatically downloaded
from GitHub releases and managed by the Lemonade Router.

## Integration Examples

### Python with requests

```python
import requests
import base64
from PIL import Image
import numpy as np

response = requests.post(
    "http://localhost:8080/v1/images/generations",
    json={
        "prompt": "A red cat sitting on a windowsill",
        "size": "512x512",
        "n": 1,
        "response_format": "b64_json"
    }
).json()

raw = base64.b64decode(response["data"][0]["b64_json"])
img = Image.frombytes("RGB", (response["width"], response["height"]), raw)
img.save("output.png")
```

## Development

### Code Style

- C++17 standard
- RAII for resource management
- Smart pointers (no raw pointers)
- Const correctness

### Building for Development

Windows:

```cmd
cmake --build . --config Debug
```

Debug executable location: `build\bin\Debug\ryzenai-sd-server.exe`

Linux:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

Debug executable location: `build/bin/ryzenai-sd-server`

## Related Projects

- [Ryzen AI Documentation](https://ryzenai.docs.amd.com)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [Lemonade Server](https://github.com/lemonade-sdk/lemonade) - Parent project providing model orchestration
- [Ryzen AI LLM Server](https://github.com/lemonade-sdk/ryzenai-server) - Companion LLM server

## License

This project's source code is licensed under the MIT License - see
[LICENSE](LICENSE) for details.

Release Artifacts (ryzenai-sd-server.zip):

- The `ryzenai-sd-server` binary and the header-only dependencies (cpp-httplib)
  are MIT licensed
- The Ryzen AI DLLs included in binary releases are licensed under the AMD
  Software End User License Agreement
