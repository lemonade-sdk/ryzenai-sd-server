#!/usr/bin/env python3
"""
Thin HTTP client for exercising a running `lemond` (Lemonade Server) from
outside the lemonade repo, plus small image sanity-check helpers.

This suite treats lemond purely as a black box over its public REST API. It
has no dependency on lemonade's internal test harness (`test/utils/*` in the
lemonade repo) since this repo does not have write access there -- see
test/lemonade_integration/README.md for the rationale.

Target server selection (all optional, sensible defaults match the
`LEMONADE_PORT` convention already used by ../validate_ryzenaisd.py):
    LEMONADE_HOST          default 127.0.0.1
    LEMONADE_PORT          default 13305
    LEMONADE_TEST_TIMEOUT  default 600 (seconds; image generation can be slow)
"""

import io
import os
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from PIL import Image

DEFAULT_HOST = os.environ.get("LEMONADE_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("LEMONADE_PORT", "13305"))
DEFAULT_TIMEOUT = int(os.environ.get("LEMONADE_TEST_TIMEOUT", "600"))

# Recipe id used for every ryzenai-sd model entry in server_models.json /
# user_models.json -- see src/cpp/include/lemon/backends/ryzenaisd/ryzenaisd.h
RYZENAI_SD_RECIPE = "ryzenai-sd"

# The 13 built-in ryzenai-sd model ids registered in lemonade's
# src/cpp/resources/server_models.json, with their `image_defaults`
# (width/height used when a test doesn't override `size`). Kept here (rather
# than re-derived from a live server) so tests can assert *all* expected
# models are present, not just whatever happens to be registered. If lemonade
# adds/renames a ryzenai-sd model, update this dict to match.
#
# NOTE: as of the ryzenai-sd-integration merge, lemonade's checkpoints target
# RyzenAI SDK 1.8 (matching this repo's CI modelMap in
# .github/workflows/build_and_release.yml). SD-Turbo-NPU, SDXL-Turbo-NPU,
# SDXL-Base-NPU, and Segmind-Vega-NPU were renamed from their old RAI 1.7
# checkpoint ids (e.g. amd/stable-diffusion-turbo-amdnpu-onnx ->
# amd/sd-turbo-amdnpu-onnx). All checkpoint ids use the `-onnx` naming
# convention (confirmed by the model author) -- a bare `amd/*-amdnpu` repo
# (no `-onnx` suffix) also exists publicly on HF for several of these models,
# but that is a different/interim repo, not the one ryzenai-sd-server expects.
EXPECTED_MODELS: Dict[str, Dict[str, int]] = {
    "SD-Turbo-NPU": {"width": 512, "height": 512},
    "SDXL-Turbo-NPU": {"width": 1024, "height": 1024},
    "SD-1.5-NPU": {"width": 512, "height": 512},
    "SDXL-Base-NPU": {"width": 1024, "height": 1024},
    "SD3-Medium-NPU": {"width": 1024, "height": 1024},
    "SD3.5-Medium-NPU": {"width": 1024, "height": 1024},
    "Segmind-Vega-NPU": {"width": 1024, "height": 1024},
    "FLUX.1-Schnell-NPU": {"width": 1024, "height": 1024},
    "FLUX.2-Klein-NPU": {"width": 1024, "height": 1024},
    "SSD-1B-NPU": {"width": 1024, "height": 1024},
    "Playground-v2.5-NPU": {"width": 1024, "height": 1024},
    "Dreamshaper-XL-Lightning-NPU": {"width": 1024, "height": 1024},
    "SD-1.5-ControlNet-Canny-NPU": {"width": 512, "height": 512},
}

# Subset that supports img2img/edits/variations testing with a small,
# fast-to-load model (SD-Turbo-NPU: 1 step, 512x512).
DEFAULT_SMOKE_MODEL = "SD-Turbo-NPU"


class LemonadeApiError(RuntimeError):
    """Raised when a lemond response has a non-2xx status or an error body."""

    def __init__(self, message: str, status_code: int, body: Any = None):
        super().__init__(f"{message} (HTTP {status_code}): {body}")
        self.status_code = status_code
        self.body = body


class LemonadeClient:
    """Minimal wrapper around lemond's `/api/v1/*` REST API."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: int = DEFAULT_TIMEOUT):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}/api/v1"

    # ---- generic ----------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=30)
        r.raise_for_status()
        return r.json()

    def loaded_models(self) -> List[Dict[str, Any]]:
        """`all_models_loaded` from /health: name/recipe/pid/device for every
        currently-loaded model (across all model types), most reliable source
        for hot-swap PID checks and NPU-eviction checks."""
        return self.health().get("all_models_loaded", [])

    def find_loaded(self, model_name: str) -> Optional[Dict[str, Any]]:
        for entry in self.loaded_models():
            if entry.get("model_name") == model_name:
                return entry
        return None

    def list_models(self, show_all: bool = True) -> List[Dict[str, Any]]:
        params = {"show_all": "true"} if show_all else {}
        r = requests.get(f"{self.base_url}/models", params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("data", [])

    def ryzenai_sd_models(self, show_all: bool = True) -> List[Dict[str, Any]]:
        """All registered models whose recipe is `ryzenai-sd`."""
        return [m for m in self.list_models(show_all) if m.get("recipe") == RYZENAI_SD_RECIPE]

    def load(self, model_name: str, timeout: Optional[int] = None,
              **extra) -> requests.Response:
        body = {"model_name": model_name}
        body.update(extra)
        return requests.post(f"{self.base_url}/load", json=body,
                              timeout=timeout or self.timeout)

    def unload(self, timeout: int = 60) -> requests.Response:
        return requests.post(f"{self.base_url}/unload", timeout=timeout)

    # ---- images -------------------------------------------------------------
    # See lemonade's docs/api/lemonade.md and src/cpp/server/server.cpp
    # (handle_image_generations / handle_image_edits / handle_image_variations)
    # for the request contract these mirror.

    def generate(self, model: str, prompt: str, timeout: Optional[int] = None,
                 **kwargs) -> requests.Response:
        """POST /images/generations (JSON body)."""
        body = {"model": model, "prompt": prompt}
        body.update(kwargs)
        return requests.post(f"{self.base_url}/images/generations", json=body,
                              timeout=timeout or self.timeout)

    def edit(self, model: str, prompt: str, image_bytes: bytes,
              mask_bytes: Optional[bytes] = None, timeout: Optional[int] = None,
              **kwargs) -> requests.Response:
        """POST /images/edits (multipart/form-data): image[] + optional mask."""
        data = {"model": model, "prompt": prompt}
        data.update({k: str(v) for k, v in kwargs.items()})
        files = {"image[]": ("image.png", image_bytes, "image/png")}
        if mask_bytes is not None:
            files["mask"] = ("mask.png", mask_bytes, "image/png")
        return requests.post(f"{self.base_url}/images/edits", data=data, files=files,
                              timeout=timeout or self.timeout)

    def variations(self, model: str, image_bytes: bytes, prompt: Optional[str] = None,
                    timeout: Optional[int] = None, **kwargs) -> requests.Response:
        """POST /images/variations (multipart/form-data): image[] only."""
        data = {"model": model}
        if prompt is not None:
            data["prompt"] = prompt
        data.update({k: str(v) for k, v in kwargs.items()})
        files = {"image[]": ("image.png", image_bytes, "image/png")}
        return requests.post(f"{self.base_url}/images/variations", data=data, files=files,
                              timeout=timeout or self.timeout)


# ---- image helpers ----------------------------------------------------------


def decode_b64_images(response_json: Dict[str, Any]) -> List[bytes]:
    """Extract raw PNG bytes from an OpenAI-style `{"data": [{"b64_json": ...}]}`
    images response."""
    import base64

    out = []
    for item in response_json.get("data", []):
        b64 = item.get("b64_json")
        if b64:
            out.append(base64.b64decode(b64))
    return out


def make_test_input_png(width: int = 512, height: int = 512) -> bytes:
    """A deterministic, non-blank input image for img2img/edits/variations
    tests (a diagonal gradient -- flat solid colors are a weaker test input
    since some backends could pass a blank frame through unchanged and still
    look "valid")."""
    xv, yv = np.meshgrid(np.linspace(0, 255, width), np.linspace(0, 255, height))
    arr = np.stack([xv, yv, (xv + yv) / 2], axis=-1).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def assert_valid_png(png_bytes: bytes, expected_width: Optional[int] = None,
                      expected_height: Optional[int] = None) -> Image.Image:
    """Verify the bytes decode as a valid PNG and (optionally) match the
    expected dimensions. Returns the opened (re-opened, since `.verify()`
    invalidates the handle) image for further inspection."""
    buf = io.BytesIO(png_bytes)
    img = Image.open(buf)
    img.verify()  # raises if corrupt

    buf.seek(0)
    img = Image.open(buf)
    img.load()

    if expected_width is not None:
        assert img.width == expected_width, (
            f"expected width {expected_width}, got {img.width}")
    if expected_height is not None:
        assert img.height == expected_height, (
            f"expected height {expected_height}, got {img.height}")
    return img


def assert_not_blank(img: Image.Image, min_stddev: float = 2.0) -> float:
    """Guard against a backend silently returning a solid/near-blank image
    (e.g. an all-black frame from a crashed generation). Returns the
    per-channel-averaged stddev for logging."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    stddev = float(arr.std())
    assert stddev >= min_stddev, (
        f"image looks blank (pixel stddev {stddev:.3f} < {min_stddev})")
    return stddev


def images_differ(png_bytes_a: bytes, png_bytes_b: bytes) -> bool:
    """True if two same-size PNGs have different pixel content (used for the
    n>1 distinctness check and seed-determinism checks)."""
    img_a = np.asarray(Image.open(io.BytesIO(png_bytes_a)).convert("RGB"))
    img_b = np.asarray(Image.open(io.BytesIO(png_bytes_b)).convert("RGB"))
    if img_a.shape != img_b.shape:
        return True
    return not np.array_equal(img_a, img_b)
