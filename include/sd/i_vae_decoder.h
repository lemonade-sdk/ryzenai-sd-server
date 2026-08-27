// i_vae_decoder.h - Abstract interface for VAE decoding across SD variants
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "sd_types.h"
#include <vector>
#include <cstdint>

namespace sd_npu {

// ============================================================================
// IVaeDecoder: Abstract interface for variant-specific VAE decoding
// ============================================================================

class IVaeDecoder {
public:
    virtual ~IVaeDecoder() = default;

    /// Decode latents to an RGB image. Returns [H, W, 3] uint8.
    virtual std::vector<uint8_t> decode(
        const std::vector<float>& latents,
        int height,
        int width) = 0;

    /// Which ComponentType entries this VAE needs loaded from ONNX models
    virtual std::vector<ComponentType> required_components() const = 0;

    /// VAE latent scaling factor (0.18215 for SD1.5/SDXL, 1.5305 for SD3)
    virtual float latent_scaling_factor() const = 0;

    /// VAE latent shift factor, applied before scaling on decode / after
    /// un-scaling on encode (0.0 for SD1.5/SDXL, 0.0609 for SD3, 0.1159 for FLUX)
    virtual float latent_shift_factor() const { return 0.0f; }
};

} // namespace sd_npu
