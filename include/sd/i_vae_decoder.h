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
};

} // namespace sd_npu
