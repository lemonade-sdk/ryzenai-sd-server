// i_denoiser.h - Abstract interface for denoising across SD variants
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "sd_types.h"
#include "scheduler.h"
#include <vector>
#include <cstdint>

namespace sd_npu {

// ============================================================================
// IDenoiser: Abstract interface for variant-specific denoising (UNet/Transformer)
// ============================================================================

class IDenoiser {
public:
    virtual ~IDenoiser() = default;

    /// Perform denoising loop with the given latents and text embeddings
    /// Returns the denoised latents (same shape as input)
    virtual std::vector<float> denoise(
        const std::vector<float>& latents,
        const std::vector<float>& text_embeddings,
        const std::vector<float>& pooled_embeddings,
        Scheduler& scheduler,
        const SDConfig& config,
        const std::vector<float>& controlnet_cond = {}) = 0;

    /// Which ComponentType entries this denoiser needs loaded from ONNX models
    virtual std::vector<ComponentType> required_components() const = 0;

    /// Number of channels in latent space (4 for SD1.5/SDXL, 16 for SD3)
    virtual int latent_channels() const = 0;

    /// Latent space dimensions (height and width are divided by this factor)
    virtual int latent_scale_factor() const { return 8; }  // Default: 8x downsampling
};

} // namespace sd_npu
