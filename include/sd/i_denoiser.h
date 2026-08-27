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

    /// Convert VAE-encoder output (raw latent space, same layout the VAE
    /// decoder consumes) into whatever space denoise() actually expects as
    /// its `latents` input, for img2img. Most variants denoise directly in
    /// raw VAE latent space, so the default is a no-op passthrough.
    ///
    /// Variants whose denoise() operates on a transformed/packed/normalized
    /// token space (e.g. FLUX.2-klein's 2x2-patchified, BatchNorm-normalized
    /// 128-channel tokens) MUST override this to perform the forward
    /// transform, otherwise img2img will feed raw VAE latents into a space
    /// the transformer never sees during training, producing garbage output.
    virtual std::vector<float> prepare_img2img_latents(
        const std::vector<float>& vae_latents,
        int height, int width) const {
        (void)height; (void)width;
        return vae_latents;
    }

    /// Some variants (e.g. FLUX.2-klein) compute their own internal sigma
    /// schedule inside denoise() and ignore the Scheduler the pipeline
    /// passes in. For those, the pipeline's generic scheduler sigma used to
    /// blend noise with encoded latents for img2img would not match what the
    /// denoiser actually uses internally at the same step index. Override
    /// these two to report the denoiser's own sigma value instead.
    virtual bool has_custom_img2img_sigma() const { return false; }
    virtual float img2img_start_sigma(int /*start_step*/, int /*total_steps*/,
                                       int /*height*/, int /*width*/) const {
        return 0.0f;
    }
};

} // namespace sd_npu
