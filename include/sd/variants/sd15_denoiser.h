// sd15_denoiser.h - SD1.5 and SD-Turbo denoiser implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "i_denoiser.h"
#include "onnx_model.h"
#include <memory>
#include <map>

namespace sd_npu {

// ============================================================================
// SD15Denoiser: Denoiser for SD1.5 and SD-Turbo (UNet-based)
// ============================================================================

class SD15Denoiser : public IDenoiser {
public:
    /// Constructor
    /// @param components Map of loaded ONNX models (must include UNET)
    /// @param embed_dim Embedding dimension (768 for SD1.5, 1024 for SD-Turbo)
    SD15Denoiser(
        const std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
        int embed_dim = 768);

    ~SD15Denoiser() override = default;

    /// Perform denoising loop
    std::vector<float> denoise(
        const std::vector<float>& latents,
        const std::vector<float>& text_embeddings,
        const std::vector<float>& pooled_embeddings,
        Scheduler& scheduler,
        const SDConfig& config,
        const std::vector<float>& controlnet_cond = {}) override;

    /// Required components for SD1.5/SD-Turbo
    std::vector<ComponentType> required_components() const override {
        return {ComponentType::UNET};
    }

    /// SD1.5 uses 4 latent channels
    int latent_channels() const override { return 4; }

private:
    /// Build model inputs for a single denoising step
    void build_inputs(
        std::vector<Ort::Value>& inputs,
        Ort::MemoryInfo& mem,
        OnnxModel* model,
        std::vector<float>& latent_model_input,
        float timestep,
        std::vector<float>& text_emb,
        int batch, int channels, int latent_h, int latent_w);

    OnnxModel* unet_;
    int embed_dim_;
    
    // Reusable buffers to avoid allocations per step
    struct DenoiserBuffers {
        std::vector<int64_t> ts_i64;
        std::vector<float> ts_f32;
        std::vector<double> ts_f64;
    };
    DenoiserBuffers buffers_;
};

} // namespace sd_npu
