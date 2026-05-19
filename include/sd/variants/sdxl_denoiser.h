// sdxl_denoiser.h - SDXL and SDXL-Turbo denoiser implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "i_denoiser.h"
#include "onnx_model.h"
#include <memory>
#include <map>

namespace sd_npu {

// ============================================================================
// SDXLDenoiser: Denoiser for SDXL and SDXL-Turbo (UNet-based with time_ids)
// ============================================================================

class SDXLDenoiser : public IDenoiser {
public:
    /// Constructor
    /// @param components Map of loaded ONNX models (must include UNET)
    SDXLDenoiser(const std::map<ComponentType, std::unique_ptr<OnnxModel>>& components);

    ~SDXLDenoiser() override = default;

    /// Perform denoising loop
    std::vector<float> denoise(
        const std::vector<float>& latents,
        const std::vector<float>& text_embeddings,
        const std::vector<float>& pooled_embeddings,
        Scheduler& scheduler,
        const SDConfig& config,
        const std::vector<float>& controlnet_cond = {}) override;

    /// Required components for SDXL
    std::vector<ComponentType> required_components() const override {
        return {ComponentType::UNET};
    }

    /// SDXL uses 4 latent channels
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
        std::vector<float>& pooled,
        int batch, int channels, int latent_h, int latent_w,
        const SDConfig& config);

    OnnxModel* unet_;
    
    // Reusable buffers to avoid allocations per step
    struct DenoiserBuffers {
        std::vector<int64_t> ts_i64;
        std::vector<float> ts_f32;
        std::vector<float> time_ids_data;
        std::vector<int64_t> time_ids_i64;
    };
    DenoiserBuffers buffers_;
};

} // namespace sd_npu
