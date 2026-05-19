// sd3_denoiser.h - SD3 and SD3.5 denoiser implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "i_denoiser.h"
#include "onnx_model.h"
#include "controlnet_runner.h"
#include <memory>
#include <map>

namespace sd_npu {

// ============================================================================
// SD3Denoiser: Denoiser for SD3 and SD3.5 (Transformer-based, float16 inputs)
// ============================================================================

class SD3Denoiser : public IDenoiser {
public:
    /// Constructor
    /// @param components Map of loaded ONNX models (must include TRANSFORMER)
    /// @param controlnet_runner Optional ControlNet helper (can be nullptr)
    /// @param seq_len Sequence length for text embeddings (160 for SD3)
    /// @param embed_dim Embedding dimension (4096 for SD3)
    /// @param pool_dim Pooled projection dimension (2048 for SD3)
    SD3Denoiser(
        const std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
        ControlNetRunner* controlnet_runner = nullptr,
        int seq_len = 160,
        int embed_dim = 4096,
        int pool_dim = 2048);

    ~SD3Denoiser() override = default;

    /// Perform denoising loop
    std::vector<float> denoise(
        const std::vector<float>& latents,
        const std::vector<float>& text_embeddings,
        const std::vector<float>& pooled_embeddings,
        Scheduler& scheduler,
        const SDConfig& config,
        const std::vector<float>& controlnet_cond = {}) override;

    /// Required components for SD3
    std::vector<ComponentType> required_components() const override {
        return {ComponentType::TRANSFORMER};
    }

    /// SD3 uses 16 latent channels
    int latent_channels() const override { return 16; }

private:
    /// Build model inputs for a single denoising step
    void build_inputs(
        std::vector<Ort::Value>& inputs,
        Ort::MemoryInfo& mem,
        std::vector<float>& latent_model_input,
        float timestep,
        std::vector<float>& text_emb,
        std::vector<float>& pooled,
        int batch, int channels, int latent_h, int latent_w,
        const SDConfig& config);
ControlNetRunner* controlnet_runner_;  // Optional ControlNet runner (not owned)
    OnnxModel* transformer_;
    OnnxModel* controlnet_;  // Optional ControlNet model
    int seq_len_;
    int embed_dim_;
    int pool_dim_;
    int controlnet_block_count_;
    int controlnet_channel_dim_;
    
    // Reusable buffers to avoid allocations per step (SD3 uses float16 everywhere)
    struct DenoiserBuffers {
        std::vector<Ort::Float16_t> hs_fp16;          // hidden_states (latents)
        std::vector<Ort::Float16_t> ts_fp16;          // timestep
        std::vector<Ort::Float16_t> ehs_fp16;         // encoder_hidden_states
        std::vector<Ort::Float16_t> pooled_fp16;      // pooled_projections
        std::vector<std::vector<Ort::Float16_t>> ctrl_fp16;  // controlnet blocks
    };
    DenoiserBuffers buffers_;
};

} // namespace sd_npu
