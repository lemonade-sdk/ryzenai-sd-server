// controlnet_runner.h - Thin typed helper for ControlNet operations
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "onnx_model.h"
#include "sd_types.h"
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace sd_npu {

/// Thin typed wrapper for ControlNet model execution.
/// Encapsulates the complex tensor preparation and block output extraction.
class ControlNetRunner {
public:
    /// Construct a ControlNet runner.
    /// @param model Pointer to the loaded ONNX model (not owned)
    /// @param type The ControlNet type (Canny, Pose, etc.)
    ControlNetRunner(OnnxModel* model, ControlNetType type);

    /// Run ControlNet inference to generate block outputs for SD3/SD3.5.
    /// @param latent_input Current latent state [batch, channels, latent_h, latent_w]
    /// @param control_cond Normalized control image [-1,1] [batch, 3, height, width]
    /// @param timestep Current diffusion timestep (scalar)
    /// @param text_emb Text embeddings [batch, seq_len, embed_dim]
    /// @param pooled_emb Pooled text embeddings [batch, pool_dim]
    /// @param batch Batch size
    /// @param channels Number of latent channels
    /// @param latent_h Latent height
    /// @param latent_w Latent width
    /// @param height Actual image height (typically latent_h * 8)
    /// @param width Actual image width (typically latent_w * 8)
    /// @param seq_len Text sequence length
    /// @param embed_dim Text embedding dimension
    /// @param pool_dim Pooled embedding dimension
    /// @param conditioning_scale ControlNet conditioning scale (default: 0.7)
    /// @return Vector of 12 block outputs (fp16), or empty on failure
    std::vector<std::vector<Ort::Float16_t>> compute(
        const std::vector<float>& latent_input,
        const std::vector<float>& control_cond,
        float timestep,
        const std::vector<float>& text_emb,
        const std::vector<float>& pooled_emb,
        int batch, int channels, int latent_h, int latent_w,
        int height, int width,
        int seq_len, int embed_dim, int pool_dim,
        float conditioning_scale = 0.7f);

    /// Get the ControlNet type.
    ControlNetType get_type() const { return type_; }

    /// Check if the model is loaded and ready.
    bool is_loaded() const { return model_ != nullptr; }

    /// Get a human-readable name for this ControlNet.
    const char* get_name() const { return controlnet_to_string(type_); }

private:
    OnnxModel* model_;      // Not owned
    ControlNetType type_;

    /// Convert 6 block outputs to 12 by duplicating each block.
    void duplicate_blocks(
        const std::vector<Ort::Value>& outputs,
        std::vector<std::vector<Ort::Float16_t>>& ctrl_fp16_out);

    /// Extract 12 block outputs directly.
    void extract_blocks(
        const std::vector<Ort::Value>& outputs,
        std::vector<std::vector<Ort::Float16_t>>& ctrl_fp16_out);
};

} // namespace sd_npu
