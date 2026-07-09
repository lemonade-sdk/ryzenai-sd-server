// generic_denoiser.h - Unified, metadata-driven denoiser for all SD variants
// Copyright (C) 2025 Advanced Micro Devices, Inc.
//
// A single denoiser implementation replaces the former per-variant SD15/SDXL/SD3
// denoisers. The classifier-free-guidance sampling loop is identical across all
// variants, so it lives here once. The only real differences between variants —
// which component to run, the text-embedding dimensions, and which optional
// inputs (pooled projections, SDXL time_ids, SD3 ControlNet blocks) the model
// expects — are expressed as data: a DenoiserSpec plus the model's own ONNX
// input metadata (names + dtypes) drive input binding.

#pragma once

#include "i_denoiser.h"
#include "onnx_model.h"
#include "controlnet_runner.h"
#include <memory>
#include <map>
#include <deque>
#include <vector>

namespace sd_npu {

// ============================================================================
// DenoiserSpec — static, per-variant description of the denoising model.
// Every behavioral difference between SD1.5 / SDXL / SD3 reduces to these
// values; the actual input binding is driven by the model's I/O metadata.
// ============================================================================

struct DenoiserSpec {
    ComponentType component = ComponentType::UNET;  // UNET (SD1.5/SDXL) or TRANSFORMER (SD3)
    int seq_len   = 77;    // text sequence length (encoder_hidden_states dim 1)
    int embed_dim = 768;   // encoder_hidden_states last dim
    int pool_dim  = 0;     // pooled projection dim (0 = model has no pooled input)
    int latent_ch = 4;     // latent channels (4 for SD1.5/SDXL, 16 for SD3)
};

// ============================================================================
// GenericDenoiser — one denoiser for every variant.
// ============================================================================

class GenericDenoiser : public IDenoiser {
public:
    /// @param components  Loaded ONNX models (must include spec.component)
    /// @param spec        Static description of the model's I/O
    /// @param controlnet_runner  Optional ControlNet helper (SD3 only; may be nullptr)
    GenericDenoiser(
        const std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
        DenoiserSpec spec,
        ControlNetRunner* controlnet_runner = nullptr);

    ~GenericDenoiser() override = default;

    std::vector<float> denoise(
        const std::vector<float>& latents,
        const std::vector<float>& text_embeddings,
        const std::vector<float>& pooled_embeddings,
        Scheduler& scheduler,
        const SDConfig& config,
        const std::vector<float>& controlnet_cond = {}) override;

    std::vector<ComponentType> required_components() const override {
        return {spec_.component};
    }

    int latent_channels() const override { return spec_.latent_ch; }

private:
    /// Bind every declared model input for a single step, dispatching by name.
    void build_inputs(
        std::vector<Ort::Value>& inputs,
        Ort::MemoryInfo& mem,
        const std::vector<float>& latent_input,
        float timestep,
        const std::vector<float>& text_emb,
        const std::vector<float>& pooled,
        int batch, int channels, int latent_h, int latent_w,
        const SDConfig& config);

    /// Create one input tensor of the model's declared dtype from a float
    /// source buffer, zero-padded / truncated to the target element count.
    Ort::Value emit(
        Ort::MemoryInfo& mem,
        const std::vector<float>& src,
        const std::vector<int64_t>& shape,
        ONNXTensorElementDataType dt);

    OnnxModel* model_ = nullptr;
    DenoiserSpec spec_;
    ControlNetRunner* controlnet_runner_ = nullptr;
    int controlnet_block_count_ = 0;
    int controlnet_channel_dim_ = 0;

    // Scratch storage kept alive across a single run() call. std::deque never
    // relocates existing elements on push_back, so data pointers handed to ORT
    // stay valid until the tensors are consumed.
    struct Scratch {
        std::deque<std::vector<float>>          f32;
        std::deque<std::vector<Ort::Float16_t>> f16;
        std::deque<std::vector<int64_t>>        i64;
        std::deque<std::vector<double>>         f64;
        void clear() { f32.clear(); f16.clear(); i64.clear(); f64.clear(); }
    };
    Scratch scratch_;

    // ControlNet block outputs (fp16), refreshed each step (SD3 only).
    std::vector<std::vector<Ort::Float16_t>> ctrl_fp16_;
};

} // namespace sd_npu
