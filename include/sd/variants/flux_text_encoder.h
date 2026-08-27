// flux_text_encoder.h - FLUX text encoder implementation (CLIP-L + T5-XXL)
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "i_text_encoder.h"
#include "onnx_model.h"
#include "clip_tokenizer.h"
#include "t5_tokenizer.h"
#include <map>
#include <memory>

namespace sd_npu {

// ============================================================================
// FluxTextEncoder: CLIP-L + T5-XXL encoder for FLUX models.
//
// FLUX uses T5-XXL last_hidden_state as encoder_hidden_states and CLIP-L
// pooler_output as pooled_projections. When no T5 component is loaded, falls
// back to CLIP-only conditioning (reduced quality).
//
// In schnell mode (guidance_embeds=false) the encoder returns a single-batch
// output (no CFG doubling).
// ============================================================================

class FluxTextEncoder : public ITextEncoder {
public:
    FluxTextEncoder(
        std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
        CLIPTokenizer& tokenizer,
        bool is_schnell = true,
        std::unique_ptr<T5Tokenizer> t5_tokenizer = nullptr,
        int t5_seq_len = 256);

    TextEncoderOutput encode(
        const std::string& prompt,
        const std::string& negative_prompt = "") override;

    std::vector<ComponentType> required_components() const override {
        return {ComponentType::TEXT_ENCODER};
    }

private:
    struct ClipOutput {
        std::vector<float> hidden;   // [seq, dim]
        std::vector<float> pooled;   // [dim]
    };

    ClipOutput run_clip(const std::string& text);

    // Returns [seq_len, 4096] flattened float32 embeddings from T5-XXL.
    // Returns empty vector if TEXT_ENCODER_2 is not loaded.
    std::vector<float> run_t5(const std::string& text);

    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components_;
    CLIPTokenizer& tokenizer_;
    std::unique_ptr<T5Tokenizer> t5_tokenizer_;  // optional; owned by this encoder
    bool is_schnell_;
    int t5_seq_len_;             // max T5 sequence length (default 256)
    int clip_dim_ = -1;

    std::string hidden_output_name_;
    std::string pooled_output_name_;
};

} // namespace sd_npu
