// flux_text_encoder.h - FLUX text encoder implementation (CLIP-L)
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "i_text_encoder.h"
#include "onnx_model.h"
#include "clip_tokenizer.h"
#include <map>
#include <memory>

namespace sd_npu {

// ============================================================================
// FluxTextEncoder: CLIP-L encoder for FLUX models.
//
// FLUX uses CLIP-L sequence output as encoder_hidden_states and CLIP-L pooled
// output as pooled_projections. T5-XXL text encoding (used in FLUX.1) is not
// currently supported since AMD ships it as a GPTQ binary rather than ONNX.
//
// In schnell mode (guidance_embeds=false) the encoder returns a single-batch
// output (no CFG doubling).
// ============================================================================

class FluxTextEncoder : public ITextEncoder {
public:
    FluxTextEncoder(
        std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
        CLIPTokenizer& tokenizer,
        bool is_schnell = true);

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

    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components_;
    CLIPTokenizer& tokenizer_;
    bool is_schnell_;

    int clip_dim_ = -1;
    std::string hidden_output_name_;
    std::string pooled_output_name_;
};

} // namespace sd_npu
