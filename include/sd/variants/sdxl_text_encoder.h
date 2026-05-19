// sdxl_text_encoder.h - SDXL text encoder implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "i_text_encoder.h"
#include "onnx_model.h"
#include "clip_tokenizer.h"
#include <map>
#include <memory>

namespace sd_npu {

// ============================================================================
// SDXLTextEncoder: CLIP-L + CLIP-G → [2, 77, 2048] + pooled [2, 1280]
// ============================================================================

class SDXLTextEncoder : public ITextEncoder {
public:
    SDXLTextEncoder(
        std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
        CLIPTokenizer& tokenizer,
        CLIPTokenizer& tokenizer_2,
        bool is_turbo = false);

    TextEncoderOutput encode(
        const std::string& prompt,
        const std::string& negative_prompt = "") override;

    std::vector<ComponentType> required_components() const override {
        return {ComponentType::TEXT_ENCODER, ComponentType::TEXT_ENCODER_2};
    }

private:
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components_;
    CLIPTokenizer& tokenizer_;
    CLIPTokenizer& tokenizer_2_;
    bool is_turbo_;
};

} // namespace sd_npu
