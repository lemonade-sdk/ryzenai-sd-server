// sd15_text_encoder.h - SD1.5 text encoder implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "i_text_encoder.h"
#include "onnx_model.h"
#include "clip_tokenizer.h"
#include <map>
#include <memory>

namespace sd_npu {

// ============================================================================
// SD15TextEncoder: Single CLIP encoder → [2, 77, 768]
// ============================================================================

class SD15TextEncoder : public ITextEncoder {
public:
    SD15TextEncoder(
        std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
        CLIPTokenizer& tokenizer,
        bool is_turbo = false);

    TextEncoderOutput encode(
        const std::string& prompt,
        const std::string& negative_prompt = "") override;

    std::vector<ComponentType> required_components() const override {
        return {ComponentType::TEXT_ENCODER};
    }

private:
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components_;
    CLIPTokenizer& tokenizer_;
    int clip_dim_ = -1;  // Cached dimension (768 for SD1.5, 1024 for SD2.1/Turbo)
    bool is_turbo_;       // SD-Turbo doesn't use CFG (batch=1)
};

} // namespace sd_npu
