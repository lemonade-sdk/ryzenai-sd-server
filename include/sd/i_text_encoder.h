// i_text_encoder.h - Abstract interface for text encoding across SD variants
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "sd_types.h"
#include <vector>
#include <string>

namespace sd_npu {

// ============================================================================
// ITextEncoder: Abstract interface for variant-specific text encoding
// ============================================================================

class ITextEncoder {
public:
    virtual ~ITextEncoder() = default;

    /// Encode prompt + negative prompt into CFG-batched embeddings
    /// Returns embeddings [2, seq, dim] and pooled [2, pool_dim]
    virtual TextEncoderOutput encode(
        const std::string& prompt,
        const std::string& negative_prompt = "") = 0;

    /// Which ComponentType entries this encoder needs loaded from ONNX models
    virtual std::vector<ComponentType> required_components() const = 0;
};

} // namespace sd_npu
