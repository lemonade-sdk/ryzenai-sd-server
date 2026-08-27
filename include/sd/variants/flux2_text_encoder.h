// flux2_text_encoder.h - FLUX.2-klein text encoder (Qwen3)
// Copyright (C) 2025 Advanced Micro Devices, Inc.
//
// FLUX.2-klein conditions on a Qwen3 text encoder. The prompt is wrapped in the
// Qwen chat template, tokenized with byte-level BPE (Qwen3Tokenizer), and run
// through the qwen3_text_encoder ONNX model to produce the sequence embeddings
// consumed as encoder_hidden_states by the klein transformer.

#pragma once

#include "i_text_encoder.h"
#include "onnx_model.h"
#include "qwen3_tokenizer.h"
#include <map>
#include <memory>
#include <string>

namespace sd_npu {

class Flux2TextEncoder : public ITextEncoder {
public:
    // model_root: FLUX.2-klein model directory (contains tokenizer/vocab.json + merges.txt).
    Flux2TextEncoder(
        std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
        const std::string& model_root,
        int max_sequence_length = 256);

    TextEncoderOutput encode(
        const std::string& prompt,
        const std::string& negative_prompt = "") override;

    std::vector<ComponentType> required_components() const override {
        return {ComponentType::TEXT_ENCODER};
    }

    int embed_dim() const { return embed_dim_; }
    int sequence_length() const { return max_seq_len_; }

private:
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components_;
    Qwen3Tokenizer tokenizer_;
    int max_seq_len_ = 256;
    int embed_dim_   = 7680;  // Qwen3 hidden size used by klein
    bool tokenizer_loaded_ = false;
};

} // namespace sd_npu
