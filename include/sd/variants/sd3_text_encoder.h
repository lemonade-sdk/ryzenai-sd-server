// sd3_text_encoder.h - SD3/SD3.5 text encoder implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "i_text_encoder.h"
#include "onnx_model.h"
#include "clip_tokenizer.h"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace sd_npu {

// ============================================================================
// SD3TextEncoder: CLIP-L + CLIP-G + T5 → [2, 160, 4096] + pooled [2, 2048]
// ============================================================================

class SD3TextEncoder : public ITextEncoder {
public:
    SD3TextEncoder(
        std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
        CLIPTokenizer& tokenizer,
        CLIPTokenizer& tokenizer_2,
        const std::string& t5_vocab_path = "");

    TextEncoderOutput encode(
        const std::string& prompt,
        const std::string& negative_prompt = "") override;

    std::vector<ComponentType> required_components() const override {
        return {ComponentType::TEXT_ENCODER, ComponentType::TEXT_ENCODER_2};
        // TEXT_ENCODER_3 (T5) is optional
    }

    void load_t5_vocab(const std::string& path);

private:
    void read_transformer_dims();
    
    void encode_clip_pair(
        const std::string& text,
        std::vector<float>& clip_hidden,
        std::vector<float>& clip_pooled);

    std::vector<int64_t> tokenize_t5(const std::string& text, int max_length) const;
    
    std::vector<float> encode_t5(const std::string& text, int max_length);

    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components_;
    CLIPTokenizer& tokenizer_;
    CLIPTokenizer& tokenizer_2_;

    // SD3 dimensions (read from transformer metadata)
    int total_seq_len_ = 160;   // 77 CLIP + 83 T5
    int embed_dim_ = 4096;

    // T5 vocabulary: SentencePiece piece → token ID
    std::unordered_map<std::string, int64_t> t5_vocab_;
    int64_t t5_unk_id_ = 2;
};

} // namespace sd_npu
