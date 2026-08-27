// qwen3_tokenizer.h - Qwen3 byte-level BPE tokenizer for FLUX.2-klein
// Copyright (C) 2025 Advanced Micro Devices, Inc.
//
// FLUX.2-klein conditions on a Qwen3 text encoder. Prompts are wrapped in the
// Qwen chat template and tokenized with GPT-2 style byte-level BPE (vocab.json +
// merges.txt), producing input_ids / attention_mask / position_ids for the
// qwen3_text_encoder ONNX model.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cstdint>

namespace sd_npu {

struct Qwen3Encoding {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> position_ids;
    int seq_len = 0;  // padded length (== input_ids.size())
};

class Qwen3Tokenizer {
public:
    Qwen3Tokenizer() = default;

    // Load from vocab.json + merges.txt (Qwen tokenizer directory).
    bool load(const std::string& vocab_path, const std::string& merges_path);
    bool is_loaded() const { return !vocab_.empty(); }

    // Apply the Qwen chat template (user turn + empty assistant "think" block),
    // tokenize, and right-pad to max_length with the pad token. Produces the
    // three int64 inputs expected by the Qwen3 text encoder ONNX.
    Qwen3Encoding encode_prompt(const std::string& prompt, int max_length = 256) const;

    // Tokenize a raw text segment (pre-tokenize + byte-level BPE) into ids.
    std::vector<int64_t> encode_segment(const std::string& text) const;

    int pad_token_id() const { return pad_token_id_; }

private:
    void init_byte_encoder();
    std::string byte_encode(const std::string& utf8_piece) const;
    std::string bpe(const std::string& piece) const;      // space-joined subtokens
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, int> merge_ranks_;
    std::map<uint8_t, char32_t> byte_encoder_;
    mutable std::unordered_map<std::string, std::string> bpe_cache_;

    // Special token ids (Qwen3).
    int im_start_id_    = 151644;  // <|im_start|>
    int im_end_id_      = 151645;  // <|im_end|>
    int think_open_id_  = 151667;  // <think>
    int think_close_id_ = 151668;  // </think>
    int pad_token_id_   = 151643;  // <|endoftext|>
};

} // namespace sd_npu
