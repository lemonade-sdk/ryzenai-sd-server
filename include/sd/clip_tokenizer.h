// clip_tokenizer.h - CLIP BPE Tokenizer for Stable Diffusion
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cstdint>

namespace sd_npu {

class CLIPTokenizer {
public:
    CLIPTokenizer() = default;

    // Load tokenizer from vocab.json and merges.txt
    bool load(const std::string& vocab_path, const std::string& merges_path);

    // Tokenize text -> padded/truncated token IDs [max_length]
    std::vector<int64_t> encode(const std::string& text, int max_length = 77) const;

    bool is_loaded() const { return !vocab_.empty(); }

    int bos_token_id() const { return bos_token_id_; }
    int eos_token_id() const { return eos_token_id_; }
    int pad_token_id() const { return pad_token_id_; }

private:
    // Text preprocessing
    static std::string whitespace_clean(const std::string& text);
    static std::string to_lower(const std::string& text);

    // Byte-level encoding (OpenAI bytes_to_unicode)
    void init_byte_encoder();
    std::string byte_encode_token(const std::string& token) const;

    // BPE algorithm
    std::string bpe(const std::string& token) const;

    // Word splitting (simplified CLIP regex)
    std::vector<std::string> split_words(const std::string& text) const;

    // Data
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, int> merge_ranks_;
    std::map<uint8_t, char32_t> byte_encoder_;

    int bos_token_id_ = 49406;  // <|startoftext|>
    int eos_token_id_ = 49407;  // <|endoftext|>
    int pad_token_id_ = 49407;  // default: pad with eos

    mutable std::unordered_map<std::string, std::string> bpe_cache_;
};

} // namespace sd_npu
