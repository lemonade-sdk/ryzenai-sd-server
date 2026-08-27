// t5_tokenizer.h - T5/SentencePiece tokenizer wrapper
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <string>
#include <vector>
#include <memory>

namespace sentencepiece { class SentencePieceProcessor; }

namespace sd_npu {

// Wraps the SentencePiece model used by the T5-XXL text encoder in FLUX.
// Encodes a string into token IDs padded/truncated to max_length.
// The T5 tokenizer appends EOS (id=1) at the end of each sequence.
class T5Tokenizer {
public:
    // Load from the spiece.model file inside the tokenizer_2/ directory.
    // Throws std::runtime_error if the model cannot be loaded.
    explicit T5Tokenizer(const std::string& spiece_model_path);
    ~T5Tokenizer();

    // Encode text → token IDs, padded with 0 to max_length.
    // If the text encodes to more than max_length tokens, it is truncated
    // (keeping the EOS token at the end).
    std::vector<int64_t> encode(const std::string& text, int max_length) const;

    // Vocabulary size
    int vocab_size() const;

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
};

} // namespace sd_npu
