// t5_tokenizer.cpp - T5/SentencePiece tokenizer wrapper
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "t5_tokenizer.h"
#include <sentencepiece_processor.h>
#include <stdexcept>
#include <algorithm>

namespace sd_npu {

T5Tokenizer::T5Tokenizer(const std::string& spiece_model_path) {
    sp_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    auto status = sp_->Load(spiece_model_path);
    if (!status.ok()) {
        throw std::runtime_error(
            "T5Tokenizer: failed to load SentencePiece model from '" +
            spiece_model_path + "': " + status.ToString());
    }
}

T5Tokenizer::~T5Tokenizer() = default;

std::vector<int64_t> T5Tokenizer::encode(const std::string& text, int max_length) const {
    std::vector<int> ids;
    sp_->Encode(text, &ids);

    // T5 appends EOS (id=1). SentencePiece should add it automatically, but
    // ensure it's present for compatibility with the ONNX model.
    const int eos_id = sp_->eos_id();  // typically 1
    if (ids.empty() || ids.back() != eos_id) {
        ids.push_back(eos_id);
    }

    // Truncate to max_length, keeping EOS at the end
    if (static_cast<int>(ids.size()) > max_length) {
        ids.resize(static_cast<size_t>(max_length) - 1);
        ids.push_back(eos_id);
    }

    // Pad to max_length with 0
    std::vector<int64_t> result(static_cast<size_t>(max_length), 0);
    for (size_t i = 0; i < ids.size(); i++) {
        result[i] = static_cast<int64_t>(ids[i]);
    }
    return result;
}

int T5Tokenizer::vocab_size() const {
    return sp_->GetPieceSize();
}

} // namespace sd_npu
