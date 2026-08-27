// flux2_text_encoder.cpp - FLUX.2-klein text encoder (Qwen3) implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variants/flux2_text_encoder.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace sd_npu {

namespace {

// Build an int32/int64 token tensor matching the model's declared input dtype.
Ort::Value make_id_tensor(
    Ort::MemoryInfo&            mem,
    const std::vector<int64_t>& i64,
    std::vector<int32_t>&       i32_scratch,
    const std::vector<int64_t>& shape,
    ONNXTensorElementDataType   dt)
{
    if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        i32_scratch.resize(i64.size());
        std::transform(i64.begin(), i64.end(), i32_scratch.begin(),
                       [](int64_t v) { return static_cast<int32_t>(v); });
        return Ort::Value::CreateTensor<int32_t>(
            mem, i32_scratch.data(), i32_scratch.size(),
            shape.data(), shape.size());
    }
    return Ort::Value::CreateTensor<int64_t>(
        mem, const_cast<int64_t*>(i64.data()), i64.size(),
        shape.data(), shape.size());
}

} // anonymous namespace

Flux2TextEncoder::Flux2TextEncoder(
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
    const std::string& model_root,
    int max_sequence_length)
    : components_(components),
      max_seq_len_(max_sequence_length) {

    fs::path root(model_root);
    fs::path vocab  = root / "tokenizer" / "vocab.json";
    fs::path merges = root / "tokenizer" / "merges.txt";

    if (fs::exists(vocab) && fs::exists(merges)) {
        tokenizer_loaded_ = tokenizer_.load(vocab.string(), merges.string());
    }
    if (!tokenizer_loaded_) {
        std::cerr << "[FLUX2] Warning: Qwen3 tokenizer not loaded from "
                  << (root / "tokenizer") << std::endl;
    } else {
        std::cout << "[FLUX2] Qwen3 tokenizer loaded from: "
                  << (root / "tokenizer") << std::endl;
    }

    // Cache the encoder hidden dimension from the ONNX output shape.
    if (components_.count(ComponentType::TEXT_ENCODER)) {
        auto* model = components_[ComponentType::TEXT_ENCODER].get();
        if (model->get_num_outputs() > 0) {
            auto shape = model->get_output_shape(0);  // [B, seq, hidden]
            if (shape.size() >= 3 && shape[2] > 0)
                embed_dim_ = static_cast<int>(shape[2]);
        }
    }
}

TextEncoderOutput Flux2TextEncoder::encode(
    const std::string& prompt,
    const std::string& /*negative_prompt*/) {

    TextEncoderOutput out;

    if (!components_.count(ComponentType::TEXT_ENCODER)) {
        throw std::runtime_error("Flux2TextEncoder: TEXT_ENCODER component not loaded");
    }
    if (!tokenizer_loaded_) {
        throw std::runtime_error("Flux2TextEncoder: Qwen3 tokenizer not loaded");
    }

    auto* model = components_[ComponentType::TEXT_ENCODER].get();

    // 1. Tokenize with chat template, padded to max_seq_len_.
    Qwen3Encoding enc = tokenizer_.encode_prompt(prompt, max_seq_len_);
    const int seq = enc.seq_len;

    // 2. Build the three int64 inputs: input_ids, attention_mask, position_ids.
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, seq};

    const auto& in_names_str = model->get_input_names();
    std::vector<const char*> in_names;
    for (const auto& n : in_names_str) in_names.push_back(n.c_str());

    // Map named inputs to encoding fields (robust to input ordering).
    std::vector<int32_t> s0, s1, s2;
    std::vector<Ort::Value> inputs;
    inputs.reserve(in_names_str.size());
    for (size_t i = 0; i < in_names_str.size(); i++) {
        const std::string& name = in_names_str[i];
        ONNXTensorElementDataType dt = model->get_input_type(i);
        if (name.find("attention") != std::string::npos) {
            inputs.push_back(make_id_tensor(mem, enc.attention_mask, s1, shape, dt));
        } else if (name.find("position") != std::string::npos) {
            inputs.push_back(make_id_tensor(mem, enc.position_ids, s2, shape, dt));
        } else {
            // input_ids (default / first)
            inputs.push_back(make_id_tensor(mem, enc.input_ids, s0, shape, dt));
        }
    }

    const auto& out_names_str = model->get_output_names();
    std::vector<const char*> out_names;
    for (const auto& n : out_names_str) out_names.push_back(n.c_str());

    auto t0 = std::chrono::high_resolution_clock::now();
    auto outputs = model->run(inputs, in_names, out_names);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "  Qwen3 text encoder: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms" << std::endl;

    // 3. Extract embeddings [1, seq, hidden] -> f32.
    auto info  = outputs[0].GetTensorTypeAndShapeInfo();
    auto oshape = info.GetShape();
    size_t count = 1;
    for (auto d : oshape) count *= static_cast<size_t>(d > 0 ? d : 1);

    if (oshape.size() >= 3 && oshape[2] > 0)
        embed_dim_ = static_cast<int>(oshape[2]);

    out.embeddings.resize(count);
    auto etype = info.GetElementType();
    if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        auto* fp16 = outputs[0].GetTensorMutableData<Ort::Float16_t>();
        for (size_t i = 0; i < count; i++) out.embeddings[i] = fp16[i].ToFloat();
    } else {
        float* fp32 = outputs[0].GetTensorMutableData<float>();
        std::copy(fp32, fp32 + count, out.embeddings.begin());
    }

    // klein transformer has no pooled projection input.
    out.pooled_embeddings.clear();
    return out;
}

} // namespace sd_npu
