// flux_text_encoder.cpp - FLUX text encoder implementation (CLIP-L)
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variants/flux_text_encoder.h"
#include <iostream>
#include <algorithm>

namespace sd_npu {

namespace {

Ort::Value make_token_tensor(
    Ort::MemoryInfo&            mem,
    const std::vector<int64_t>& i64_tokens,
    std::vector<int32_t>&       i32_scratch,
    const std::vector<int64_t>& shape,
    OnnxModel*                  model)
{
    if (model->get_input_type(0) == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        i32_scratch.resize(i64_tokens.size());
        std::transform(i64_tokens.begin(), i64_tokens.end(), i32_scratch.begin(),
                       [](int64_t v) { return static_cast<int32_t>(v); });
        return Ort::Value::CreateTensor<int32_t>(
            mem, i32_scratch.data(), i32_scratch.size(),
            shape.data(), shape.size());
    }
    return Ort::Value::CreateTensor<int64_t>(
        mem, const_cast<int64_t*>(i64_tokens.data()), i64_tokens.size(),
        shape.data(), shape.size());
}

std::vector<float> tensor_to_float32(Ort::Value& tensor, size_t count)
{
    std::vector<float> result(count);
    auto etype = tensor.GetTensorTypeAndShapeInfo().GetElementType();
    if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        auto* fp16 = tensor.GetTensorMutableData<Ort::Float16_t>();
        for (size_t i = 0; i < count; i++)
            result[i] = fp16[i].ToFloat();
    } else {
        float* fp32 = tensor.GetTensorMutableData<float>();
        std::copy(fp32, fp32 + count, result.begin());
    }
    return result;
}

} // anonymous namespace


FluxTextEncoder::FluxTextEncoder(
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
    CLIPTokenizer& tokenizer,
    bool is_schnell)
    : components_(components),
      tokenizer_(tokenizer),
      is_schnell_(is_schnell) {

    if (!components_.count(ComponentType::TEXT_ENCODER)) return;

    auto* model = components_[ComponentType::TEXT_ENCODER].get();
    const auto& out_names = model->get_output_names();

    // Detect hidden-state output name (penultimate or last_hidden_state)
    hidden_output_name_ = out_names.empty() ? "last_hidden_state" : out_names.back();
    for (const auto& n : out_names) {
        if (n == "last_hidden_state" || n.find("hidden") != std::string::npos) {
            hidden_output_name_ = n;
        }
    }

    // Detect pooled output name (pooler_output or text_embeds or second output)
    pooled_output_name_ = (out_names.size() > 1) ? out_names[1] : hidden_output_name_;
    for (const auto& n : out_names) {
        if (n == "pooler_output" || n == "text_embeds") {
            pooled_output_name_ = n;
            break;
        }
    }

    // Cache output dimension
    if (model->get_num_outputs() > 0) {
        auto shape = model->get_output_shape(0);
        if (shape.size() >= 3 && shape[2] > 0)
            clip_dim_ = static_cast<int>(shape[2]);
        else if (shape.size() >= 2 && shape[1] > 0)
            clip_dim_ = static_cast<int>(shape[1]);
    }
}

FluxTextEncoder::ClipOutput FluxTextEncoder::run_clip(const std::string& text) {
    const int seq = 77;
    int dim = (clip_dim_ > 0) ? clip_dim_ : 768;

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, seq};
    std::vector<const char*> in_names = {"input_ids"};

    auto* model = components_[ComponentType::TEXT_ENCODER].get();
    auto tokens = tokenizer_.encode(text, seq);

    std::vector<int32_t> tok_buf;
    std::vector<Ort::Value> inputs;
    inputs.push_back(make_token_tensor(mem, tokens, tok_buf, shape, model));

    ClipOutput result;

    bool has_separate_pooled = (pooled_output_name_ != hidden_output_name_);
    if (has_separate_pooled) {
        std::vector<const char*> out_names = {
            hidden_output_name_.c_str(),
            pooled_output_name_.c_str()
        };
        auto out = model->run(inputs, in_names, out_names);

        if (clip_dim_ < 0) {
            auto s = out[0].GetTensorTypeAndShapeInfo().GetShape();
            if (s.size() >= 3 && s[2] > 0) dim = clip_dim_ = static_cast<int>(s[2]);
        }

        result.hidden = tensor_to_float32(out[0], seq * dim);

        // Pooled output: [1, dim] or [1, 1, dim]
        auto pool_shape = out[1].GetTensorTypeAndShapeInfo().GetShape();
        size_t pool_n = 1;
        for (auto d : pool_shape) pool_n *= static_cast<size_t>(d > 0 ? d : 1);
        result.pooled = tensor_to_float32(out[1], pool_n);
        result.pooled.resize(dim, 0.0f);  // ensure exactly dim values
    } else {
        std::vector<const char*> out_names = {hidden_output_name_.c_str()};
        auto out = model->run(inputs, in_names, out_names);

        if (clip_dim_ < 0) {
            auto s = out[0].GetTensorTypeAndShapeInfo().GetShape();
            if (s.size() >= 3 && s[2] > 0) dim = clip_dim_ = static_cast<int>(s[2]);
        }

        result.hidden = tensor_to_float32(out[0], seq * dim);
        // Use position 0 (CLS/BOS token) as the pooled representation
        result.pooled.assign(result.hidden.begin(), result.hidden.begin() + dim);
    }

    return result;
}

TextEncoderOutput FluxTextEncoder::encode(
    const std::string& prompt,
    const std::string& neg) {

    std::cout << "Encoding prompt with CLIP-L text encoder (FLUX)..." << std::endl;

    const int seq = 77;
    int dim = (clip_dim_ > 0) ? clip_dim_ : 768;

    auto pos = run_clip(prompt);
    dim = static_cast<int>(pos.pooled.size());

    TextEncoderOutput result;

    if (is_schnell_) {
        // Schnell (distilled): no CFG, batch=1
        result.embeddings = pos.hidden;
        result.pooled_embeddings = pos.pooled;
        std::cout << "  FLUX prompt_embeds (schnell, no CFG): [1, " << seq << ", " << dim << "]" << std::endl;
    } else {
        // Standard FLUX: CFG, batch=2 → [uncond, cond]
        auto neg_out = run_clip(neg.empty() ? "" : neg);

        result.embeddings.resize(2 * seq * dim);
        std::copy(neg_out.hidden.begin(), neg_out.hidden.end(), result.embeddings.begin());
        std::copy(pos.hidden.begin(), pos.hidden.end(), result.embeddings.begin() + seq * dim);

        result.pooled_embeddings.resize(2 * dim);
        std::copy(neg_out.pooled.begin(), neg_out.pooled.end(), result.pooled_embeddings.begin());
        std::copy(pos.pooled.begin(), pos.pooled.end(), result.pooled_embeddings.begin() + dim);

        std::cout << "  FLUX prompt_embeds: [2, " << seq << ", " << dim << "]" << std::endl;
    }

    return result;
}

} // namespace sd_npu
