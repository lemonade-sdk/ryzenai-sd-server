// flux_text_encoder.cpp - FLUX text encoder implementation (CLIP-L)
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variants/flux_text_encoder.h"
#include "t5_tokenizer.h"
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
    bool is_schnell,
    std::unique_ptr<T5Tokenizer> t5_tokenizer,
    int t5_seq_len)
    : components_(components),
      tokenizer_(tokenizer),
      t5_tokenizer_(std::move(t5_tokenizer)),
      is_schnell_(is_schnell),
      t5_seq_len_(t5_seq_len) {

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

std::vector<float> FluxTextEncoder::run_t5(const std::string& text) {
    if (!t5_tokenizer_ || !components_.count(ComponentType::TEXT_ENCODER_2)) {
        return {};
    }

    auto* model = components_[ComponentType::TEXT_ENCODER_2].get();
    auto token_ids = t5_tokenizer_->encode(text, t5_seq_len_);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, static_cast<int64_t>(t5_seq_len_)};
    std::vector<const char*> in_names = {"input_ids"};
    std::vector<const char*> out_names = {"last_hidden_state"};

    // T5 model may expect int32 or int64 input_ids depending on the export.
    std::vector<int32_t> tok_buf;
    std::vector<Ort::Value> inputs;
    inputs.push_back(make_token_tensor(mem, token_ids, tok_buf, shape, model));

    auto outputs = model->run(inputs, in_names, out_names);

    // Output: [1, t5_seq_len, 4096]
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    size_t count = 1;
    for (auto d : info.GetShape()) count *= static_cast<size_t>(d > 0 ? d : 1);

    return tensor_to_float32(outputs[0], count);
}

TextEncoderOutput FluxTextEncoder::encode(
    const std::string& prompt,
    const std::string& neg) {

    const int clip_seq = 77;
    int dim = (clip_dim_ > 0) ? clip_dim_ : 768;

    auto clip_pos = run_clip(prompt);
    dim = static_cast<int>(clip_pos.pooled.size());

    // T5 embeddings: [t5_seq_len, 4096] if available, else zeros matching
    // the CLIP hidden state shape so the denoiser can still build a tensor.
    std::vector<float> t5_pos = run_t5(prompt);
    const bool has_t5 = !t5_pos.empty();
    const int t5_dim = 4096;

    TextEncoderOutput result;

    if (is_schnell_) {
        // No CFG — single batch.
        // encoder_hidden_states: T5 [1, t5_seq_len, 4096] or CLIP [1, 77, 768]
        if (has_t5) {
            result.embeddings = t5_pos;   // [t5_seq_len * 4096]
            std::cout << "  FLUX prompt_embeds (T5, schnell): [1, " << t5_seq_len_
                      << ", " << t5_dim << "]" << std::endl;
        } else {
            result.embeddings = clip_pos.hidden;
            std::cout << "  FLUX prompt_embeds (CLIP fallback, schnell): [1, "
                      << clip_seq << ", " << dim << "]" << std::endl;
        }
        result.pooled_embeddings = clip_pos.pooled;  // CLIP pooled always
    } else {
        // CFG — batch=2 [uncond, cond].
        auto clip_neg = run_clip(neg.empty() ? "" : neg);
        std::vector<float> t5_neg = run_t5(neg.empty() ? "" : neg);

        if (has_t5) {
            result.embeddings.resize(2 * static_cast<size_t>(t5_seq_len_) * t5_dim);
            std::copy(t5_neg.begin(), t5_neg.end(), result.embeddings.begin());
            std::copy(t5_pos.begin(), t5_pos.end(),
                      result.embeddings.begin() + t5_seq_len_ * t5_dim);
            std::cout << "  FLUX prompt_embeds (T5): [2, " << t5_seq_len_
                      << ", " << t5_dim << "]" << std::endl;
        } else {
            result.embeddings.resize(2 * clip_seq * dim);
            std::copy(clip_neg.hidden.begin(), clip_neg.hidden.end(),
                      result.embeddings.begin());
            std::copy(clip_pos.hidden.begin(), clip_pos.hidden.end(),
                      result.embeddings.begin() + clip_seq * dim);
            std::cout << "  FLUX prompt_embeds (CLIP fallback): [2, "
                      << clip_seq << ", " << dim << "]" << std::endl;
        }

        result.pooled_embeddings.resize(2 * static_cast<size_t>(dim));
        std::copy(clip_neg.pooled.begin(), clip_neg.pooled.end(),
                  result.pooled_embeddings.begin());
        std::copy(clip_pos.pooled.begin(), clip_pos.pooled.end(),
                  result.pooled_embeddings.begin() + dim);
    }

    return result;
}

} // namespace sd_npu
