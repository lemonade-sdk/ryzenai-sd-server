// sd15_text_encoder.cpp - SD1.5 text encoder implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variants/sd15_text_encoder.h"
#include <iostream>
#include <algorithm>

namespace sd_npu {

namespace {

// Create a [batch, seq] input_ids tensor using the element type expected by model
Ort::Value make_token_tensor(
    Ort::MemoryInfo&            mem,
    const std::vector<int64_t>& i64_tokens,
    std::vector<int32_t>&       i32_scratch,
    const std::vector<int64_t>& shape,
    OnnxModel*                  model,
    size_t                      input_idx = 0)
{
    if (model->get_input_type(input_idx) == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
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

// Convert an Ort::Value (float16 or float32) to a flat float32 vector
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


SD15TextEncoder::SD15TextEncoder(
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
    CLIPTokenizer& tokenizer,
    bool is_turbo)
    : components_(components),
      tokenizer_(tokenizer),
      is_turbo_(is_turbo) {

    // Cache CLIP output dimension (SD1.5=768, SD2.1/Turbo=1024)
    if (components_.count(ComponentType::TEXT_ENCODER)) {
        auto* model = components_[ComponentType::TEXT_ENCODER].get();
        if (model->get_num_outputs() > 0) {
            auto out_shape = model->get_output_shape(0);
            if (out_shape.size() >= 3 && out_shape[2] > 0)
                clip_dim_ = (int)out_shape[2];
            else if (out_shape.size() >= 2 && out_shape[1] > 0)
                clip_dim_ = (int)out_shape[1];
        }
    }
}

TextEncoderOutput SD15TextEncoder::encode(
    const std::string& prompt,
    const std::string& neg) {

    std::cout << "Encoding prompt with CLIP text encoder (SD1.5)..." << std::endl;

    const int seq = 77;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, seq};
    std::vector<const char*> in_names  = {"input_ids"};
    std::vector<const char*> out_names = {"last_hidden_state"};

    // Detect output embed dimension from the actual model output
    int dim = (clip_dim_ > 0) ? clip_dim_ : 768;

    auto run_clip = [&](const std::string& text) -> std::vector<float> {
        auto tokens_i64 = tokenizer_.encode(text, seq);
        auto model = components_[ComponentType::TEXT_ENCODER].get();
        std::vector<int32_t> tok_buf;
        std::vector<Ort::Value> inputs;
        inputs.push_back(make_token_tensor(mem, tokens_i64, tok_buf, shape, model));
        auto out = model->run(inputs, in_names, out_names);
        
        // If dim wasn't cached, read from first inference output and cache it
        if (clip_dim_ < 0) {
            auto out_shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
            if (out_shape.size() >= 3 && out_shape[2] > 0)
                dim = clip_dim_ = (int)out_shape[2];
            else if (out_shape.size() >= 2 && out_shape[1] > 0)
                dim = clip_dim_ = (int)out_shape[1];
        }
        return tensor_to_float32(out[0], seq * dim);
    };

    auto pos = run_clip(prompt);

    TextEncoderOutput result;
    
    if (is_turbo_) {
        // Turbo: batch=1, only conditional (positive) embeddings
        result.embeddings.resize(seq * dim);
        std::copy(pos.begin(), pos.end(), result.embeddings.begin());
        std::cout << "  prompt_embeds (SD-Turbo, no CFG): [1, 77, " << dim << "]" << std::endl;
    } else {
        // CFG batch: [uncond, cond] → [2, 77, dim]
        auto neg_emb = run_clip(neg.empty() ? "" : neg);
        result.embeddings.resize(2 * seq * dim);
        std::copy(neg_emb.begin(), neg_emb.end(), result.embeddings.begin());
        std::copy(pos.begin(), pos.end(), result.embeddings.begin() + seq * dim);
        std::cout << "  prompt_embeds: [2, 77, " << dim << "]" << std::endl;
    }
    
    return result;
}

} // namespace sd_npu
