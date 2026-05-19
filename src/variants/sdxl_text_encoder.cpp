// sdxl_text_encoder.cpp - SDXL text encoder implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variants/sdxl_text_encoder.h"
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


SDXLTextEncoder::SDXLTextEncoder(
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
    CLIPTokenizer& tokenizer,
    CLIPTokenizer& tokenizer_2,
    bool is_turbo)
    : components_(components),
      tokenizer_(tokenizer),
      tokenizer_2_(tokenizer_2),
      is_turbo_(is_turbo) {}

TextEncoderOutput SDXLTextEncoder::encode(
    const std::string& prompt,
    const std::string& neg) {

    std::cout << "Encoding prompt with CLIP text encoders (SDXL)..." << std::endl;

    const int seq = 77, d1 = 768, d2 = 1280, d = d1 + d2;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, seq};

    auto run_sdxl = [&](const std::string& text,
                        std::vector<float>& hidden_out,
                        std::vector<float>& pooled_out) {
        auto tok1_i64 = tokenizer_.encode(text, seq);
        auto tok2_i64 = tokenizer_2_.encode(text, seq);

        // CLIP-L → hidden_states.11 [1,77,768]
        auto te1 = components_[ComponentType::TEXT_ENCODER].get();
        std::vector<int32_t> tok1_buf;
        std::vector<Ort::Value> in1;
        in1.push_back(make_token_tensor(mem, tok1_i64, tok1_buf, shape, te1));
        std::vector<const char*> in_n = {"input_ids"};
        std::vector<const char*> out1 = {"hidden_states.11"};
        auto te1_out = te1->run(in1, in_n, out1);
        auto h1_buf = tensor_to_float32(te1_out[0], seq * d1);

        // CLIP-G → hidden_states.31 [1,77,1280], text_embeds [1,1280]
        auto te2 = components_[ComponentType::TEXT_ENCODER_2].get();
        std::vector<int32_t> tok2_buf;
        std::vector<Ort::Value> in2;
        in2.push_back(make_token_tensor(mem, tok2_i64, tok2_buf, shape, te2));
        std::vector<const char*> out2 = {"hidden_states.31", "text_embeds"};
        auto te2_out = te2->run(in2, in_n, out2);
        auto h2_buf     = tensor_to_float32(te2_out[0], seq * d2);
        auto pooled_buf = tensor_to_float32(te2_out[1], d2);

        // Concat on embed dim → [77, 2048]
        hidden_out.resize(seq * d);
        for (int t = 0; t < seq; t++) {
            std::copy(h1_buf.data() + t*d1, h1_buf.data() + (t+1)*d1, hidden_out.data() + t*d);
            std::copy(h2_buf.data() + t*d2, h2_buf.data() + (t+1)*d2, hidden_out.data() + t*d + d1);
        }
        pooled_out.assign(pooled_buf.begin(), pooled_buf.end());
    };

    std::vector<float> pos_h, pos_p, neg_h, neg_p;
    run_sdxl(prompt, pos_h, pos_p);
    run_sdxl(neg.empty() ? "" : neg, neg_h, neg_p);

    TextEncoderOutput result;
    
    if (is_turbo_) {
        // Turbo: batch=1, only conditional (positive) embeddings
        result.embeddings.resize(seq * d);
        std::copy(pos_h.begin(), pos_h.end(), result.embeddings.begin());
        result.pooled_embeddings.resize(d2);
        std::copy(pos_p.begin(), pos_p.end(), result.pooled_embeddings.begin());
        std::cout << "  prompt_embeds (SDXL Turbo, no CFG): [1, 77, 2048]  pooled: [1, 1280]" << std::endl;
    } else {
        // Embeddings: [2, 77, 2048]
        result.embeddings.resize(2 * seq * d);
        std::copy(neg_h.begin(), neg_h.end(), result.embeddings.begin());
        std::copy(pos_h.begin(), pos_h.end(), result.embeddings.begin() + seq * d);

        // Pooled: [2, 1280]
        result.pooled_embeddings.resize(2 * d2);
        std::copy(neg_p.begin(), neg_p.end(), result.pooled_embeddings.begin());
        std::copy(pos_p.begin(), pos_p.end(), result.pooled_embeddings.begin() + d2);
        std::cout << "  prompt_embeds: [2, 77, 2048]  pooled: [2, 1280]" << std::endl;
    }
    return result;
}

} // namespace sd_npu
