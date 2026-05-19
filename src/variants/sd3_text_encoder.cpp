// sd3_text_encoder.cpp - SD3/SD3.5 text encoder implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variants/sd3_text_encoder.h"
#include <iostream>
#include <algorithm>
#include <fstream>

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


SD3TextEncoder::SD3TextEncoder(
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
    CLIPTokenizer& tokenizer,
    CLIPTokenizer& tokenizer_2,
    const std::string& t5_vocab_path)
    : components_(components),
      tokenizer_(tokenizer),
      tokenizer_2_(tokenizer_2) {

    read_transformer_dims();
    
    if (!t5_vocab_path.empty()) {
        load_t5_vocab(t5_vocab_path);
    }
}

void SD3TextEncoder::read_transformer_dims() {
    if (!components_.count(ComponentType::TRANSFORMER)) return;
    auto& tmodel = components_[ComponentType::TRANSFORMER];
    auto& names = tmodel->get_input_names();
    for (size_t i = 0; i < names.size(); i++) {
        if (names[i] == "encoder_hidden_states") {
            auto shape = tmodel->get_input_shape(i);
            if (shape.size() >= 3) {
                if (shape[1] > 0) total_seq_len_ = (int)shape[1];
                if (shape[2] > 0) embed_dim_ = (int)shape[2];
            }
        }
    }
}

void SD3TextEncoder::load_t5_vocab(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cout << "  WARNING: T5 tokenizer.json not found: " << path << std::endl;
        return;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Locate the Unigram "vocab" array in the "model" section
    size_t model_pos = content.find("\"Unigram\"");
    if (model_pos == std::string::npos) model_pos = 0;

    size_t vocab_pos = content.find("\"vocab\"", model_pos);
    if (vocab_pos == std::string::npos) {
        std::cout << "  WARNING: No vocab found in T5 tokenizer.json" << std::endl;
        return;
    }

    size_t arr_start = content.find('[', vocab_pos + 7);
    if (arr_start == std::string::npos) return;

    t5_vocab_.clear();
    int64_t id = 0;
    size_t pos = arr_start + 1;

    while (pos < content.size()) {
        // Skip whitespace
        while (pos < content.size() && std::isspace((unsigned char)content[pos])) pos++;
        if (pos >= content.size() || content[pos] != '[') break;

        // Find the piece string
        size_t q1 = content.find('"', pos + 1);
        if (q1 == std::string::npos) break;

        // Parse JSON string (handle \n, \t, \uXXXX, etc.)
        std::string piece;
        size_t p = q1 + 1;
        while (p < content.size() && content[p] != '"') {
            if (content[p] == '\\' && p + 1 < content.size()) {
                char esc = content[p + 1];
                if      (esc == 'n')  { piece += '\n'; p += 2; }
                else if (esc == 't')  { piece += '\t'; p += 2; }
                else if (esc == '"')  { piece += '"';  p += 2; }
                else if (esc == '\\') { piece += '\\'; p += 2; }
                else if (esc == 'u' && p + 5 < content.size()) {
                    std::string hex = content.substr(p + 2, 4);
                    try {
                        unsigned int cp = std::stoul(hex, nullptr, 16);
                        if      (cp < 0x80)  { piece += (char)cp; }
                        else if (cp < 0x800) { piece += (char)(0xC0|(cp>>6)); piece += (char)(0x80|(cp&0x3F)); }
                        else                 { piece += (char)(0xE0|(cp>>12)); piece += (char)(0x80|((cp>>6)&0x3F)); piece += (char)(0x80|(cp&0x3F)); }
                    } catch (...) {}
                    p += 6;
                } else { piece += content[p + 1]; p += 2; }
            } else {
                piece += content[p++];
            }
        }

        t5_vocab_[piece] = id++;

        // Advance past the closing ']' of this entry
        size_t entry_end = content.find(']', p);
        if (entry_end == std::string::npos) break;
        pos = entry_end + 1;
        while (pos < content.size() && (std::isspace((unsigned char)content[pos]) || content[pos] == ',')) pos++;
    }

    // Read unk_id from "model" section if present
    size_t unk_pos = content.find("\"unk_id\"", model_pos);
    if (unk_pos != std::string::npos) {
        size_t colon = content.find(':', unk_pos + 8);
        if (colon != std::string::npos) {
            try { t5_unk_id_ = std::stoll(content.substr(colon + 1, 10)); } catch (...) {}
        }
    }

    std::cout << "  T5 vocab: " << t5_vocab_.size() << " tokens from " << path << std::endl;
}

// Greedy longest-prefix SentencePiece tokenization using the loaded T5 vocabulary
std::vector<int64_t> SD3TextEncoder::tokenize_t5(const std::string& text, int max_length) const {
    static const int64_t EOS_ID = 1;
    static const int64_t PAD_ID = 0;
    // U+2581 LOWER ONE EIGHTH BLOCK — used as space prefix in SentencePiece
    static const std::string SPACE_PIECE = "\xe2\x96\x81";

    std::vector<int64_t> ids;
    ids.reserve(max_length);

    if (t5_vocab_.empty()) {
        ids.resize(max_length, PAD_ID);
        return ids;
    }

    // Split on whitespace
    std::vector<std::string> words;
    std::string word;
    for (unsigned char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!word.empty()) { words.push_back(word); word.clear(); }
        } else {
            word += (char)c;
        }
    }
    if (!word.empty()) words.push_back(word);

    // Greedy longest-prefix tokenize each word (with ▁ prefix)
    for (const auto& w : words) {
        std::string piece_text = SPACE_PIECE + w;
        size_t p = 0;
        while (p < piece_text.size() && (int)ids.size() < max_length - 1) {
            size_t best_len = 0;
            int64_t best_id  = -1;
            size_t max_try   = std::min(piece_text.size() - p, (size_t)32);

            for (size_t len = max_try; len >= 1; len--) {
                // Skip lengths that cut mid-UTF-8 sequence
                if (p + len < piece_text.size()) {
                    unsigned char nb = (unsigned char)piece_text[p + len];
                    if ((nb & 0xC0) == 0x80) continue;
                }
                auto it = t5_vocab_.find(piece_text.substr(p, len));
                if (it != t5_vocab_.end()) { best_len = len; best_id = it->second; break; }
            }

            if (best_id >= 0) {
                ids.push_back(best_id);
                p += best_len;
            } else {
                ids.push_back(t5_unk_id_);
                // Advance one UTF-8 character
                unsigned char byte = (unsigned char)piece_text[p++];
                if ((byte & 0xE0) == 0xE0 && p < piece_text.size()) p++;  // 3-byte seq
                if ((byte & 0xC0) == 0xC0 && p < piece_text.size()) p++;  // 2-byte seq
            }
        }
    }

    if ((int)ids.size() < max_length) ids.push_back(EOS_ID);
    ids.resize(max_length, PAD_ID);
    return ids;
}

void SD3TextEncoder::encode_clip_pair(
    const std::string& text,
    std::vector<float>& clip_hidden,
    std::vector<float>& clip_pooled) {

    const int clip_seq = 77, d1 = 768, d2 = 1280;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, clip_seq};

    // CLIP-L
    auto& te1 = components_[ComponentType::TEXT_ENCODER];
    auto& te1_names = te1->get_output_names();
    std::string clip_l_hidden_name = te1_names[te1_names.size() - 2];
    std::string clip_l_pooled_name = te1_names.size() > 1 ? te1_names[1] : te1_names[0];
    for (auto& n : te1_names) { if (n == "pooler_output") { clip_l_pooled_name = n; break; } }

    auto tokens_1_i64 = tokenizer_.encode(text, clip_seq);
    std::vector<int32_t> tok1_buf;
    std::vector<Ort::Value> te1_in;
    te1_in.push_back(make_token_tensor(mem, tokens_1_i64, tok1_buf, shape, te1.get()));
    std::vector<const char*> in1  = {"input_ids"};
    std::vector<const char*> out1 = {clip_l_hidden_name.c_str(), clip_l_pooled_name.c_str()};
    auto te1_out = te1->run(te1_in, in1, out1);
    auto h1 = tensor_to_float32(te1_out[0], clip_seq * d1);
    auto p1 = tensor_to_float32(te1_out[1], d1);

    // CLIP-G
    auto& te2 = components_[ComponentType::TEXT_ENCODER_2];
    auto& te2_names = te2->get_output_names();
    std::string clip_g_hidden_name = te2_names[te2_names.size() - 2];
    std::string clip_g_pooled_name = te2_names[0];

    auto tokens_2_i64 = tokenizer_2_.encode(text, clip_seq);
    std::vector<int32_t> tok2_buf;
    std::vector<Ort::Value> te2_in;
    te2_in.push_back(make_token_tensor(mem, tokens_2_i64, tok2_buf, shape, te2.get()));
    std::vector<const char*> in2  = {"input_ids"};
    std::vector<const char*> out2 = {clip_g_hidden_name.c_str(), clip_g_pooled_name.c_str()};
    auto te2_out = te2->run(te2_in, in2, out2);
    auto h2 = tensor_to_float32(te2_out[0], clip_seq * d2);
    auto p2 = tensor_to_float32(te2_out[1], d2);

    // Concat hidden: [77, 768] + [77, 1280] → [77, 2048]
    clip_hidden.resize(clip_seq * (d1 + d2));
    for (int t = 0; t < clip_seq; t++) {
        std::copy(h1.data() + t*d1, h1.data() + (t+1)*d1,
                  clip_hidden.data() + t*(d1+d2));
        std::copy(h2.data() + t*d2, h2.data() + (t+1)*d2,
                  clip_hidden.data() + t*(d1+d2) + d1);
    }

    // Concat pooled: [768] + [1280] → [2048]
    clip_pooled.resize(d1 + d2);
    std::copy(p1.begin(), p1.end(), clip_pooled.begin());
    std::copy(p2.begin(), p2.end(), clip_pooled.begin() + d1);
}

std::vector<float> SD3TextEncoder::encode_t5(const std::string& text, int max_length) {
    if (!components_.count(ComponentType::TEXT_ENCODER_3)) {
        // No T5 encoder available, return zeros
        return std::vector<float>(max_length * embed_dim_, 0.0f);
    }

    // Tokenize using the loaded SentencePiece vocabulary
    auto token_ids = tokenize_t5(text, max_length);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto& te3 = components_[ComponentType::TEXT_ENCODER_3];
    std::vector<int64_t> shape = {1, (int64_t)max_length};
    std::vector<const char*> in_names  = {"input_ids"};
    std::vector<const char*> out_names = {"last_hidden_state"};

    std::vector<Ort::Value> inputs;
    std::vector<int32_t> ids_i32;
    inputs.push_back(make_token_tensor(mem, token_ids, ids_i32, shape, te3.get(), 0));

    auto out = te3->run(inputs, in_names, out_names);

    size_t total = (size_t)max_length * embed_dim_;
    std::vector<float> embeddings(total);
    auto etype = out[0].GetTensorTypeAndShapeInfo().GetElementType();
    if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        auto* fp16 = out[0].GetTensorMutableData<Ort::Float16_t>();
        for (size_t i = 0; i < total; i++) embeddings[i] = fp16[i].ToFloat();
    } else {
        auto* fp32 = out[0].GetTensorMutableData<float>();
        std::copy(fp32, fp32 + total, embeddings.begin());
    }
    return embeddings;
}

TextEncoderOutput SD3TextEncoder::encode(
    const std::string& prompt,
    const std::string& neg) {

    std::cout << "SD3: Encoding prompt with CLIP-L + CLIP-G + T5 text encoders..." << std::endl;

    const int clip_seq = 77;
    const int total_seq = total_seq_len_;
    const int embed_dim = embed_dim_;
    const int t5_seq = total_seq - clip_seq;

    // Encode positive and negative prompts with CLIP
    std::vector<float> pos_clip_h, pos_clip_p, neg_clip_h, neg_clip_p;
    encode_clip_pair(prompt, pos_clip_h, pos_clip_p);
    encode_clip_pair(neg.empty() ? "" : neg, neg_clip_h, neg_clip_p);

    std::cout << "  CLIP-L + CLIP-G done." << std::endl;

    // Encode with T5
    bool use_t5 = components_.count(ComponentType::TEXT_ENCODER_3) > 0;
    std::cout << "  Encoding with T5: " << (use_t5 ? "yes" : "no (using zeros)") << std::endl;
    
    std::vector<float> pos_t5_h, neg_t5_h;
    if (use_t5) {
        pos_t5_h = encode_t5(prompt, t5_seq);
        neg_t5_h = encode_t5(neg.empty() ? "" : neg, t5_seq);
    } else {
        pos_t5_h.resize(t5_seq * embed_dim, 0.0f);
        neg_t5_h.resize(t5_seq * embed_dim, 0.0f);
    }

    std::cout << "  Building final embeddings..." << std::endl;
    std::cout << "  total_seq_len=" << total_seq << " embed_dim=" << embed_dim
              << " clip_seq=" << clip_seq << " t5_seq=" << t5_seq << std::endl;

    // Build final embeddings: CLIP [77, 2048] → [77, 4096], then append T5 [t5_seq, 4096]
    auto build_final = [&](const std::vector<float>& clip_hidden,
                           const std::vector<float>& t5_hidden) -> std::vector<float> {
        std::vector<float> result(total_seq * embed_dim, 0.0f);
        
        // Copy CLIP embeddings [77, 2048] into first 2048 dims of [77, 4096]
        for (int t = 0; t < clip_seq; t++) {
            std::copy(clip_hidden.data() + t * 2048,
                      clip_hidden.data() + (t + 1) * 2048,
                      result.data() + t * embed_dim);
        }
        
        // Copy T5 embeddings [t5_seq, 4096]
        std::copy(t5_hidden.begin(), t5_hidden.end(),
                  result.data() + clip_seq * embed_dim);
        
        return result;
    };

    auto pos_embeds = build_final(pos_clip_h, pos_t5_h);
    auto neg_embeds = build_final(neg_clip_h, neg_t5_h);

    TextEncoderOutput result;
    // Embeddings: [2, total_seq, embed_dim]
    result.embeddings.resize(2 * total_seq * embed_dim);
    std::copy(neg_embeds.begin(), neg_embeds.end(), result.embeddings.begin());
    std::copy(pos_embeds.begin(), pos_embeds.end(),
              result.embeddings.begin() + total_seq * embed_dim);

    // Pooled: [2, 2048]
    result.pooled_embeddings.resize(2 * 2048);
    std::copy(neg_clip_p.begin(), neg_clip_p.end(), result.pooled_embeddings.begin());
    std::copy(pos_clip_p.begin(), pos_clip_p.end(), result.pooled_embeddings.begin() + 2048);

    std::cout << "  prompt_embeds: [2, " << total_seq << ", " << embed_dim
              << "]  pooled: [2, 2048]" << std::endl;
    return result;
}

} // namespace sd_npu
