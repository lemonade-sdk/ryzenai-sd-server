// flux2_denoiser.cpp - FLUX.2-klein transformer denoiser implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variants/flux2_denoiser.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace sd_npu {

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Emit a tensor of the model's declared dtype from an f32 source buffer.
// Scratch buffers are kept alive by the caller until run() completes.
struct Scratch {
    std::vector<std::vector<float>>         f32;
    std::vector<std::vector<Ort::Float16_t>> f16;
    std::vector<std::vector<int64_t>>       i64;
    std::vector<std::vector<int32_t>>       i32;
    void clear() { f32.clear(); f16.clear(); i64.clear(); i32.clear(); }
};

Ort::Value emit(Ort::MemoryInfo& mem, Scratch& sc,
                const std::vector<float>& src,
                const std::vector<int64_t>& shape,
                ONNXTensorElementDataType dt) {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d > 0 ? d : 1);
    switch (dt) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
        sc.f16.emplace_back(n);
        auto& b = sc.f16.back();
        for (size_t i = 0; i < n; i++) b[i] = Ort::Float16_t(i < src.size() ? src[i] : 0.0f);
        return Ort::Value::CreateTensor<Ort::Float16_t>(mem, b.data(), n, shape.data(), shape.size());
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
        sc.i64.emplace_back(n);
        auto& b = sc.i64.back();
        for (size_t i = 0; i < n; i++) b[i] = static_cast<int64_t>(i < src.size() ? src[i] : 0.0f);
        return Ort::Value::CreateTensor<int64_t>(mem, b.data(), n, shape.data(), shape.size());
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
        sc.i32.emplace_back(n);
        auto& b = sc.i32.back();
        for (size_t i = 0; i < n; i++) b[i] = static_cast<int32_t>(i < src.size() ? src[i] : 0.0f);
        return Ort::Value::CreateTensor<int32_t>(mem, b.data(), n, shape.data(), shape.size());
    }
    default: {  // float32
        sc.f32.emplace_back(n, 0.0f);
        auto& b = sc.f32.back();
        std::copy_n(src.begin(), std::min(n, src.size()), b.begin());
        return Ort::Value::CreateTensor<float>(mem, b.data(), n, shape.data(), shape.size());
    }
    }
}

} // anonymous namespace

Flux2Denoiser::Flux2Denoiser(
    const std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
    const std::string& model_root,
    int text_seq_len, int text_embed_dim)
    : text_seq_len_(text_seq_len), text_embed_dim_(text_embed_dim) {
    auto it = components.find(ComponentType::TRANSFORMER);
    if (it == components.end()) {
        throw std::runtime_error("Flux2Denoiser: TRANSFORMER component not found");
    }
    model_ = it->second.get();
    load_bn_stats(model_root);
}

// Parse bn.running_x.safetensors (keys "bn.running_mean", "bn.running_var",
// both BF16 [128]) and precompute bn_mean_ / bn_std_ = sqrt(var + eps).
void Flux2Denoiser::load_bn_stats(const std::string& model_root) {
    const float batch_norm_eps = 1e-4f;
    fs::path root(model_root);
    fs::path st = root / "bn.running_x.safetensors";
    if (!fs::exists(st)) {
        // Also try the vae_decoder subdirectory.
        fs::path alt = root / "vae_decoder" / "bn.running_x.safetensors";
        if (fs::exists(alt)) st = alt; else {
            std::cerr << "[FLUX2] Warning: bn.running_x.safetensors not found under "
                      << model_root << " (skipping BatchNorm denorm)" << std::endl;
            return;
        }
    }

    std::ifstream f(st, std::ios::binary);
    if (!f) { std::cerr << "[FLUX2] Warning: cannot open " << st << std::endl; return; }

    uint64_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 8);
    if (!f || header_len == 0 || header_len > (1u << 20)) {
        std::cerr << "[FLUX2] Warning: bad safetensors header" << std::endl; return;
    }
    std::string header(header_len, '\0');
    f.read(header.data(), static_cast<std::streamsize>(header_len));

    // Read the entire data blob following the header.
    std::vector<char> blob((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());

    // Minimal JSON scan for each tensor's dtype/offsets. safetensors metadata:
    // "name":{"dtype":"BF16","shape":[128],"data_offsets":[start,end]}
    auto parse_bf16 = [&](const std::string& key) -> std::vector<float> {
        std::vector<float> out;
        size_t kpos = header.find("\"" + key + "\"");
        if (kpos == std::string::npos) return out;
        size_t opos = header.find("data_offsets", kpos);
        if (opos == std::string::npos) return out;
        size_t lb = header.find('[', opos);
        size_t rb = header.find(']', lb);
        if (lb == std::string::npos || rb == std::string::npos) return out;
        std::string nums = header.substr(lb + 1, rb - lb - 1);
        long long start = 0, end = 0;
        std::sscanf(nums.c_str(), " %lld , %lld", &start, &end);
        if (end <= start) return out;
        size_t count = static_cast<size_t>(end - start) / 2;  // BF16 = 2 bytes
        out.resize(count);
        for (size_t i = 0; i < count; i++) {
            uint16_t bf = 0;
            std::memcpy(&bf, blob.data() + start + i * 2, 2);
            uint32_t bits = static_cast<uint32_t>(bf) << 16;  // BF16 -> F32
            float v; std::memcpy(&v, &bits, 4);
            out[i] = v;
        }
        return out;
    };

    std::vector<float> mean = parse_bf16("bn.running_mean");
    std::vector<float> var  = parse_bf16("bn.running_var");
    if (mean.size() != 128 || var.size() != 128) {
        std::cerr << "[FLUX2] Warning: bn stats size mismatch (mean=" << mean.size()
                  << ", var=" << var.size() << "); skipping denorm" << std::endl;
        return;
    }
    bn_mean_ = std::move(mean);
    bn_std_.resize(128);
    for (int i = 0; i < 128; i++) bn_std_[i] = std::sqrt(var[i] + batch_norm_eps);
    std::cout << "[FLUX2] Loaded BatchNorm denorm stats from: " << st << std::endl;
}

// Empirical resolution/step-dependent shift parameter (reference
// compute_empirical_mu). Determines the exponential time shift applied to the
// flow-match sigma schedule.
double Flux2Denoiser::compute_empirical_mu(int image_seq_len, int num_steps) {
    const double a1 = 8.73809524e-05, b1 = 1.89833333;
    const double a2 = 0.00016927,     b2 = 0.45666666;
    if (image_seq_len > 4300) {
        return a2 * image_seq_len + b2;
    }
    double m_200 = a2 * image_seq_len + b2;
    double m_10  = a1 * image_seq_len + b1;
    double a = (m_200 - m_10) / 190.0;
    double b = m_200 - 200.0 * a;
    return a * num_steps + b;
}

std::vector<float> Flux2Denoiser::denoise(
    const std::vector<float>& latents,
    const std::vector<float>& text_embeddings,
    const std::vector<float>& /*pooled_embeddings*/,
    Scheduler& /*scheduler*/,
    const SDConfig& config,
    const std::vector<float>& /*controlnet_cond*/) {

    // Packed latent grid: latH = latW = height/16 (2x2 patchify of the H/8 VAE grid).
    const int latH = config.height / 16;
    const int latW = config.width  / 16;
    const int packed_ch = 128;                 // 32 VAE channels * 4 (2x2 patch)
    const int img_seq = latH * latW;           // e.g. 64*64 = 4096
    const int N = config.num_inference_steps;

    if (static_cast<int>(latents.size()) < packed_ch * latH * latW) {
        throw std::runtime_error("Flux2Denoiser: latent buffer too small");
    }

    // ---- 1. Pack noise [128, latH, latW] -> tokens [img_seq, 128] ------------
    // token[p*128 + c] = latents[c*latH*latW + row*latW + col], p = row*latW+col.
    std::vector<float> tok(static_cast<size_t>(img_seq) * packed_ch);
    for (int c = 0; c < packed_ch; c++) {
        const size_t cbase = static_cast<size_t>(c) * latH * latW;
        for (int p = 0; p < img_seq; p++) {
            tok[static_cast<size_t>(p) * packed_ch + c] = latents[cbase + p];
        }
    }

    // ---- 2. Positional ids ---------------------------------------------------
    // img_ids [1, img_seq, 4]: (t=0, row, col, l=0), row-major over the grid.
    std::vector<float> img_ids(static_cast<size_t>(img_seq) * 4, 0.0f);
    for (int row = 0; row < latH; row++) {
        for (int col = 0; col < latW; col++) {
            const size_t base = static_cast<size_t>(row * latW + col) * 4;
            img_ids[base + 0] = 0.0f;
            img_ids[base + 1] = static_cast<float>(row);
            img_ids[base + 2] = static_cast<float>(col);
            img_ids[base + 3] = 0.0f;
        }
    }
    // Text sequence length derived from the actual embedding size.
    int txt_seq = text_seq_len_;
    if (text_embed_dim_ > 0 && !text_embeddings.empty()) {
        int computed = static_cast<int>(text_embeddings.size() / text_embed_dim_);
        if (computed > 0) txt_seq = computed;
    }
    // txt_ids [1, txt_seq, 4]: (0, 0, 0, l).
    std::vector<float> txt_ids(static_cast<size_t>(txt_seq) * 4, 0.0f);
    for (int l = 0; l < txt_seq; l++) {
        txt_ids[static_cast<size_t>(l) * 4 + 3] = static_cast<float>(l);
    }

    // ---- 3. Dynamic-shifting sigma schedule ---------------------------------
    const double mu = compute_empirical_mu(img_seq, N);
    const double emu = std::exp(mu);
    std::vector<float> sigmas(N + 1);
    for (int i = 0; i < N; i++) {
        // linspace(1.0, 1/N, N)
        double s = (N > 1) ? (1.0 + (1.0 / N - 1.0) * (static_cast<double>(i) / (N - 1)))
                           : 1.0;
        // exponential time shift: s' = e^mu / (e^mu + (1/s - 1))
        double shifted = emu / (emu + (1.0 / s - 1.0));
        sigmas[i] = static_cast<float>(shifted);
    }
    sigmas[N] = 0.0f;

    std::cout << "  FLUX.2 denoiser: img_seq=" << img_seq << " txt_seq=" << txt_seq
              << " steps=" << N << " mu=" << mu
              << " sigma[0]=" << sigmas[0] << " sigma[1]=" << sigmas[1] << std::endl;

    // ---- 4. Denoising loop ---------------------------------------------------
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const auto& in_names = model_->get_input_names();
    Scratch sc;

    std::vector<float> noise_pred(static_cast<size_t>(img_seq) * packed_ch);

    for (int step = 0; step < N; step++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        sc.clear();

        std::vector<Ort::Value> inputs;
        inputs.reserve(in_names.size());

        for (size_t i = 0; i < in_names.size(); i++) {
            const std::string name = to_lower(in_names[i]);
            const ONNXTensorElementDataType dt = model_->get_input_type(i);

            if (name == "img_ids") {
                std::vector<int64_t> shape = {1, img_seq, 4};
                inputs.push_back(emit(mem, sc, img_ids, shape, dt));
            } else if (name == "txt_ids") {
                std::vector<int64_t> shape = {1, txt_seq, 4};
                inputs.push_back(emit(mem, sc, txt_ids, shape, dt));
            } else if (name.find("encoder_hidden") != std::string::npos) {
                std::vector<int64_t> shape = {1, txt_seq, text_embed_dim_};
                inputs.push_back(emit(mem, sc, text_embeddings, shape, dt));
            } else if (name.find("timestep") != std::string::npos ||
                       name == "t" || name == "time") {
                // Reference passes timestep = t/1000 where t = sigma*1000,
                // i.e. the transformer receives sigma directly.
                std::vector<int64_t> shape = model_->get_input_shape(i);
                for (auto& d : shape) { if (d <= 0) d = 1; }
                size_t n = 1; for (auto d : shape) n *= static_cast<size_t>(d);
                std::vector<float> ts(n, sigmas[step]);
                inputs.push_back(emit(mem, sc, ts, shape, dt));
            } else if (i == 0 || name.find("hidden_states") != std::string::npos) {
                // hidden_states: packed latent tokens [1, img_seq, 128].
                std::vector<int64_t> shape = {1, img_seq, packed_ch};
                inputs.push_back(emit(mem, sc, tok, shape, dt));
            } else {
                // Unknown optional input: emit zeros of the declared shape.
                std::vector<int64_t> shape = model_->get_input_shape(i);
                for (auto& d : shape) { if (d <= 0) d = 1; }
                inputs.push_back(emit(mem, sc, {}, shape, dt));
            }
        }

        std::vector<Ort::Value> outputs;
        try {
            outputs = model_->run(inputs);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("FLUX.2 transformer failed at step ") +
                std::to_string(step + 1) + "/" + std::to_string(N) + ": " + e.what());
        }

        // Extract noise prediction [1, img_seq, 128] -> f32.
        auto info = outputs[0].GetTensorTypeAndShapeInfo();
        auto etype = info.GetElementType();
        const size_t np = noise_pred.size();
        if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            auto* fp16 = outputs[0].GetTensorMutableData<Ort::Float16_t>();
            for (size_t k = 0; k < np; k++) noise_pred[k] = fp16[k].ToFloat();
        } else {
            float* fp32 = outputs[0].GetTensorMutableData<float>();
            std::copy(fp32, fp32 + np, noise_pred.begin());
        }

        // Flow-match Euler step: x += pred * (sigma[step+1] - sigma[step]).
        const float dt = sigmas[step + 1] - sigmas[step];
        for (size_t k = 0; k < tok.size(); k++) {
            tok[k] += noise_pred[k] * dt;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "  step " << (step + 1) << "/" << N << " ("
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms)" << std::endl;
        std::cout.flush();
    }

    // ---- 5. Unpack tokens [img_seq, 128] -> packed [128, latH, latW] ---------
    std::vector<float> packed(static_cast<size_t>(packed_ch) * latH * latW);
    for (int c = 0; c < packed_ch; c++) {
        const size_t cbase = static_cast<size_t>(c) * latH * latW;
        for (int p = 0; p < img_seq; p++) {
            packed[cbase + p] = tok[static_cast<size_t>(p) * packed_ch + c];
        }
    }

    // ---- 6. BatchNorm denorm in packed 128-channel space --------------------
    // Reference: latents = latents * sqrt(var + eps) + mean  (per packed channel).
    // Must be applied BEFORE unpatchify because it is spatially-varying within
    // each 2x2 patch group once unpacked.
    if (bn_mean_.size() == static_cast<size_t>(packed_ch) &&
        bn_std_.size()  == static_cast<size_t>(packed_ch)) {
        for (int c = 0; c < packed_ch; c++) {
            const float sd = bn_std_[c];
            const float mn = bn_mean_[c];
            const size_t cbase = static_cast<size_t>(c) * latH * latW;
            for (int p = 0; p < img_seq; p++) {
                packed[cbase + p] = packed[cbase + p] * sd + mn;
            }
        }
    }

    // ---- 7. Unpatchify [128, latH, latW] -> [32, latH*2, latW*2] -------------
    // out[c, 2*row+dh, 2*col+dw] = packed[c*4 + dh*2 + dw, row, col].
    const int out_ch = packed_ch / 4;          // 32
    const int outH = latH * 2;                 // H/8
    const int outW = latW * 2;                 // W/8
    std::vector<float> result(static_cast<size_t>(out_ch) * outH * outW);
    for (int c = 0; c < out_ch; c++) {
        for (int row = 0; row < latH; row++) {
            for (int col = 0; col < latW; col++) {
                const size_t src_sp = static_cast<size_t>(row) * latW + col;
                for (int dh = 0; dh < 2; dh++) {
                    for (int dw = 0; dw < 2; dw++) {
                        const int pc = c * 4 + dh * 2 + dw;
                        const float v = packed[static_cast<size_t>(pc) * latH * latW + src_sp];
                        const size_t dst = static_cast<size_t>(c) * outH * outW
                                         + static_cast<size_t>(row * 2 + dh) * outW
                                         + (col * 2 + dw);
                        result[dst] = v;
                    }
                }
            }
        }
    }
    return result;
}

} // namespace sd_npu
