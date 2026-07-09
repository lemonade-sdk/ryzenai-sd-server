// generic_denoiser.cpp - Unified, metadata-driven denoiser for all SD variants
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variants/generic_denoiser.h"
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <string>

namespace sd_npu {

namespace {
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}
} // namespace

GenericDenoiser::GenericDenoiser(
    const std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
    DenoiserSpec spec,
    ControlNetRunner* controlnet_runner)
    : spec_(spec), controlnet_runner_(controlnet_runner) {

    auto it = components.find(spec_.component);
    if (it == components.end()) {
        throw std::runtime_error("GenericDenoiser: required denoising component not found");
    }
    model_ = it->second.get();

    // Detect ControlNet block inputs (SD3-style transformers declare a series of
    // "block_controlnet_hidden_states_*" inputs even when ControlNet is unused).
    const auto& input_names = model_->get_input_names();
    for (size_t i = 0; i < input_names.size(); i++) {
        if (input_names[i].find("block_controlnet_hidden_states_") != std::string::npos) {
            controlnet_block_count_++;
            if (controlnet_channel_dim_ == 0) {
                auto sh = model_->get_input_shape(i);
                if (sh.size() >= 3 && sh[2] > 0)
                    controlnet_channel_dim_ = static_cast<int>(sh[2]);
            }
        }
    }

    if (controlnet_block_count_ > 0) {
        std::cout << "  Denoiser: model expects " << controlnet_block_count_
                  << " controlnet blocks (channel_dim=" << controlnet_channel_dim_ << ")"
                  << std::endl;
        if (controlnet_runner_) {
            std::cout << "  ControlNet runner available: "
                      << controlnet_runner_->get_name() << std::endl;
        }
    }
}

Ort::Value GenericDenoiser::emit(
    Ort::MemoryInfo& mem,
    const std::vector<float>& src,
    const std::vector<int64_t>& shape,
    ONNXTensorElementDataType dt) {

    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d > 0 ? d : 1);

    switch (dt) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
        scratch_.f16.emplace_back(n);
        auto& b = scratch_.f16.back();
        for (size_t i = 0; i < n; i++)
            b[i] = Ort::Float16_t(i < src.size() ? src[i] : 0.0f);
        return Ort::Value::CreateTensor<Ort::Float16_t>(
            mem, b.data(), n, shape.data(), shape.size());
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
        scratch_.i64.emplace_back(n);
        auto& b = scratch_.i64.back();
        for (size_t i = 0; i < n; i++)
            b[i] = static_cast<int64_t>(i < src.size() ? src[i] : 0.0f);
        return Ort::Value::CreateTensor<int64_t>(
            mem, b.data(), n, shape.data(), shape.size());
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
        scratch_.f64.emplace_back(n);
        auto& b = scratch_.f64.back();
        for (size_t i = 0; i < n; i++)
            b[i] = static_cast<double>(i < src.size() ? src[i] : 0.0f);
        return Ort::Value::CreateTensor<double>(
            mem, b.data(), n, shape.data(), shape.size());
    }
    default: {  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        scratch_.f32.emplace_back(n, 0.0f);
        auto& b = scratch_.f32.back();
        std::copy_n(src.begin(), std::min(n, src.size()), b.begin());
        return Ort::Value::CreateTensor<float>(
            mem, b.data(), n, shape.data(), shape.size());
    }
    }
}

void GenericDenoiser::build_inputs(
    std::vector<Ort::Value>& inputs,
    Ort::MemoryInfo& mem,
    const std::vector<float>& latent_input,
    float timestep,
    const std::vector<float>& text_emb,
    const std::vector<float>& pooled,
    int batch, int channels, int latent_h, int latent_w,
    const SDConfig& config) {

    scratch_.clear();
    const auto& names = model_->get_input_names();
    int ctrl_idx = 0;

    for (size_t i = 0; i < names.size(); i++) {
        const std::string name = to_lower(names[i]);
        const ONNXTensorElementDataType dt = model_->get_input_type(i);

        // Input 0 is always the latent sample (named "sample" or "hidden_states").
        if (i == 0) {
            std::vector<int64_t> shape = {batch, channels, latent_h, latent_w};
            inputs.push_back(emit(mem, latent_input, shape, dt));
            continue;
        }

        if (name.find("controlnet") != std::string::npos) {
            // SD3 ControlNet block: use produced blocks if available, else zeros.
            const int spatial = (config.height / 16) * (config.width / 16);
            std::vector<int64_t> shape =
                {batch, static_cast<int64_t>(spatial), static_cast<int64_t>(controlnet_channel_dim_)};
            const size_t blk_n =
                static_cast<size_t>(batch) * spatial * controlnet_channel_dim_;

            if (ctrl_idx < static_cast<int>(ctrl_fp16_.size()) &&
                ctrl_fp16_[ctrl_idx].size() == blk_n) {
                // Reference the block produced by the ControlNet runner this step
                // (kept alive in ctrl_fp16_ until run() completes).
                inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                    mem, ctrl_fp16_[ctrl_idx].data(), blk_n,
                    shape.data(), shape.size()));
            } else {
                scratch_.f16.emplace_back(blk_n, Ort::Float16_t(0.0f));
                auto& b = scratch_.f16.back();
                inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                    mem, b.data(), blk_n, shape.data(), shape.size()));
            }
            ctrl_idx++;
        }
        else if (name.find("encoder_hidden_states") != std::string::npos) {
            std::vector<int64_t> shape =
                {batch, static_cast<int64_t>(spec_.seq_len), static_cast<int64_t>(spec_.embed_dim)};
            inputs.push_back(emit(mem, text_emb, shape, dt));
        }
        else if (name.find("pooled") != std::string::npos ||
                 name.find("text_embeds") != std::string::npos) {
            std::vector<int64_t> shape = {batch, static_cast<int64_t>(spec_.pool_dim)};
            inputs.push_back(emit(mem, pooled, shape, dt));
        }
        else if (name.find("time_ids") != std::string::npos) {
            // SDXL micro-conditioning: [orig_h, orig_w, crop_top, crop_left, target_h, target_w].
            std::vector<int64_t> shape = {batch, 6};
            std::vector<float> tid(static_cast<size_t>(batch) * 6);
            for (int b = 0; b < batch; b++) {
                tid[b * 6 + 0] = static_cast<float>(config.height);
                tid[b * 6 + 1] = static_cast<float>(config.width);
                tid[b * 6 + 2] = 0.0f;
                tid[b * 6 + 3] = 0.0f;
                tid[b * 6 + 4] = static_cast<float>(config.height);
                tid[b * 6 + 5] = static_cast<float>(config.width);
            }
            inputs.push_back(emit(mem, tid, shape, dt));
        }
        else if (name.find("timestep") != std::string::npos ||
                 name == "t" || name == "timesteps" || name == "time") {
            std::vector<int64_t> shape = model_->get_input_shape(i);
            for (auto& d : shape) { if (d <= 0) d = batch; }
            size_t n = 1;
            for (auto d : shape) n *= static_cast<size_t>(d);
            std::vector<float> ts(n, timestep);
            inputs.push_back(emit(mem, ts, shape, dt));
        }
        else {
            // Unknown/optional input: emit a zero tensor of the declared shape.
            std::vector<int64_t> shape = model_->get_input_shape(i);
            for (auto& d : shape) { if (d <= 0) d = batch; }
            inputs.push_back(emit(mem, {}, shape, dt));
        }
    }
}

std::vector<float> GenericDenoiser::denoise(
    const std::vector<float>& latents,
    const std::vector<float>& text_embeddings,
    const std::vector<float>& pooled_embeddings,
    Scheduler& scheduler,
    const SDConfig& config,
    const std::vector<float>& controlnet_cond) {

    // Latent dimensions from the model (fall back to config-derived values).
    auto sample_shape = model_->get_input_shape(0);  // [batch, ch, h, w]
    int channels = (sample_shape.size() >= 4 && sample_shape[1] > 0)
                       ? static_cast<int>(sample_shape[1]) : spec_.latent_ch;
    int latent_h = (sample_shape.size() >= 4 && sample_shape[2] > 0)
                       ? static_cast<int>(sample_shape[2]) : config.height / 8;
    int latent_w = (sample_shape.size() >= 4 && sample_shape[3] > 0)
                       ? static_cast<int>(sample_shape[3]) : config.width / 8;

    // CFG: double the batch (uncond + cond) when guidance is active.
    int batch = (config.guidance_scale > 1.0f) ? 2 : 1;

    scheduler.set_timesteps(config.num_inference_steps);
    const auto& timesteps = scheduler.timesteps();

    std::vector<float> current_latents = latents;

    int start_step = config.img2img_start_step;

    // txt2img: scale pure noise by init_noise_sigma. img2img: latents already carry
    // noise at the correct sigma, so don't rescale. is_img2img guards start_step==0
    // clamping (e.g. 1-step models).
    if (start_step == 0 && !config.is_img2img) {
        scheduler.scale_initial_latents(current_latents);
    }

    size_t single_size = static_cast<size_t>(channels) * latent_h * latent_w;

    std::vector<float> latent_model_input(batch * single_size);
    std::vector<float> noise_pred_f32;
    std::vector<float> guided(single_size);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> text_emb = text_embeddings;
    std::vector<float> pooled = pooled_embeddings;

    for (int step = start_step; step < config.num_inference_steps; step++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Prepare latent input for CFG.
        if (batch == 2) {
            std::copy(current_latents.begin(), current_latents.end(),
                      latent_model_input.begin());
            std::copy(current_latents.begin(), current_latents.end(),
                      latent_model_input.begin() + single_size);
        } else {
            std::copy(current_latents.begin(), current_latents.end(),
                      latent_model_input.begin());
        }

        // Scale latent input (e.g. Euler divides by sqrt(sigma^2+1); FlowMatch is a no-op).
        scheduler.scale_model_input(latent_model_input, step);

        // ControlNet pre-step (SD3) — only when a runner and blocks are present.
        if (controlnet_runner_ && controlnet_runner_->is_loaded() &&
            controlnet_block_count_ > 0 && !controlnet_cond.empty()) {
            ctrl_fp16_ = controlnet_runner_->compute(
                latent_model_input, controlnet_cond, timesteps[step],
                text_emb, pooled,
                batch, channels, latent_h, latent_w,
                config.height, config.width,
                spec_.seq_len, spec_.embed_dim, spec_.pool_dim,
                config.controlnet_scale);
            if (ctrl_fp16_.empty() && step == 0) {
                std::cout << "  WARNING: ControlNet failed, using zero blocks" << std::endl;
            }
        }

        // Build inputs (name-driven) and run the model.
        std::vector<Ort::Value> inputs;
        build_inputs(inputs, mem, latent_model_input, timesteps[step],
                     text_emb, pooled, batch, channels, latent_h, latent_w, config);

        std::vector<Ort::Value> outputs;
        try {
            outputs = model_->run(inputs);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("Denoise model failed at step ") +
                std::to_string(step + 1) + "/" +
                std::to_string(config.num_inference_steps) +
                ": " + e.what());
        }

        // Extract noise prediction (handle fp16 -> fp32 conversion).
        auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
        auto out_shape = out_info.GetShape();
        auto out_type = out_info.GetElementType();

        noise_pred_f32.resize(out_shape[0] * single_size);
        if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            auto* fp16 = outputs[0].GetTensorMutableData<Ort::Float16_t>();
            for (size_t i = 0; i < noise_pred_f32.size(); i++)
                noise_pred_f32[i] = fp16[i].ToFloat();
        } else {
            float* fp32 = outputs[0].GetTensorMutableData<float>();
            std::copy(fp32, fp32 + noise_pred_f32.size(), noise_pred_f32.begin());
        }

        // Classifier-Free Guidance.
        if (config.guidance_scale > 1.0f && out_shape[0] >= 2) {
            for (size_t i = 0; i < single_size; i++) {
                float uncond = noise_pred_f32[i];
                float cond = noise_pred_f32[single_size + i];
                guided[i] = uncond + config.guidance_scale * (cond - uncond);
            }
        } else {
            std::copy(noise_pred_f32.begin(),
                      noise_pred_f32.begin() + single_size, guided.begin());
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "[PROGRESS] Step " << (step + 1) << "/" << config.num_inference_steps
                  << " (" << ms << " ms)" << std::endl;
        std::cout.flush();

        scheduler.step(guided, step, current_latents);
    }

    return current_latents;
}

} // namespace sd_npu
