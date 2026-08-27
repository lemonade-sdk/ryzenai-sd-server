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
            auto declared = model_->get_input_shape(i);
            if (declared.size() == 3) {
                // FLUX-style packed latent: [B, n_patches, patch_dim]
                // Pack [B, C, H, W] -> [B, (H/2)*(W/2), C*4] with 2x2 patches.
                const int patch = 2;
                const int ph = latent_h / patch;
                const int pw = latent_w / patch;
                const int n_patches = ph * pw;
                const int patch_dim = channels * patch * patch;  // 16*4 = 64
                const size_t total = static_cast<size_t>(batch) * n_patches * patch_dim;
                scratch_.f32.emplace_back(total, 0.0f);
                auto& packed = scratch_.f32.back();

                // latent_input is [B, C, H, W] row-major
                for (int b = 0; b < batch; b++) {
                    for (int h = 0; h < ph; h++) {
                        for (int w = 0; w < pw; w++) {
                            const int patch_idx = h * pw + w;
                            int out_ch = 0;
                            for (int c = 0; c < channels; c++) {
                                for (int dh = 0; dh < patch; dh++) {
                                    for (int dw = 0; dw < patch; dw++) {
                                        const size_t src = static_cast<size_t>(b) * channels * latent_h * latent_w
                                            + static_cast<size_t>(c) * latent_h * latent_w
                                            + static_cast<size_t>(h * patch + dh) * latent_w
                                            + (w * patch + dw);
                                        const size_t dst = static_cast<size_t>(b) * n_patches * patch_dim
                                            + static_cast<size_t>(patch_idx) * patch_dim
                                            + out_ch;
                                        packed[dst] = (src < latent_input.size()) ? latent_input[src] : 0.0f;
                                        out_ch++;
                                    }
                                }
                            }
                        }
                    }
                }
                std::vector<int64_t> shape = {batch, n_patches, patch_dim};
                inputs.push_back(emit(mem, packed, shape, dt));
            } else {
                // Standard 4D latent: [B, C, H, W]
                std::vector<int64_t> shape = {batch, channels, latent_h, latent_w};
                inputs.push_back(emit(mem, latent_input, shape, dt));
            }
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
            // seq_len is derived from the actual embedding size when the spec
            // uses a dynamic length (e.g. T5 output varies by prompt).
            int64_t actual_seq = static_cast<int64_t>(spec_.seq_len);
            if (spec_.embed_dim > 0 && !text_emb.empty()) {
                int64_t computed = static_cast<int64_t>(text_emb.size())
                                   / (batch * spec_.embed_dim);
                if (computed > 0) actual_seq = computed;
            }
            std::vector<int64_t> shape =
                {batch, actual_seq, static_cast<int64_t>(spec_.embed_dim)};
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
        else if (name == "img_ids") {
            // FLUX: 2D positional IDs for packed latent patches.
            // Latent is (H/8) x (W/8); patches are 2x2, so grid is (H/16) x (W/16).
            const int ph = latent_h / 2;
            const int pw = latent_w / 2;
            const int64_t n_patches = static_cast<int64_t>(ph) * pw;
            std::vector<int64_t> shape = {n_patches, 3};
            scratch_.f32.emplace_back(static_cast<size_t>(n_patches) * 3, 0.0f);
            auto& buf = scratch_.f32.back();
            for (int h = 0; h < ph; h++) {
                for (int w = 0; w < pw; w++) {
                    const size_t base = static_cast<size_t>(h * pw + w) * 3;
                    buf[base + 0] = 0.0f;
                    buf[base + 1] = static_cast<float>(h);
                    buf[base + 2] = static_cast<float>(w);
                }
            }
            inputs.push_back(emit(mem, buf, shape, dt));
        }
        else if (name == "txt_ids") {
            // FLUX: positional IDs for T5 text tokens — convention is all zeros.
            int64_t actual_seq = static_cast<int64_t>(spec_.seq_len);
            if (spec_.embed_dim > 0 && !text_emb.empty()) {
                int64_t computed = static_cast<int64_t>(text_emb.size())
                                   / (batch * spec_.embed_dim);
                if (computed > 0) actual_seq = computed;
            }
            std::vector<int64_t> shape = {actual_seq, 3};
            inputs.push_back(emit(mem, {}, shape, dt));
        }
        else if (name.find("timestep") != std::string::npos ||
                 name == "t" || name == "timesteps" || name == "time") {
            std::vector<int64_t> shape = model_->get_input_shape(i);
            for (auto& d : shape) { if (d <= 0) d = batch; }
            size_t n = 1;
            for (auto d : shape) n *= static_cast<size_t>(d);
            // FLUX transformers expect the timestep normalized to [0,1]
            // (reference passes timestep/1000); other models use it as-is.
            std::vector<float> ts(n, timestep * spec_.timestep_scale);
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

    // CFG: double the batch (uncond + cond) whenever the text encoder actually
    // produced both halves. The NPU-compiled (DD) UNet/transformer binaries are
    // compiled for a fixed batch layout matching what the encoder emits, so the
    // decision must track the embeddings' actual batch — not guidance_scale —
    // otherwise low/zero guidance values (e.g. lightning checkpoints defaulting
    // to guidance=1.0, or ControlNet types requiring guidance=0.0) mismatch the
    // compiled shape and crash the NPU execution provider.
    // FLUX distilled models (schnell/klein) are guidance-distilled: the text
    // encoder emits a single batch and the transformer has no guidance input,
    // so classifier-free guidance must never double the batch here.
    const size_t single_emb = (spec_.seq_len > 0 && spec_.embed_dim > 0)
        ? static_cast<size_t>(spec_.seq_len) * spec_.embed_dim : 0;
    const bool encoder_gave_two_batch =
        single_emb > 0 && text_embeddings.size() == 2 * single_emb;
    int batch = (!spec_.disable_cfg && encoder_gave_two_batch) ? 2 : 1;

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

    // If, despite the above, the encoder emitted two batches but CFG was
    // disabled (spec_.disable_cfg), keep only the conditional (second) half so
    // the single-batch model still gets a valid input.
    if (batch == 1 && encoder_gave_two_batch) {
        text_emb.assign(text_emb.begin() + single_emb, text_emb.end());
        if (spec_.pool_dim > 0 && pooled.size() == 2 * static_cast<size_t>(spec_.pool_dim)) {
            pooled.assign(pooled.begin() + spec_.pool_dim, pooled.end());
        }
    }

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

        // For FLUX the output is packed [B, n_patches, 64]; total elements == batch*single_size.
        const bool packed_output = (out_shape.size() == 3);
        noise_pred_f32.resize(static_cast<size_t>(out_shape[0]) * single_size);
        if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            auto* fp16 = outputs[0].GetTensorMutableData<Ort::Float16_t>();
            for (size_t i = 0; i < noise_pred_f32.size(); i++)
                noise_pred_f32[i] = fp16[i].ToFloat();
        } else {
            float* fp32 = outputs[0].GetTensorMutableData<float>();
            std::copy(fp32, fp32 + noise_pred_f32.size(), noise_pred_f32.begin());
        }

        // Unpack FLUX [B, n_patches, 64] → [B, C, H, W] using inverse 2x2 patchify.
        if (packed_output) {
            const int patch = 2;
            const int ph = latent_h / patch;
            const int pw = latent_w / patch;
            const int patch_dim = channels * patch * patch;
            std::vector<float> unpacked(noise_pred_f32.size());
            for (int b = 0; b < static_cast<int>(out_shape[0]); b++) {
                for (int h = 0; h < ph; h++) {
                    for (int w = 0; w < pw; w++) {
                        const int patch_idx = h * pw + w;
                        int in_ch = 0;
                        for (int c = 0; c < channels; c++) {
                            for (int dh = 0; dh < patch; dh++) {
                                for (int dw = 0; dw < patch; dw++) {
                                    const size_t src = static_cast<size_t>(b) * ph * pw * patch_dim
                                        + static_cast<size_t>(patch_idx) * patch_dim + in_ch;
                                    const size_t dst = static_cast<size_t>(b) * channels * latent_h * latent_w
                                        + static_cast<size_t>(c) * latent_h * latent_w
                                        + static_cast<size_t>(h * patch + dh) * latent_w
                                        + (w * patch + dw);
                                    unpacked[dst] = noise_pred_f32[src];
                                    in_ch++;
                                }
                            }
                        }
                    }
                }
            }
            noise_pred_f32 = std::move(unpacked);
        }

        // Classifier-Free Guidance. Gated on the actual output batch (not
        // guidance_scale) since batch==2 whenever the encoder provided both
        // uncond+cond halves (see encoder_gave_two_batch above); this keeps
        // the blend correct even when guidance_scale <= 1.0 (e.g. guidance=1.0
        // reduces to pure "cond", guidance=0.0 reduces to pure "uncond").
        if (out_shape[0] >= 2) {
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
