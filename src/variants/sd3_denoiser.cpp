// sd3_denoiser.cpp - SD3 and SD3.5 denoiser implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variants/sd3_denoiser.h"
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <algorithm>

namespace sd_npu {

SD3Denoiser::SD3Denoiser(
    const std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
    ControlNetRunner* controlnet_runner,
    int seq_len, int embed_dim, int pool_dim)
    : seq_len_(seq_len), embed_dim_(embed_dim), pool_dim_(pool_dim),
      controlnet_runner_(controlnet_runner) {
    
    // Get Transformer model reference
    auto it = components.find(ComponentType::TRANSFORMER);
    if (it == components.end()) {
        throw std::runtime_error("SD3Denoiser: TRANSFORMER component not found");
    }
    transformer_ = it->second.get();
    
    // Detect controlnet inputs
    auto& input_names = transformer_->get_input_names();
    controlnet_block_count_ = 0;
    controlnet_channel_dim_ = 0;
    
    for (size_t i = 0; i < input_names.size(); i++) {
        if (input_names[i].find("block_controlnet_hidden_states_") != std::string::npos) {
            controlnet_block_count_++;
            if (controlnet_channel_dim_ == 0) {
                auto sh = transformer_->get_input_shape(i);
                if (sh.size() >= 3 && sh[2] > 0)
                    controlnet_channel_dim_ = (int)sh[2];
            }
        }
    }
    
    if (controlnet_block_count_ > 0) {
        std::cout << "  SD3 Transformer expects " << controlnet_block_count_
                  << " controlnet blocks (channel_dim=" << controlnet_channel_dim_ << ")" << std::endl;
        if (controlnet_runner_) {
            std::cout << "  ControlNet runner available: " 
                      << controlnet_runner_->get_name() << std::endl;
        }
    }
}

std::vector<float> SD3Denoiser::denoise(
    const std::vector<float>& latents,
    const std::vector<float>& text_embeddings,
    const std::vector<float>& pooled_embeddings,
    Scheduler& scheduler,
    const SDConfig& config,
    const std::vector<float>& controlnet_cond) {

    // Get latent dimensions from model
    auto sample_shape = transformer_->get_input_shape(0);  // [batch, ch, h, w]
    int channels = (sample_shape.size() >= 4 && sample_shape[1] > 0) ? (int)sample_shape[1] : 16;
    int latent_h = (sample_shape.size() >= 4 && sample_shape[2] > 0) ? (int)sample_shape[2] : config.height / 8;
    int latent_w = (sample_shape.size() >= 4 && sample_shape[3] > 0) ? (int)sample_shape[3] : config.width / 8;

    // CFG: double batch (uncond + cond)
    int batch = (config.guidance_scale > 1.0f) ? 2 : 1;
    
    // Timesteps
    scheduler.set_timesteps(config.num_inference_steps);
    const auto& timesteps = scheduler.timesteps();

    // Working latents
    std::vector<float> current_latents = latents;

    int start_step = config.img2img_start_step;

    // For txt2img, scale by init_noise_sigma (pure random noise needs scaling).
    // For img2img, latents already have noise at the correct sigma level - do NOT rescale.
    // is_img2img guards the case where start_step==0 due to clamping (e.g. 1-step models).
    if (start_step == 0 && !config.is_img2img) {
        scheduler.scale_initial_latents(current_latents);
    }
    
    size_t single_size = channels * latent_h * latent_w;
    
    // Allocate buffers
    std::vector<float> latent_model_input(batch * single_size);
    std::vector<float> noise_pred_f32;
    std::vector<float> guided(single_size);
    
    // Memory info for ONNX tensors
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Pre-compute text embeddings and pooled embeddings
    std::vector<float> text_emb_precomp = text_embeddings;
    std::vector<float> pooled_precomp = pooled_embeddings;
    
    // Denoising loop (start from start_step for img2img, 0 for txt2img)
    for (int step = start_step; step < config.num_inference_steps; step++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        
        // Prepare latent input for CFG
        if (batch == 2) {
            // Duplicate latents: [uncond, cond]
            std::copy(current_latents.begin(), current_latents.end(),
                     latent_model_input.begin());
            std::copy(current_latents.begin(), current_latents.end(),
                     latent_model_input.begin() + single_size);
        } else {
            std::copy(current_latents.begin(), current_latents.end(),
                     latent_model_input.begin());
        }
        
        // Scale latent input (FlowMatch uses identity scaling, no-op)
        scheduler.scale_model_input(latent_model_input, step);
        
        // Run ControlNet model if available to populate ctrl_fp16 blocks
        if (controlnet_runner_ && controlnet_runner_->is_loaded() && !controlnet_cond.empty()) {
            buffers_.ctrl_fp16 = controlnet_runner_->compute(
                latent_model_input, controlnet_cond, timesteps[step],
                text_emb_precomp, pooled_precomp,
                batch, channels, latent_h, latent_w,
                config.height, config.width,
                seq_len_, embed_dim_, pool_dim_,
                config.controlnet_scale);
            
            if (buffers_.ctrl_fp16.empty() && step == 0) {
                std::cout << "  WARNING: ControlNet failed, using zero blocks" << std::endl;
            }
        }
        
        // Build inputs
        std::vector<Ort::Value> inputs;
        build_inputs(inputs, mem, latent_model_input,
                    timesteps[step], text_emb_precomp, pooled_precomp,
                    batch, channels, latent_h, latent_w, config);
        
        // Run Transformer
        std::vector<Ort::Value> outputs;
        try {
            outputs = transformer_->run(inputs);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("Transformer failed at step ") +
                std::to_string(step + 1) + "/" +
                std::to_string(config.num_inference_steps) +
                ": " + e.what());
        }
        
        // Extract noise prediction (fp16→fp32 conversion)
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
        
        // Apply Classifier-Free Guidance
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
        
        // Scheduler step
        scheduler.step(guided, step, current_latents);
    }
    
    return current_latents;
}

void SD3Denoiser::build_inputs(
    std::vector<Ort::Value>& inputs,
    Ort::MemoryInfo& mem,
    std::vector<float>& latent_model_input,
    float timestep,
    std::vector<float>& text_emb,
    std::vector<float>& pooled,
    int batch, int channels, int latent_h, int latent_w,
    const SDConfig& config) {
    
    (void)config;  // Currently unused but kept for consistency
    
    // Input 0: hidden_states [batch, channels, latent_h, latent_w] - float16
    size_t hs_sz = batch * channels * latent_h * latent_w;
    buffers_.hs_fp16.resize(hs_sz);
    for (size_t i = 0; i < hs_sz; i++)
        buffers_.hs_fp16[i] = Ort::Float16_t(latent_model_input[i]);
    std::vector<int64_t> hs_shape = {batch, channels, latent_h, latent_w};
    inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        mem, buffers_.hs_fp16.data(), buffers_.hs_fp16.size(),
        hs_shape.data(), hs_shape.size()));
    
    // Input 1: timestep [batch] - float16
    buffers_.ts_fp16.assign(batch, Ort::Float16_t(timestep));
    std::vector<int64_t> ts_shape = {batch};
    inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        mem, buffers_.ts_fp16.data(), buffers_.ts_fp16.size(),
        ts_shape.data(), ts_shape.size()));
    
    // Input 2: encoder_hidden_states [batch, seq_len, embed_dim] - float16
    size_t ehs_sz = batch * seq_len_ * embed_dim_;
    buffers_.ehs_fp16.resize(ehs_sz);
    for (size_t i = 0; i < std::min(ehs_sz, text_emb.size()); i++)
        buffers_.ehs_fp16[i] = Ort::Float16_t(text_emb[i]);
    for (size_t i = text_emb.size(); i < ehs_sz; i++)
        buffers_.ehs_fp16[i] = Ort::Float16_t(0.0f);
    std::vector<int64_t> ehs_shape = {batch, (int64_t)seq_len_, (int64_t)embed_dim_};
    inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        mem, buffers_.ehs_fp16.data(), buffers_.ehs_fp16.size(),
        ehs_shape.data(), ehs_shape.size()));
    
    // Input 3: pooled_projections [batch, pool_dim] - float16
    size_t pool_sz = batch * pool_dim_;
    buffers_.pooled_fp16.resize(pool_sz);
    for (size_t i = 0; i < std::min(pool_sz, pooled.size()); i++)
        buffers_.pooled_fp16[i] = Ort::Float16_t(pooled[i]);
    for (size_t i = pooled.size(); i < pool_sz; i++)
        buffers_.pooled_fp16[i] = Ort::Float16_t(0.0f);
    std::vector<int64_t> pool_shape = {batch, (int64_t)pool_dim_};
    inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        mem, buffers_.pooled_fp16.data(), buffers_.pooled_fp16.size(),
        pool_shape.data(), pool_shape.size()));
    
    // Inputs 4+: ControlNet blocks
    // SD3 models expect these inputs even when controlnet is not being used
    // Use actual controlnet blocks if available, otherwise use zero-filled blocks
    if (controlnet_block_count_ > 0 && controlnet_channel_dim_ > 0) {
        int spatial = (config.height / 16) * (config.width / 16);
        size_t blk_sz = batch * spatial * controlnet_channel_dim_;
        std::vector<int64_t> blk_shape = {batch, (int64_t)spatial, (int64_t)controlnet_channel_dim_};
        
        // If ctrl_fp16 is already populated (from run_controlnet), use those blocks
        // Otherwise, create zero-filled blocks
        bool use_actual_blocks = (buffers_.ctrl_fp16.size() == static_cast<size_t>(controlnet_block_count_) &&
                                  !buffers_.ctrl_fp16[0].empty());
        
        if (!use_actual_blocks) {
            // Create zero-filled blocks as fallback
            buffers_.ctrl_fp16.resize(controlnet_block_count_);
            for (int b = 0; b < controlnet_block_count_; b++) {
                buffers_.ctrl_fp16[b].assign(blk_sz, Ort::Float16_t(0.0f));
            }
        }
        
        // Add blocks to inputs (either actual controlnet blocks or zeros)
        for (int b = 0; b < controlnet_block_count_; b++) {
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                mem, buffers_.ctrl_fp16[b].data(), buffers_.ctrl_fp16[b].size(),
                blk_shape.data(), blk_shape.size()));
        }
    }
}

} // namespace sd_npu
