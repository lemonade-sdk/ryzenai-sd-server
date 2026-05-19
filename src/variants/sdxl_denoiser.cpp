// sdxl_denoiser.cpp - SDXL and SDXL-Turbo denoiser implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variants/sdxl_denoiser.h"
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <algorithm>

namespace sd_npu {

SDXLDenoiser::SDXLDenoiser(
    const std::map<ComponentType, std::unique_ptr<OnnxModel>>& components) {
    
    // Get UNet model reference
    auto it = components.find(ComponentType::UNET);
    if (it == components.end()) {
        throw std::runtime_error("SDXLDenoiser: UNET component not found");
    }
    unet_ = it->second.get();
}

std::vector<float> SDXLDenoiser::denoise(
    const std::vector<float>& latents,
    const std::vector<float>& text_embeddings,
    const std::vector<float>& pooled_embeddings,
    Scheduler& scheduler,
    const SDConfig& config,
    const std::vector<float>& controlnet_cond) {

    (void)controlnet_cond;  // ControlNet support not yet implemented

    // Get latent dimensions from model
    auto sample_shape = unet_->get_input_shape(0);  // [batch, ch, h, w]
    int channels = (sample_shape.size() >= 4 && sample_shape[1] > 0) ? (int)sample_shape[1] : 4;
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
    // For img2img, latents already have noise added at the correct sigma level - do NOT rescale.
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
        
        // Scale latent input (e.g., Euler divides by sqrt(sigma^2+1))
        scheduler.scale_model_input(latent_model_input, step);
        
        // Build inputs
        std::vector<Ort::Value> inputs;
        build_inputs(inputs, mem, unet_, latent_model_input,
                    timesteps[step], text_emb_precomp, pooled_precomp,
                    batch, channels, latent_h, latent_w, config);
        
        // Run UNet
        std::vector<Ort::Value> outputs;
        try {
            outputs = unet_->run(inputs);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("UNet failed at step ") +
                std::to_string(step + 1) + "/" +
                std::to_string(config.num_inference_steps) +
                ": " + e.what());
        }
        
        // Extract noise prediction (handle fp16→fp32 conversion)
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

void SDXLDenoiser::build_inputs(
    std::vector<Ort::Value>& inputs,
    Ort::MemoryInfo& mem,
    OnnxModel* model,
    std::vector<float>& latent_model_input,
    float timestep,
    std::vector<float>& text_emb,
    std::vector<float>& pooled,
    int batch, int channels, int latent_h, int latent_w,
    const SDConfig& config) {
    
    // Input 0: sample [batch, channels, latent_h, latent_w]
    std::vector<int64_t> s_shape = {batch, channels, latent_h, latent_w};
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem, latent_model_input.data(), latent_model_input.size(),
        s_shape.data(), s_shape.size()));
    
    // Input 1: timestep (can be int64 or float32)
    auto ts_shape = model->get_input_shape(1);
    for (auto& d : ts_shape) { if (d <= 0) d = batch; }
    size_t ts_sz = 1;
    for (auto d : ts_shape) ts_sz *= d;
    auto ts_type = model->get_input_type(1);
    
    if (ts_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        buffers_.ts_i64.assign(ts_sz, (int64_t)timestep);
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            mem, buffers_.ts_i64.data(), buffers_.ts_i64.size(),
            ts_shape.data(), ts_shape.size()));
    } else {
        buffers_.ts_f32.assign(ts_sz, timestep);
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem, buffers_.ts_f32.data(), buffers_.ts_f32.size(),
            ts_shape.data(), ts_shape.size()));
    }
    
    // Input 2: encoder_hidden_states [batch, 77, 2048]
    std::vector<int64_t> ehs_shape = {batch, 77, 2048};
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem, text_emb.data(), text_emb.size(),
        ehs_shape.data(), ehs_shape.size()));
    
    // Input 3: text_embeds (pooled) [batch, 1280]
    std::vector<int64_t> te_shape = {batch, 1280};
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem, pooled.data(), pooled.size(),
        te_shape.data(), te_shape.size()));
    
    // Input 4: time_ids [batch, 6]
    std::vector<int64_t> tid_shape = {batch, 6};
    buffers_.time_ids_data.resize(batch * 6);
    for (int b = 0; b < batch; b++) {
        buffers_.time_ids_data[b * 6 + 0] = (float)config.height;
        buffers_.time_ids_data[b * 6 + 1] = (float)config.width;
        buffers_.time_ids_data[b * 6 + 2] = 0.0f;
        buffers_.time_ids_data[b * 6 + 3] = 0.0f;
        buffers_.time_ids_data[b * 6 + 4] = (float)config.height;
        buffers_.time_ids_data[b * 6 + 5] = (float)config.width;
    }
    
    auto tid_type = model->get_input_type(4);
    if (tid_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        buffers_.time_ids_i64.resize(batch * 6);
        for (int i = 0; i < batch * 6; i++)
            buffers_.time_ids_i64[i] = (int64_t)buffers_.time_ids_data[i];
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            mem, buffers_.time_ids_i64.data(), buffers_.time_ids_i64.size(),
            tid_shape.data(), tid_shape.size()));
    } else {
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem, buffers_.time_ids_data.data(), buffers_.time_ids_data.size(),
            tid_shape.data(), tid_shape.size()));
    }
}

} // namespace sd_npu
