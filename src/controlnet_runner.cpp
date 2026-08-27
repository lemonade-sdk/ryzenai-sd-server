// controlnet_runner.cpp - ControlNet helper implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "controlnet_runner.h"
#include <iostream>

namespace sd_npu {

ControlNetRunner::ControlNetRunner(OnnxModel* model, ControlNetType type)
    : model_(model), type_(type) {
}

std::vector<std::vector<Ort::Float16_t>> ControlNetRunner::compute(
    const std::vector<float>& latent_input,
    const std::vector<float>& control_cond,
    float timestep,
    const std::vector<float>& text_emb,
    const std::vector<float>& pooled_emb,
    int batch, int channels, int latent_h, int latent_w,    int /*height*/, int /*width*/,    int seq_len, int embed_dim, int pool_dim,
    float conditioning_scale) {

    if (!model_ || control_cond.empty()) {
        return {};
    }

    std::cout << "  Running ControlNet (" << get_name() << ") inference..." << std::endl;

    // ControlNet inputs (based on GenAI-SD reference):
    // - hidden_states: latent_model_input [batch, channels, latent_h, latent_w]
    // - controlnet_cond: normalized control image [-1,1] [batch, 3, height, width] at FULL resolution
    // - timestep: scalar timestep [1] (fp16)
    // - pooled_projections: pooled text embeddings [batch, pool_dim]
    // - encoder_hidden_states (optional): text embeddings [batch, seq_len, embed_dim]
    // - conditioning_scale: scalar scale factor [1] (fp16)

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Prepare all tensor data
    // 1. hidden_states (latent_model_input) - fp16
    std::vector<int64_t> hs_shape = {batch, channels, latent_h, latent_w};
    size_t hs_size = batch * channels * latent_h * latent_w;
    std::vector<Ort::Float16_t> hs_fp16(hs_size);
    for (size_t i = 0; i < hs_size; i++) {
        hs_fp16[i] = Ort::Float16_t(latent_input[i]);
    }

    // 2. controlnet_cond (VAE-encoded control image in latent space) - fp16
    // control_cond as received from the pipeline is always single-batch;
    // CFG batching happens here, keyed on the same `batch` the denoiser computed
    // from the encoder's actual output (not guidance_scale). Channel count is
    // derived from the single-batch buffer rather than assumed equal to the main
    // latent's `channels`: mask-aware ControlNets (e.g. SD3 ControlNet-Inpainting)
    // concatenate a 1-channel downsampled mask onto the 16-channel masked-image
    // latents, producing a 17-channel cond tensor (extra_conditioning_channels=1
    // in the model config). Declaring the ONNX shape as `channels` here instead
    // caused a shape mismatch against the actual buffer size.
    size_t cond_hw = static_cast<size_t>(latent_h) * static_cast<size_t>(latent_w);
    size_t cond_channels = (cond_hw > 0)
        ? control_cond.size() / cond_hw
        : static_cast<size_t>(channels);
    if (cond_channels == 0) cond_channels = static_cast<size_t>(channels);
    std::vector<int64_t> cond_shape = {batch, static_cast<int64_t>(cond_channels), latent_h, latent_w};
    size_t single_cond_size = cond_channels * cond_hw;
    size_t cond_size = static_cast<size_t>(batch) * single_cond_size;
    std::vector<Ort::Float16_t> cond_fp16(cond_size);
    for (int b = 0; b < batch; b++) {
        for (size_t i = 0; i < single_cond_size && i < control_cond.size(); i++) {
            cond_fp16[static_cast<size_t>(b) * single_cond_size + i] = Ort::Float16_t(control_cond[i]);
        }
    }

    // 3. conditioning_scale - fp16 scalar
    std::vector<int64_t> scale_shape = {1};
    std::vector<Ort::Float16_t> scale_fp16 = {Ort::Float16_t(conditioning_scale)};

    // 4. encoder_hidden_states - fp16 (optional)
    std::vector<int64_t> emb_shape = {batch, seq_len, embed_dim};
    size_t emb_size = batch * seq_len * embed_dim;
    std::vector<Ort::Float16_t> emb_fp16(emb_size);
    for (size_t i = 0; i < emb_size && i < text_emb.size(); i++) {
        emb_fp16[i] = Ort::Float16_t(text_emb[i]);
    }

    // 5. pooled_projections - fp16
    // Mask-aware ControlNets (SD3 ControlNet-Inpainting) are trained with
    // force_zeros_for_pooled_projection=true and expect an all-zero pooled
    // projection input regardless of the actual pooled text embeddings.
    std::vector<int64_t> pooled_shape = {batch, pool_dim};
    size_t pooled_size = batch * pool_dim;
    std::vector<Ort::Float16_t> pooled_fp16(pooled_size);
    if (!is_inpainting_controlnet(type_)) {
        for (size_t i = 0; i < pooled_size && i < pooled_emb.size(); i++) {
            pooled_fp16[i] = Ort::Float16_t(pooled_emb[i]);
        }
    }

    // 6. timestep - fp16, must match batch dimension (DD models key on all dims)
    std::vector<int64_t> ts_shape = {batch};
    std::vector<Ort::Float16_t> ts_fp16(batch, Ort::Float16_t(timestep));

    // Build inputs in the order the model expects (match Python implementation)
    // Check model input names to determine order
    const auto& input_names = model_->get_input_names();
    std::vector<Ort::Value> inputs;
    std::vector<const char*> input_name_ptrs;

    // The input shapes are static for the lifetime of a request (batch/channels/
    // latent size/seq_len/embed_dim don't change between denoising steps), so
    // only dump them once instead of on every step to avoid log spam.
    const bool log_shapes = !logged_shapes_;
    if (log_shapes) {
        std::cout << "  ControlNet inputs:" << std::endl;
    }
    for (const auto& name : input_names) {
        input_name_ptrs.push_back(name.c_str());
        
        if (name == "hidden_states") {
            if (log_shapes) std::cout << "    " << name << ": [" << batch << ", " << channels << ", " << latent_h << ", " << latent_w << "]" << std::endl;
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                mem, hs_fp16.data(), hs_fp16.size(), hs_shape.data(), hs_shape.size()));
        } else if (name == "controlnet_cond") {
            if (log_shapes) std::cout << "    " << name << ": [" << batch << ", " << cond_channels << ", " << latent_h << ", " << latent_w << "]" << std::endl;
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                mem, cond_fp16.data(), cond_fp16.size(), cond_shape.data(), cond_shape.size()));
        } else if (name == "conditioning_scale") {
            if (log_shapes) std::cout << "    " << name << ": [1]" << std::endl;
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                mem, scale_fp16.data(), scale_fp16.size(), scale_shape.data(), scale_shape.size()));
        } else if (name == "encoder_hidden_states") {
            if (log_shapes) std::cout << "    " << name << ": [" << batch << ", " << seq_len << ", " << embed_dim << "]" << std::endl;
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                mem, emb_fp16.data(), emb_fp16.size(), emb_shape.data(), emb_shape.size()));
        } else if (name == "pooled_projections") {
            if (log_shapes) std::cout << "    " << name << ": [" << batch << ", " << pool_dim << "]" << std::endl;
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                mem, pooled_fp16.data(), pooled_fp16.size(), pooled_shape.data(), pooled_shape.size()));
        } else if (name == "timestep") {
            if (log_shapes) std::cout << "    " << name << ": [" << batch << "]" << std::endl;
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                mem, ts_fp16.data(), ts_fp16.size(), ts_shape.data(), ts_shape.size()));
        } else {
            std::cerr << "  WARNING: Unknown ControlNet input: " << name << std::endl;
        }
    }
    logged_shapes_ = true;

    // Get output names
    const auto& output_names = model_->get_output_names();
    std::vector<const char*> output_name_ptrs;
    for (const auto& name : output_names) {
        output_name_ptrs.push_back(name.c_str());
    }

    // Run ControlNet model with named inputs/outputs
    auto outputs = model_->run(inputs, input_name_ptrs, output_name_ptrs);

    if (outputs.empty()) {
        std::cout << "  WARNING: ControlNet produced no outputs" << std::endl;
        return {};
    }

    // Extract block outputs
    size_t num_blocks = outputs.size();
    std::cout << "  ControlNet produced " << num_blocks << " block outputs" << std::endl;

    std::vector<std::vector<Ort::Float16_t>> ctrl_fp16_out;

    if (num_blocks == 6) {
        // Legacy SD3 ControlNets (Canny/Pose/Tile/Depth) emit one block per
        // pair of transformer layers; duplicate each to fill all 12 slots.
        duplicate_blocks(outputs, ctrl_fp16_out);
    } else {
        // Otherwise pass the blocks straight through (e.g. mask-aware
        // ControlNets like SD3-Controlnet-Inpainting emit one block per
        // transformer layer -- 23 for SD3 Medium's 23-layer MMDiT).
        extract_blocks(outputs, ctrl_fp16_out);
    }

    return ctrl_fp16_out;
}

void ControlNetRunner::duplicate_blocks(
    const std::vector<Ort::Value>& outputs,
    std::vector<std::vector<Ort::Float16_t>>& ctrl_fp16_out) {

    ctrl_fp16_out.resize(12);
    
    for (int i = 0; i < 6; i++) {
        auto& block = outputs[i];
        auto shape = block.GetTensorTypeAndShapeInfo().GetShape();
        size_t block_size = 1;
        for (auto s : shape) block_size *= s;

        const Ort::Float16_t* data = block.GetTensorData<Ort::Float16_t>();
        std::vector<Ort::Float16_t> block_data(data, data + block_size);

        // Duplicate: blocks 2*i and 2*i+1 get the same data
        ctrl_fp16_out[2 * i] = block_data;
        ctrl_fp16_out[2 * i + 1] = block_data;
    }
}

void ControlNetRunner::extract_blocks(
    const std::vector<Ort::Value>& outputs,
    std::vector<std::vector<Ort::Float16_t>>& ctrl_fp16_out) {

    ctrl_fp16_out.resize(outputs.size());

    for (size_t i = 0; i < outputs.size(); i++) {
        auto& block = outputs[i];
        auto shape = block.GetTensorTypeAndShapeInfo().GetShape();
        size_t block_size = 1;
        for (auto s : shape) block_size *= s;

        const Ort::Float16_t* data = block.GetTensorData<Ort::Float16_t>();
        ctrl_fp16_out[i] = std::vector<Ort::Float16_t>(data, data + block_size);
    }
}

} // namespace sd_npu
