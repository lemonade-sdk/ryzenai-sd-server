// sd3_vae_decoder.cpp - SD3/SD3.5 VAE decoder implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variants/sd3_vae_decoder.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace sd_npu {

SD3VaeDecoder::SD3VaeDecoder(
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
    : components_(components) {}

std::vector<uint8_t> SD3VaeDecoder::decode(
    const std::vector<float>& latents,
    int height, int width) {

    std::cout << "Decoding latents to image (SD3)..." << std::endl;

    if (!components_.count(ComponentType::VAE_DECODER)) {
        std::cout << "  WARNING: No VAE decoder loaded, returning blank image" << std::endl;
        return std::vector<uint8_t>(width * height * 3, 128);
    }

    auto& vae = components_[ComponentType::VAE_DECODER];

    const float scaling_factor = 1.5305f;
    const float shift_factor = 0.0609f;
    const int vae_channels = 16;

    int latent_h = height / 8;
    int latent_w = width  / 8;
    size_t latent_size = vae_channels * latent_h * latent_w;

    // Apply: scaled = latents / scaling_factor + shift_factor
    std::vector<float> scaled_latents(latent_size);
    for (size_t i = 0; i < latent_size; i++) {
        scaled_latents[i] = (latents[i] / scaling_factor) + shift_factor;
    }

    std::cout << "  VAE pre-scaling: scaling_factor=" << scaling_factor
              << " shift_factor=" << shift_factor << std::endl;

    // Build input tensor
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> vae_shape = {1, vae_channels, latent_h, latent_w};

    auto vae_input_type = vae->get_input_type(0);
    std::vector<Ort::Value> vae_inputs;

    // Float16 path (DD/NPU models)
    if (vae_input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        std::cout << "  VAE input type: float16 (converting from float32)" << std::endl;

        std::vector<Ort::Float16_t> fp16_latents(latent_size);
        for (size_t i = 0; i < latent_size; i++) {
            fp16_latents[i] = Ort::Float16_t(scaled_latents[i]);
        }
        vae_inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
            mem, fp16_latents.data(), fp16_latents.size(),
            vae_shape.data(), vae_shape.size()));

        auto t0 = std::chrono::high_resolution_clock::now();
        auto outputs = vae->run(vae_inputs);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "  VAE decode: " << ms << " ms" << std::endl;

        auto out_info  = outputs[0].GetTensorTypeAndShapeInfo();
        auto out_shape = out_info.GetShape();
        auto out_type  = out_info.GetElementType();

        std::cout << "  VAE output shape: [";
        for (size_t i = 0; i < out_shape.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << out_shape[i];
        }
        std::cout << "] type=" << (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? "float16" : "float32") << std::endl;

        int out_h = (int)out_shape[2];
        int out_w = (int)out_shape[3];
        std::vector<uint8_t> image;

        if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            image = convert_image_f16(
                outputs[0].GetTensorMutableData<Ort::Float16_t>(), out_h, out_w);
        } else {
            image = convert_image_f32(
                outputs[0].GetTensorMutableData<float>(), out_h, out_w);
        }

        std::cout << "  Output image: " << out_w << "x" << out_h
                  << " (" << image.size() << " bytes)" << std::endl;
        return image;
    }

    // Float32 path
    vae_inputs.push_back(Ort::Value::CreateTensor<float>(
        mem, scaled_latents.data(), scaled_latents.size(),
        vae_shape.data(), vae_shape.size()));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto outputs = vae->run(vae_inputs);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "  VAE decode: " << ms << " ms" << std::endl;

    float* image_data = outputs[0].GetTensorMutableData<float>();
    auto out_info  = outputs[0].GetTensorTypeAndShapeInfo();
    auto out_shape = out_info.GetShape();

    std::cout << "  VAE output shape: [";
    for (size_t i = 0; i < out_shape.size(); i++) {
        if (i > 0) std::cout << ",";
        std::cout << out_shape[i];
    }
    std::cout << "]" << std::endl;

    int out_h = (int)out_shape[2];
    int out_w = (int)out_shape[3];

    auto image = convert_image_f32(image_data, out_h, out_w, /*print_stats=*/true);
    std::cout << "  Output image: " << out_w << "x" << out_h
              << " (" << image.size() << " bytes)" << std::endl;
    return image;
}

std::vector<uint8_t> SD3VaeDecoder::convert_image_f32(
    float* data, int out_h, int out_w, bool print_stats) {

    if (print_stats) {
        int total = 3 * out_h * out_w;
        float min_val = data[0], max_val = data[0];
        double sum = 0.0;
        for (int i = 0; i < total; i++) {
            float v = data[i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            sum += v;
        }
        std::cout << "  VAE output range: [" << min_val << ", " << max_val
                  << "] mean=" << (sum / total) << std::endl;
    }

    std::vector<uint8_t> image(out_h * out_w * 3);
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            for (int c = 0; c < 3; c++) {
                float val = data[c * out_h * out_w + y * out_w + x];
                val = (val + 1.0f) * 0.5f;
                val = std::max(0.0f, std::min(1.0f, val));
                image[(y * out_w + x) * 3 + c] = (uint8_t)(val * 255.0f);
            }
        }
    }
    return image;
}

std::vector<uint8_t> SD3VaeDecoder::convert_image_f16(
    Ort::Float16_t* data, int out_h, int out_w) {

    std::vector<uint8_t> image(out_h * out_w * 3);
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            for (int c = 0; c < 3; c++) {
                float val = static_cast<float>(data[c * out_h * out_w + y * out_w + x]);
                val = (val + 1.0f) * 0.5f;
                val = std::max(0.0f, std::min(1.0f, val));
                image[(y * out_w + x) * 3 + c] = (uint8_t)(val * 255.0f);
            }
        }
    }
    return image;
}

} // namespace sd_npu
