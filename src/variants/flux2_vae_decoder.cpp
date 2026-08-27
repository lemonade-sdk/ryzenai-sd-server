// flux2_vae_decoder.cpp - FLUX.2-klein VAE decoder implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variants/flux2_vae_decoder.h"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace sd_npu {

Flux2VaeDecoder::Flux2VaeDecoder(
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
    : components_(components) {}

std::vector<uint8_t> Flux2VaeDecoder::decode(
    const std::vector<float>& latents,
    int height, int width) {

    std::cout << "Decoding latents to image (FLUX.2)..." << std::endl;

    if (!components_.count(ComponentType::VAE_DECODER)) {
        std::cout << "  WARNING: No VAE decoder loaded, returning blank image" << std::endl;
        return std::vector<uint8_t>(static_cast<size_t>(width) * height * 3, 128);
    }

    auto& vae = components_[ComponentType::VAE_DECODER];

    const int channels = 32;
    const int latent_h = height / 8;
    const int latent_w = width  / 8;
    const size_t latent_size = static_cast<size_t>(channels) * latent_h * latent_w;

    if (latents.size() < latent_size) {
        std::cout << "  WARNING: latent buffer too small (" << latents.size()
                  << " < " << latent_size << "), returning blank image" << std::endl;
        return std::vector<uint8_t>(static_cast<size_t>(width) * height * 3, 128);
    }

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, channels, latent_h, latent_w};

    auto in_type = vae->get_input_type(0);
    std::vector<Ort::Value> inputs;
    std::vector<Ort::Float16_t> fp16;

    if (in_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        fp16.resize(latent_size);
        for (size_t i = 0; i < latent_size; i++) fp16[i] = Ort::Float16_t(latents[i]);
        inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
            mem, fp16.data(), fp16.size(), shape.data(), shape.size()));
    } else {
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(latents.data()), latent_size,
            shape.data(), shape.size()));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto outputs = vae->run(inputs);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "  VAE decode: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms" << std::endl;

    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    auto oshape = info.GetShape();  // [1, 3, H, W]
    int out_h = (oshape.size() >= 4) ? static_cast<int>(oshape[2]) : height;
    int out_w = (oshape.size() >= 4) ? static_cast<int>(oshape[3]) : width;
    size_t count = static_cast<size_t>(3) * out_h * out_w;

    std::vector<float> chw(count);
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        auto* p = outputs[0].GetTensorMutableData<Ort::Float16_t>();
        for (size_t i = 0; i < count; i++) chw[i] = p[i].ToFloat();
    } else {
        float* p = outputs[0].GetTensorMutableData<float>();
        std::copy(p, p + count, chw.begin());
    }

    return convert_image(chw, out_h, out_w);
}

std::vector<uint8_t> Flux2VaeDecoder::convert_image(
    const std::vector<float>& chw, int out_h, int out_w) {
    std::vector<uint8_t> image(static_cast<size_t>(out_h) * out_w * 3);
    const size_t plane = static_cast<size_t>(out_h) * out_w;
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            for (int c = 0; c < 3; c++) {
                float val = chw[c * plane + static_cast<size_t>(y) * out_w + x];
                val = (val + 1.0f) * 0.5f;
                val = std::max(0.0f, std::min(1.0f, val));
                image[(static_cast<size_t>(y) * out_w + x) * 3 + c] =
                    static_cast<uint8_t>(val * 255.0f);
            }
        }
    }
    return image;
}

} // namespace sd_npu
