// sd3_vae_decoder.h - SD3/SD3.5 VAE decoder implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "i_vae_decoder.h"
#include "onnx_model.h"
#include <map>
#include <memory>

namespace sd_npu {

// ============================================================================
// SD3VaeDecoder: SD3/SD3.5 VAE decoder (16 channels, scaling factor 1.5305)
// ============================================================================

class SD3VaeDecoder : public IVaeDecoder {
public:
    SD3VaeDecoder(std::map<ComponentType, std::unique_ptr<OnnxModel>>& components);

    std::vector<uint8_t> decode(
        const std::vector<float>& latents,
        int height,
        int width) override;

    std::vector<ComponentType> required_components() const override {
        return {ComponentType::VAE_DECODER};
    }

    float latent_scaling_factor() const override {
        return 1.5305f;
    }

private:
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components_;

    static std::vector<uint8_t> convert_image_f32(
        float* data, int out_h, int out_w, bool print_stats = false);

    static std::vector<uint8_t> convert_image_f16(
        Ort::Float16_t* data, int out_h, int out_w);
};

} // namespace sd_npu
