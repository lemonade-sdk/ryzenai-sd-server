// flux_vae_decoder.h - FLUX VAE decoder implementation (16 channels, FLUX scaling)
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "i_vae_decoder.h"
#include "onnx_model.h"
#include <map>
#include <memory>

namespace sd_npu {

// ============================================================================
// FluxVaeDecoder: 16-channel VAE decoder for FLUX models.
//
// Same architecture as SD3VaeDecoder but with FLUX-specific scaling factors:
//   scaling_factor = 0.3611
//   shift_factor   = 0.1159
// ============================================================================

class FluxVaeDecoder : public IVaeDecoder {
public:
    FluxVaeDecoder(std::map<ComponentType, std::unique_ptr<OnnxModel>>& components);

    std::vector<uint8_t> decode(
        const std::vector<float>& latents,
        int height,
        int width) override;

    std::vector<ComponentType> required_components() const override {
        return {ComponentType::VAE_DECODER};
    }

    float latent_scaling_factor() const override {
        return 0.3611f;
    }

private:
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components_;

    static std::vector<uint8_t> convert_image_f32(float* data, int h, int w);
    static std::vector<uint8_t> convert_image_f16(Ort::Float16_t* data, int h, int w);
};

} // namespace sd_npu
