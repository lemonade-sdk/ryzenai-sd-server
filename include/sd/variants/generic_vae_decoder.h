// generic_vae_decoder.h - Parameterized VAE decoder shared by SD1.5, SDXL, SD3/SD3.5, and FLUX
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "i_vae_decoder.h"
#include "onnx_model.h"
#include <map>
#include <memory>
#include <string>

namespace sd_npu {

// ============================================================================
// GenericVaeDecoder: Standard "scale, run VAE, denormalize" decoder used by
// every variant whose VAE follows the diffusers convention:
//
//     scaled_latents = latents / scaling_factor + shift_factor
//     image = (vae(scaled_latents) + 1) / 2, clamped to [0, 1]
//
// Variants only differ in their scaling_factor / shift_factor / channel count
// and the human-readable label used in log output — everything else
// (fp16/fp32 input dispatch, timing, output decode) is identical, so a single
// implementation is shared instead of one copy per variant.
// ============================================================================

class GenericVaeDecoder : public IVaeDecoder {
public:
    GenericVaeDecoder(
        std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
        float scaling_factor,
        float shift_factor,
        int channels,
        std::string label);

    std::vector<uint8_t> decode(
        const std::vector<float>& latents,
        int height,
        int width) override;

    std::vector<ComponentType> required_components() const override {
        return {ComponentType::VAE_DECODER};
    }

    float latent_scaling_factor() const override { return scaling_factor_; }
    float latent_shift_factor() const override { return shift_factor_; }

private:
    std::vector<uint8_t> convert_image_f32(float* data, int out_h, int out_w, bool print_stats = false);
    std::vector<uint8_t> convert_image_f16(Ort::Float16_t* data, int out_h, int out_w);

    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components_;
    float scaling_factor_;
    float shift_factor_;
    int channels_;
    std::string label_;
};

} // namespace sd_npu
