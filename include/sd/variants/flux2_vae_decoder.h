// flux2_vae_decoder.h - FLUX.2-klein VAE decoder (32 channels)
// Copyright (C) 2025 Advanced Micro Devices, Inc.
//
// The klein VAE decoder consumes 32-channel latents at H/8 x W/8 resolution.
// The BatchNorm denorm and unpatchify are performed by Flux2Denoiser (they must
// run in packed 128-channel space), so this decoder simply runs the ONNX model
// and converts the [1,3,H,W] output to RGB.

#pragma once

#include "i_vae_decoder.h"
#include "onnx_model.h"
#include <map>
#include <memory>
#include <vector>

namespace sd_npu {

class Flux2VaeDecoder : public IVaeDecoder {
public:
    explicit Flux2VaeDecoder(
        std::map<ComponentType, std::unique_ptr<OnnxModel>>& components);

    std::vector<uint8_t> decode(
        const std::vector<float>& latents,
        int height,
        int width) override;

    std::vector<ComponentType> required_components() const override {
        return {ComponentType::VAE_DECODER};
    }

    float latent_scaling_factor() const override { return 1.0f; }

private:
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components_;

    static std::vector<uint8_t> convert_image(
        const std::vector<float>& chw, int out_h, int out_w);
};

} // namespace sd_npu
