// sd15_variant.cpp - SD1.5 variant registration
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variant_registry.h"
#include "variants/sd15_text_encoder.h"
#include "variants/sd15_vae_decoder.h"
#include "variants/generic_denoiser.h"

namespace sd_npu {

static VariantDescriptor make_sd15_descriptor() {
    VariantDescriptor d;
    d.name = "sd15";
    d.aliases = {"sd1.5", "1.5"};
    d.variant = ModelVariant::SD15;

    d.default_width  = 512;
    d.default_height = 512;
    d.default_steps  = 20;
    d.default_guidance = 7.5f;
    d.latent_channels = 4;
    d.default_scheduler = SchedulerType::EULER_DISCRETE;

    d.has_common_dir = false;

    d.components = {
        {"text_encoder", ComponentType::TEXT_ENCODER, true,
         {"text_encoder/text_encoder.onnx",
          "text_encoder/text_encoder_fp16.onnx",
          "text_encoder/model.onnx"}, false},
        {"unet", ComponentType::UNET, true,
         {"unet/dd/replaced.onnx",
          "unet/dynamic/dd/replaced.onnx"}, false},
        {"vae_decoder", ComponentType::VAE_DECODER, true,
         {"vae_decoder/dd/replaced.onnx",
          "vae_decoder/dynamic/dd/replaced.onnx"}, false},
        {"vae_encoder", ComponentType::VAE_ENCODER, false,
         {"vae_encoder/dd/replaced.onnx",
          "vae_encoder/dynamic/dd/replaced.onnx",
          "vae_encoder/model.onnx",
          "vae_encoder/vae_encoder.onnx"}, false},
    };

    // SD1.5: has text_encoder, has unet, no text_encoder_2, no transformer, not turbo
    d.detect = [](const std::filesystem::path& base) -> int {
        namespace fs = std::filesystem;
        if (fs::exists(base / "normal" / "transformer")) return 0;
        if (fs::exists(base / "text_encoder_2")) return 0;
        // Check the full path for "turbo" (not just the leaf dir, which may be a hash)
        std::string full = base.string();
        std::string lower;
        lower.resize(full.size());
        std::transform(full.begin(), full.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lower.find("turbo") != std::string::npos) return 0;
        return 10;
    };

    d.create_encoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           CLIPTokenizer& tok, CLIPTokenizer&, const SDConfig&)
        -> std::unique_ptr<ITextEncoder> {
        return std::make_unique<SD15TextEncoder>(components, tok, false);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner*, const SDConfig&)
        -> std::unique_ptr<IDenoiser> {
        return std::make_unique<GenericDenoiser>(
            components, DenoiserSpec{ComponentType::UNET, 77, 768, 0, 4});
    };
    d.create_vae_decoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
        -> std::unique_ptr<IVaeDecoder> {
        return std::make_unique<SD15VaeDecoder>(components);
    };

    return d;
}

static VariantDescriptor make_sd_turbo_descriptor() {
    VariantDescriptor d;
    d.name = "sd-turbo";
    d.aliases = {"turbo"};
    d.variant = ModelVariant::SD_TURBO;

    d.default_width  = 512;
    d.default_height = 512;
    d.default_steps  = 1;
    d.default_guidance = 0.0f;
    d.latent_channels = 4;
    d.default_scheduler = SchedulerType::EULER_DISCRETE;

    d.has_common_dir = false;

    d.components = {
        {"text_encoder", ComponentType::TEXT_ENCODER, true,
         {"text_encoder/text_encoder.onnx",
          "text_encoder/text_encoder_fp16.onnx",
          "text_encoder/model.onnx"}, false},
        {"unet", ComponentType::UNET, true,
         {"unet/dd/replaced.onnx",
          "unet/dynamic/dd/replaced.onnx"}, false},
        {"vae_decoder", ComponentType::VAE_DECODER, true,
         {"vae_decoder/dd/replaced.onnx",
          "vae_decoder/dynamic/dd/replaced.onnx"}, false},
        {"vae_encoder", ComponentType::VAE_ENCODER, false,
         {"vae_encoder/dd/replaced.onnx",
          "vae_encoder/dynamic/dd/replaced.onnx",
          "vae_encoder/model.onnx",
          "vae_encoder/vae_encoder.onnx"}, false},
    };

    // SD-Turbo: has "turbo" in path, no text_encoder_2, no transformer
    d.detect = [](const std::filesystem::path& base) -> int {
        namespace fs = std::filesystem;
        if (fs::exists(base / "normal" / "transformer")) return 0;
        if (fs::exists(base / "text_encoder_2")) return 0;
        // Check the full path for "turbo" (not just the leaf dir, which may be a hash)
        std::string full = base.string();
        std::string lower;
        lower.resize(full.size());
        std::transform(full.begin(), full.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lower.find("turbo") == std::string::npos) return 0;
        return 15;  // Higher priority than plain SD1.5
    };

    d.create_encoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           CLIPTokenizer& tok, CLIPTokenizer&, const SDConfig&)
        -> std::unique_ptr<ITextEncoder> {
        return std::make_unique<SD15TextEncoder>(components, tok, true);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner*, const SDConfig&)
        -> std::unique_ptr<IDenoiser> {
        return std::make_unique<GenericDenoiser>(
            components, DenoiserSpec{ComponentType::UNET, 77, 1024, 0, 4});
    };
    d.create_vae_decoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
        -> std::unique_ptr<IVaeDecoder> {
        return std::make_unique<SD15VaeDecoder>(components);
    };

    return d;
}

REGISTER_VARIANT(make_sd15_descriptor)
REGISTER_VARIANT(make_sd_turbo_descriptor)

} // namespace sd_npu
