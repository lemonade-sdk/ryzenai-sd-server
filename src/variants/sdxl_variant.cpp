// sdxl_variant.cpp - SDXL and SDXL-Turbo variant registration
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variant_registry.h"
#include "variants/sdxl_text_encoder.h"
#include "variants/sdxl_vae_decoder.h"
#include "variants/sdxl_denoiser.h"

namespace sd_npu {

static VariantDescriptor make_sdxl_descriptor() {
    VariantDescriptor d;
    d.name = "sdxl";
    d.aliases = {"xl"};
    d.variant = ModelVariant::SDXL;

    d.default_width  = 1024;
    d.default_height = 1024;
    d.default_steps  = 20;
    d.default_guidance = 5.0f;
    d.latent_channels = 4;
    d.default_scheduler = SchedulerType::EULER_DISCRETE;

    d.has_common_dir = false;

    d.components = {
        {"text_encoder", ComponentType::TEXT_ENCODER, true,
         {"text_encoder/text_encoder.onnx",
          "text_encoder/text_encoder_fp16.onnx",
          "text_encoder/model.onnx"}, false},
        {"text_encoder_2", ComponentType::TEXT_ENCODER_2, true,
         {"text_encoder_2/text_encoder_2.onnx",
          "text_encoder_2/text_encoder_2_fp16.onnx",
          "text_encoder_2/model.onnx"}, false},
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

    // SDXL: has text_encoder_2 and unet, no transformer, not turbo
    d.detect = [](const std::filesystem::path& base) -> int {
        namespace fs = std::filesystem;
        if (fs::exists(base / "normal" / "transformer")) return 0;
        if (!fs::exists(base / "text_encoder_2")) return 0;
        if (fs::exists(base / "unet") || fs::exists(base / "unet/dd")) {
            // Check the full path for "turbo" (not just the leaf dir, which may be a hash)
            std::string full = base.string();
            std::string lower;
            lower.resize(full.size());
            std::transform(full.begin(), full.end(), lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (lower.find("turbo") != std::string::npos) return 0;
            return 10;
        }
        return 0;
    };

    d.create_encoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           CLIPTokenizer& tok, CLIPTokenizer& tok2, const SDConfig&)
        -> std::unique_ptr<ITextEncoder> {
        return std::make_unique<SDXLTextEncoder>(components, tok, tok2, false);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner*, const SDConfig&)
        -> std::unique_ptr<IDenoiser> {
        return std::make_unique<SDXLDenoiser>(components);
    };
    d.create_vae_decoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
        -> std::unique_ptr<IVaeDecoder> {
        return std::make_unique<SDXLVaeDecoder>(components);
    };

    return d;
}

static VariantDescriptor make_sdxl_turbo_descriptor() {
    VariantDescriptor d;
    d.name = "sdxl-turbo";
    d.aliases = {"xl-turbo"};
    d.variant = ModelVariant::SDXL_TURBO;

    d.default_width  = 512;
    d.default_height = 512;
    d.default_steps  = 4;
    d.default_guidance = 0.0f;
    d.latent_channels = 4;
    d.default_scheduler = SchedulerType::EULER_DISCRETE;

    d.has_common_dir = false;

    d.components = {
        {"text_encoder", ComponentType::TEXT_ENCODER, true,
         {"text_encoder/text_encoder.onnx",
          "text_encoder/text_encoder_fp16.onnx",
          "text_encoder/model.onnx"}, false},
        {"text_encoder_2", ComponentType::TEXT_ENCODER_2, true,
         {"text_encoder_2/text_encoder_2.onnx",
          "text_encoder_2/text_encoder_2_fp16.onnx",
          "text_encoder_2/model.onnx"}, false},
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

    // SDXL-Turbo: has text_encoder_2, unet, "turbo" in path
    d.detect = [](const std::filesystem::path& base) -> int {
        namespace fs = std::filesystem;
        if (fs::exists(base / "normal" / "transformer")) return 0;
        if (!fs::exists(base / "text_encoder_2")) return 0;
        if (fs::exists(base / "unet") || fs::exists(base / "unet/dd")) {
            // Check the full path for "turbo" (not just the leaf dir, which may be a hash)
            std::string full = base.string();
            std::string lower;
            lower.resize(full.size());
            std::transform(full.begin(), full.end(), lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (lower.find("turbo") != std::string::npos) return 15;
        }
        return 0;
    };

    d.create_encoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           CLIPTokenizer& tok, CLIPTokenizer& tok2, const SDConfig&)
        -> std::unique_ptr<ITextEncoder> {
        return std::make_unique<SDXLTextEncoder>(components, tok, tok2, true);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner*, const SDConfig&)
        -> std::unique_ptr<IDenoiser> {
        return std::make_unique<SDXLDenoiser>(components);
    };
    d.create_vae_decoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
        -> std::unique_ptr<IVaeDecoder> {
        return std::make_unique<SDXLVaeDecoder>(components);
    };

    return d;
}

REGISTER_VARIANT(make_sdxl_descriptor)
REGISTER_VARIANT(make_sdxl_turbo_descriptor)

} // namespace sd_npu
