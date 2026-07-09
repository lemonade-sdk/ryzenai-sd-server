// sd3_variant.cpp - SD3 and SD3.5 variant registration
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variant_registry.h"
#include "variants/sd3_text_encoder.h"
#include "variants/sd3_vae_decoder.h"
#include "variants/generic_denoiser.h"

namespace sd_npu {

static VariantDescriptor make_sd3_descriptor() {
    VariantDescriptor d;
    d.name = "sd3";
    d.aliases = {"sd3.0"};
    d.variant = ModelVariant::SD3;

    d.default_width  = 1024;
    d.default_height = 1024;
    d.default_steps  = 28;
    d.default_guidance = 7.0f;
    d.latent_channels = 16;
    d.default_scheduler = SchedulerType::FLOW_MATCH_EULER;

    d.has_common_dir = true;
    d.sub_model_dir = "normal";

    d.components = {
        {"text_encoder", ComponentType::TEXT_ENCODER, true,
         {"text_encoder/text_encoder.onnx",
          "text_encoder/text_encoder_fp16.onnx",
          "text_encoder/model.onnx"}, true},
        {"text_encoder_2", ComponentType::TEXT_ENCODER_2, true,
         {"text_encoder_2/text_encoder_2.onnx",
          "text_encoder_2/text_encoder_2_fp16.onnx",
          "text_encoder_2/model.onnx"}, true},
        {"text_encoder_3", ComponentType::TEXT_ENCODER_3, false,
         {"text_encoder_3/text_encoder_3.onnx",
          "text_encoder_3/text_encoder_3_fp16.onnx",
          "text_encoder_3/dd/replaced.onnx",
          "text_encoder_3/model.onnx"}, true},
        {"transformer", ComponentType::TRANSFORMER, true,
         {"transformer/dd/replaced.onnx",
          "transformer/dynamic/dd/replaced.onnx"}, false},
        {"vae_decoder", ComponentType::VAE_DECODER, true,
         {"vae_decoder/dd/replaced.onnx",
          "vae_decoder/dynamic/dd/replaced.onnx"}, true},
        {"vae_encoder", ComponentType::VAE_ENCODER, false,
         {"vae_encoder/dd/replaced.onnx",
          "vae_encoder/dynamic/dd/replaced.onnx",
          "vae_encoder/model.onnx",
          "vae_encoder/vae_encoder.onnx"}, true},
    };

    // SD3: has transformer dir under normal/, no "3.5" in path
    d.detect = [](const std::filesystem::path& base) -> int {
        namespace fs = std::filesystem;
        if (fs::exists(base / "normal" / "transformer") ||
            fs::exists(base / "normal" / "transformer" / "dd")) {
            // Check the full path (not just the leaf dir, which may be a hash)
            std::string full = base.string();
            std::string lower;
            lower.resize(full.size());
            std::transform(full.begin(), full.end(), lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (lower.find("3.5") != std::string::npos || lower.find("3_5") != std::string::npos) {
                return 0;
            }
            return 10;
        }
        return 0;
    };

    d.create_encoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           CLIPTokenizer& tok, CLIPTokenizer& tok2, const SDConfig&)
        -> std::unique_ptr<ITextEncoder> {
        return std::make_unique<SD3TextEncoder>(components, tok, tok2);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner* cn, const SDConfig&)
        -> std::unique_ptr<IDenoiser> {
        return std::make_unique<GenericDenoiser>(
            components, DenoiserSpec{ComponentType::TRANSFORMER, 160, 4096, 2048, 16}, cn);
    };
    d.create_vae_decoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
        -> std::unique_ptr<IVaeDecoder> {
        return std::make_unique<SD3VaeDecoder>(components);
    };

    return d;
}

static VariantDescriptor make_sd35_descriptor() {
    VariantDescriptor d;
    d.name = "sd35";
    d.aliases = {"sd3.5"};
    d.variant = ModelVariant::SD35;

    d.default_width  = 1024;
    d.default_height = 1024;
    d.default_steps  = 28;
    d.default_guidance = 4.5f;
    d.latent_channels = 16;
    d.default_scheduler = SchedulerType::FLOW_MATCH_EULER;

    d.has_common_dir = true;
    d.sub_model_dir = "normal";

    d.components = {
        {"text_encoder", ComponentType::TEXT_ENCODER, true,
         {"text_encoder/text_encoder.onnx",
          "text_encoder/text_encoder_fp16.onnx",
          "text_encoder/model.onnx"}, true},
        {"text_encoder_2", ComponentType::TEXT_ENCODER_2, true,
         {"text_encoder_2/text_encoder_2.onnx",
          "text_encoder_2/text_encoder_2_fp16.onnx",
          "text_encoder_2/model.onnx"}, true},
        {"text_encoder_3", ComponentType::TEXT_ENCODER_3, false,
         {"text_encoder_3/text_encoder_3.onnx",
          "text_encoder_3/text_encoder_3_fp16.onnx",
          "text_encoder_3/dd/replaced.onnx",
          "text_encoder_3/model.onnx"}, true},
        {"transformer", ComponentType::TRANSFORMER, true,
         {"transformer/dd/replaced.onnx",
          "transformer/dynamic/dd/replaced.onnx"}, false},
        {"vae_decoder", ComponentType::VAE_DECODER, true,
         {"vae_decoder/dd/replaced.onnx",
          "vae_decoder/dynamic/dd/replaced.onnx"}, true},
        {"vae_encoder", ComponentType::VAE_ENCODER, false,
         {"vae_encoder/dd/replaced.onnx",
          "vae_encoder/dynamic/dd/replaced.onnx",
          "vae_encoder/model.onnx",
          "vae_encoder/vae_encoder.onnx"}, true},
    };

    // SD3.5: has transformer dir, "3.5" or "3_5" in path
    d.detect = [](const std::filesystem::path& base) -> int {
        namespace fs = std::filesystem;
        if (fs::exists(base / "normal" / "transformer") ||
            fs::exists(base / "normal" / "transformer" / "dd")) {
            // Check the full path (not just the leaf dir, which may be a hash)
            std::string full = base.string();
            std::string lower;
            lower.resize(full.size());
            std::transform(full.begin(), full.end(), lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (lower.find("3.5") != std::string::npos || lower.find("3_5") != std::string::npos) {
                return 15;
            }
            // If we can't tell 3.0 from 3.5, fallback to SD3
            return 0;
        }
        return 0;
    };

    d.create_encoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           CLIPTokenizer& tok, CLIPTokenizer& tok2, const SDConfig&)
        -> std::unique_ptr<ITextEncoder> {
        return std::make_unique<SD3TextEncoder>(components, tok, tok2);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner* cn, const SDConfig&)
        -> std::unique_ptr<IDenoiser> {
        return std::make_unique<GenericDenoiser>(
            components, DenoiserSpec{ComponentType::TRANSFORMER, 160, 4096, 2048, 16}, cn);
    };
    d.create_vae_decoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
        -> std::unique_ptr<IVaeDecoder> {
        return std::make_unique<SD3VaeDecoder>(components);
    };

    return d;
}

REGISTER_VARIANT(make_sd3_descriptor)
REGISTER_VARIANT(make_sd35_descriptor)

} // namespace sd_npu
