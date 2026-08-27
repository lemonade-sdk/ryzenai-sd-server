// sdxl_derived_variants.cpp - Playground v2.5, Dreamshaper XL Lightning, SSD-1B variant registration
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variant_registry.h"
#include "variants/sdxl_text_encoder.h"
#include "variants/sdxl_vae_decoder.h"
#include "variants/generic_denoiser.h"
#include <algorithm>
#include <cctype>

namespace sd_npu {

namespace {
// Common SDXL component search paths
std::vector<ComponentSpec> sdxl_components() {
    return {
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
}

// Check that a model dir looks like an SDXL layout (text_encoder_2 + unet, no transformer)
bool is_sdxl_layout(const std::filesystem::path& base) {
    namespace fs = std::filesystem;
    if (fs::exists(base / "normal" / "transformer")) return false;
    if (fs::exists(base / "transformer")) return false;
    if (!fs::exists(base / "text_encoder_2")) return false;
    return fs::exists(base / "unet") || fs::exists(base / "unet" / "dd");
}

std::string to_lower(const std::string& s) {
    std::string lower(s.size(), '\0');
    std::transform(s.begin(), s.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower;
}
} // anonymous namespace

// ============================================================================
// Playground v2.5
// ============================================================================

static VariantDescriptor make_playground_v25_descriptor() {
    VariantDescriptor d;
    d.name    = "playground-v25";
    d.aliases = {"playground-v2.5", "playground2.5"};
    d.variant = ModelVariant::PLAYGROUND_V25;

    d.default_width    = 1024;
    d.default_height   = 1024;
    d.default_steps    = 20;
    d.default_guidance = 3.0f;
    d.latent_channels  = 4;
    d.default_scheduler = SchedulerType::EULER_DISCRETE;

    d.has_common_dir = false;
    d.components = sdxl_components();

    // Playground v2.5: SDXL layout with "playground" and "2.5" (or "v2") in path
    d.detect = [](const std::filesystem::path& base) -> int {
        if (!is_sdxl_layout(base)) return 0;
        const auto lower = to_lower(base.string());
        if (lower.find("playground") == std::string::npos) return 0;
        if (lower.find("2.5") == std::string::npos &&
            lower.find("2_5") == std::string::npos &&
            lower.find("v2")  == std::string::npos) return 0;
        return 15;
    };

    d.create_encoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           CLIPTokenizer& tok, CLIPTokenizer& tok2, const SDConfig&)
        -> std::unique_ptr<ITextEncoder> {
        return std::make_unique<SDXLTextEncoder>(components, tok, tok2, false);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner*, const SDConfig&)
        -> std::unique_ptr<IDenoiser> {
        return std::make_unique<GenericDenoiser>(
            components, DenoiserSpec{ComponentType::UNET, 77, 2048, 1280, 4});
    };
    d.create_vae_decoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
        -> std::unique_ptr<IVaeDecoder> {
        return std::make_unique<SDXLVaeDecoder>(components);
    };

    return d;
}

// ============================================================================
// Dreamshaper XL Lightning
// ============================================================================

static VariantDescriptor make_dreamshaper_xl_lightning_descriptor() {
    VariantDescriptor d;
    d.name    = "dreamshaper-xl-lightning";
    d.aliases = {"dreamshaper-lightning", "dsxl-lightning"};
    d.variant = ModelVariant::DREAMSHAPER_XL_LIGHTNING;

    d.default_width    = 1024;
    d.default_height   = 1024;
    d.default_steps    = 4;
    d.default_guidance = 1.0f;
    d.latent_channels  = 4;
    d.default_scheduler = SchedulerType::EULER_DISCRETE;

    d.has_common_dir = false;
    d.components = sdxl_components();

    // Dreamshaper XL Lightning: SDXL layout with "dreamshaper" and "lightning" in path
    d.detect = [](const std::filesystem::path& base) -> int {
        if (!is_sdxl_layout(base)) return 0;
        const auto lower = to_lower(base.string());
        if (lower.find("dreamshaper") == std::string::npos) return 0;
        if (lower.find("lightning") == std::string::npos) return 0;
        return 15;
    };

    d.create_encoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           CLIPTokenizer& tok, CLIPTokenizer& tok2, const SDConfig&)
        -> std::unique_ptr<ITextEncoder> {
        return std::make_unique<SDXLTextEncoder>(components, tok, tok2, false);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner*, const SDConfig&)
        -> std::unique_ptr<IDenoiser> {
        return std::make_unique<GenericDenoiser>(
            components, DenoiserSpec{ComponentType::UNET, 77, 2048, 1280, 4});
    };
    d.create_vae_decoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
        -> std::unique_ptr<IVaeDecoder> {
        return std::make_unique<SDXLVaeDecoder>(components);
    };

    return d;
}

// ============================================================================
// SSD-1B (distilled SDXL)
// ============================================================================

static VariantDescriptor make_ssd_1b_descriptor() {
    VariantDescriptor d;
    d.name    = "ssd-1b";
    d.aliases = {"ssd1b"};
    d.variant = ModelVariant::SSD_1B;

    d.default_width    = 1024;
    d.default_height   = 1024;
    d.default_steps    = 20;
    d.default_guidance = 5.0f;
    d.latent_channels  = 4;
    d.default_scheduler = SchedulerType::EULER_DISCRETE;

    d.has_common_dir = false;
    d.components = sdxl_components();

    // SSD-1B: SDXL layout with "ssd" and "1b" in path
    d.detect = [](const std::filesystem::path& base) -> int {
        if (!is_sdxl_layout(base)) return 0;
        const auto lower = to_lower(base.string());
        if (lower.find("ssd") == std::string::npos) return 0;
        if (lower.find("1b") == std::string::npos) return 0;
        return 15;
    };

    d.create_encoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           CLIPTokenizer& tok, CLIPTokenizer& tok2, const SDConfig&)
        -> std::unique_ptr<ITextEncoder> {
        return std::make_unique<SDXLTextEncoder>(components, tok, tok2, false);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner*, const SDConfig&)
        -> std::unique_ptr<IDenoiser> {
        return std::make_unique<GenericDenoiser>(
            components, DenoiserSpec{ComponentType::UNET, 77, 2048, 1280, 4});
    };
    d.create_vae_decoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
        -> std::unique_ptr<IVaeDecoder> {
        return std::make_unique<SDXLVaeDecoder>(components);
    };

    return d;
}

REGISTER_VARIANT(make_playground_v25_descriptor)
REGISTER_VARIANT(make_dreamshaper_xl_lightning_descriptor)
REGISTER_VARIANT(make_ssd_1b_descriptor)

} // namespace sd_npu
