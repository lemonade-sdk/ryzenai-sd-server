// flux_variant.cpp - FLUX.1-schnell and FLUX.2-klein variant registration
// Copyright (C) 2025 Advanced Micro Devices, Inc.
//
// FLUX models use a transformer-based denoiser (not UNet), the FlowMatch Euler
// scheduler, and a 16-channel VAE. Text conditioning uses CLIP-L; T5-XXL is
// optional (FLUX.1-schnell ships T5 as a GPTQ binary, not ONNX, so it is not
// loaded by the server — the transformer still runs using CLIP-only conditioning).
//
// DenoiserSpec notes:
//   - seq_len / embed_dim describe the CLIP-L output (77 tokens, 768 dim).
//     The GenericDenoiser creates the encoder_hidden_states tensor with exactly
//     these dimensions. If the ONNX transformer expects a larger T5-shaped input,
//     update seq_len / embed_dim to match (GenericDenoiser zero-pads from our
//     CLIP data into the declared shape automatically).
//   - pool_dim = 768 (CLIP-L pooled output).
//   - latent_ch = 16 (shared with SD3, same VAE architecture).

#include "variant_registry.h"
#include "variants/flux_text_encoder.h"
#include "variants/flux_vae_decoder.h"
#include "variants/generic_denoiser.h"
#include <algorithm>
#include <cctype>

namespace sd_npu {

namespace {
std::string to_lower_str(const std::string& s) {
    std::string r(s.size(), '\0');
    std::transform(s.begin(), s.end(), r.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return r;
}
} // anonymous namespace

// ============================================================================
// FLUX.1-schnell
//   - CLIP-L text encoder (text_encoder/)
//   - T5 text encoder (text_encoder_3_gptq_v2/) — GPTQ binary, not loaded
//   - Transformer (transformer/dynamic/dd/replaced.onnx)
//   - VAE decoder (vae_decoder/dynamic/dd/replaced.onnx)
//   - FlowMatch Euler, 4 steps, no CFG (guidance_embeds=false)
// ============================================================================

static VariantDescriptor make_flux1_schnell_descriptor() {
    VariantDescriptor d;
    d.name    = "flux1-schnell";
    d.aliases = {"flux.1-schnell", "flux1schnell", "flux-schnell"};
    d.variant = ModelVariant::FLUX1_SCHNELL;

    d.default_width    = 1024;
    d.default_height   = 1024;
    d.default_steps    = 4;
    d.default_guidance = 0.0f;  // distilled, no classifier-free guidance
    d.latent_channels  = 16;
    d.default_scheduler = SchedulerType::FLOW_MATCH_EULER;

    d.has_common_dir = false;

    d.components = {
        {"text_encoder", ComponentType::TEXT_ENCODER, true,
         {"text_encoder/model.onnx",
          "text_encoder/text_encoder.onnx"}, false},
        {"transformer", ComponentType::TRANSFORMER, true,
         {"transformer/dynamic/dd/replaced.onnx",
          "transformer/dd/replaced.onnx"}, false},
        {"vae_decoder", ComponentType::VAE_DECODER, true,
         {"vae_decoder/dynamic/dd/replaced.onnx",
          "vae_decoder/dd/replaced.onnx"}, false},
        {"vae_encoder", ComponentType::VAE_ENCODER, false,
         {"vae_encoder/dynamic/dd/replaced.onnx",
          "vae_encoder/dd/replaced.onnx",
          "vae_encoder/model.onnx"}, false},
    };

    // FLUX.1-schnell: flat transformer dir, has text_encoder_3_gptq_v2, schnell in path
    d.detect = [](const std::filesystem::path& base) -> int {
        namespace fs = std::filesystem;
        if (!fs::exists(base / "transformer")) return 0;
        if (fs::exists(base / "normal" / "transformer")) return 0;  // SD3 layout
        if (fs::exists(base / "text_encoder_2")) return 0;          // SDXL/SD3 layout
        const auto lower = to_lower_str(base.string());
        if (lower.find("flux") == std::string::npos) return 0;
        // FLUX.1-schnell: has GPTQ T5 dir OR "schnell" in path
        bool has_t5_gptq = fs::exists(base / "text_encoder_3_gptq_v2");
        bool is_schnell   = lower.find("schnell") != std::string::npos;
        if (has_t5_gptq || is_schnell) return 15;
        return 0;
    };

    d.create_encoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           CLIPTokenizer& tok, CLIPTokenizer&, const SDConfig&)
        -> std::unique_ptr<ITextEncoder> {
        return std::make_unique<FluxTextEncoder>(components, tok, /*is_schnell=*/true);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner*, const SDConfig&)
        -> std::unique_ptr<IDenoiser> {
        // seq_len=77, embed_dim=768: CLIP-L dimensions. Update if the ONNX
        // transformer declares a different encoder_hidden_states shape.
        return std::make_unique<GenericDenoiser>(
            components, DenoiserSpec{ComponentType::TRANSFORMER, 77, 768, 768, 16});
    };
    d.create_vae_decoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
        -> std::unique_ptr<IVaeDecoder> {
        return std::make_unique<FluxVaeDecoder>(components);
    };

    return d;
}

// ============================================================================
// FLUX.2-klein (4B parameter model)
//   - CLIP-L text encoder (text_encoder/)
//   - No T5 encoder
//   - Transformer (transformer/dynamic/dd/replaced.onnx)
//   - VAE decoder (vae_decoder/dynamic/dd/replaced.onnx)
//   - FlowMatch Euler, 20 steps, no CFG (guidance_embeds=false)
// ============================================================================

static VariantDescriptor make_flux2_klein_descriptor() {
    VariantDescriptor d;
    d.name    = "flux2-klein";
    d.aliases = {"flux.2-klein", "flux2klein", "flux-klein"};
    d.variant = ModelVariant::FLUX2_KLEIN;

    d.default_width    = 1024;
    d.default_height   = 1024;
    d.default_steps    = 20;
    d.default_guidance = 0.0f;  // guidance_embeds=false in config
    d.latent_channels  = 16;
    d.default_scheduler = SchedulerType::FLOW_MATCH_EULER;

    d.has_common_dir = false;

    d.components = {
        {"text_encoder", ComponentType::TEXT_ENCODER, true,
         {"text_encoder/model.onnx",
          "text_encoder/text_encoder.onnx"}, false},
        {"transformer", ComponentType::TRANSFORMER, true,
         {"transformer/dynamic/dd/replaced.onnx",
          "transformer/dd/replaced.onnx"}, false},
        {"vae_decoder", ComponentType::VAE_DECODER, true,
         {"vae_decoder/dynamic/dd/replaced.onnx",
          "vae_decoder/dd/replaced.onnx"}, false},
        {"vae_encoder", ComponentType::VAE_ENCODER, false,
         {"vae_encoder/dynamic/dd/replaced.onnx",
          "vae_encoder/dd/replaced.onnx",
          "vae_encoder/model.onnx"}, false},
    };

    // FLUX.2-klein: flat transformer dir, no T5 GPTQ dir, "klein" in path
    d.detect = [](const std::filesystem::path& base) -> int {
        namespace fs = std::filesystem;
        if (!fs::exists(base / "transformer")) return 0;
        if (fs::exists(base / "normal" / "transformer")) return 0;  // SD3 layout
        if (fs::exists(base / "text_encoder_2")) return 0;          // SDXL/SD3 layout
        if (fs::exists(base / "text_encoder_3_gptq_v2")) return 0;  // FLUX.1 layout
        const auto lower = to_lower_str(base.string());
        if (lower.find("flux") == std::string::npos) return 0;
        if (lower.find("klein") == std::string::npos) return 0;
        return 15;
    };

    d.create_encoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           CLIPTokenizer& tok, CLIPTokenizer&, const SDConfig&)
        -> std::unique_ptr<ITextEncoder> {
        return std::make_unique<FluxTextEncoder>(components, tok, /*is_schnell=*/true);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner*, const SDConfig&)
        -> std::unique_ptr<IDenoiser> {
        // FLUX.2-klein uses joint_attention_dim=7680 in its transformer config;
        // this may require a custom embedding shape. Using CLIP-L dims (77, 768)
        // as a starting point — update if the ONNX declares different input shapes.
        return std::make_unique<GenericDenoiser>(
            components, DenoiserSpec{ComponentType::TRANSFORMER, 77, 768, 0, 16});
    };
    d.create_vae_decoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
        -> std::unique_ptr<IVaeDecoder> {
        return std::make_unique<FluxVaeDecoder>(components);
    };

    return d;
}

REGISTER_VARIANT(make_flux1_schnell_descriptor)
REGISTER_VARIANT(make_flux2_klein_descriptor)

} // namespace sd_npu
