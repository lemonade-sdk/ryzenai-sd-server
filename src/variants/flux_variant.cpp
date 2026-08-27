// flux_variant.cpp - FLUX.1-schnell and FLUX.2-klein variant registration
// Copyright (C) 2025 Advanced Micro Devices, Inc.
//
// FLUX models use a transformer-based denoiser (not UNet), the FlowMatch Euler
// scheduler, and a 16-channel VAE. Text conditioning uses T5-XXL for
// encoder_hidden_states and CLIP-L for pooled_projections.

#include "variant_registry.h"
#include "variants/flux_text_encoder.h"
#include "variants/flux_vae_decoder.h"
#include "variants/flux2_text_encoder.h"
#include "variants/flux2_denoiser.h"
#include "variants/flux2_vae_decoder.h"
#include "variants/generic_denoiser.h"
#include "t5_tokenizer.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

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
        {"text_encoder_2", ComponentType::TEXT_ENCODER_2, false,
         {"text_encoder_2/model.onnx"}, false},
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
        if (fs::exists(base / "unet")) return 0;                    // SDXL/SD15 layout
        const auto lower = to_lower_str(base.string());
        if (lower.find("flux") == std::string::npos) return 0;
        // FLUX.1-schnell: has GPTQ T5 dir OR "schnell" in path
        bool has_t5_gptq = fs::exists(base / "text_encoder_3_gptq_v2");
        bool is_schnell   = lower.find("schnell") != std::string::npos;
        if (has_t5_gptq || is_schnell) return 15;
        return 0;
    };

    d.create_encoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           CLIPTokenizer& tok, CLIPTokenizer&, const SDConfig& cfg)
        -> std::unique_ptr<ITextEncoder> {
        // Derive model root from the text_encoder_2 or text_encoder path
        std::unique_ptr<T5Tokenizer> t5_tok;
        // FLUX transformer RoPE (RopeCachePack) requires the text sequence length
        // to be divisible by 8; the reference schnell pipeline uses 256. The global
        // default (83, SD3-oriented) is not valid here, so snap to 256 unless the
        // user supplied an explicit multiple of 8.
        int t5_seq_len = (cfg.t5_sequence_len > 0 && cfg.t5_sequence_len % 8 == 0)
                             ? cfg.t5_sequence_len : 256;
        auto it = cfg.component_paths.find("text_encoder_2");
        if (it != cfg.component_paths.end()) {
            namespace fs = std::filesystem;
            // text_encoder_2/model.onnx -> parent->parent = model root
            fs::path spiece = fs::path(it->second).parent_path().parent_path()
                              / "tokenizer_2" / "spiece.model";
            if (fs::exists(spiece)) {
                try {
                    t5_tok = std::make_unique<T5Tokenizer>(spiece.string());
                    std::cout << "[FLUX] T5 tokenizer loaded from: " << spiece << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "[FLUX] Warning: failed to load T5 tokenizer: "
                              << e.what() << std::endl;
                }
            } else {
                std::cout << "[FLUX] tokenizer_2/spiece.model not found, T5 disabled" << std::endl;
            }
        }
        return std::make_unique<FluxTextEncoder>(
            components, tok, /*is_schnell=*/true, std::move(t5_tok), t5_seq_len);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner*, const SDConfig& cfg)
        -> std::unique_ptr<IDenoiser> {
        // encoder_hidden_states: T5 [B, t5_seq_len, 4096]
        // pooled_projections:    CLIP [B, 768]
        // Must match the encoder's padded length and be divisible by 8 (RoPE).
        int t5_seq = (cfg.t5_sequence_len > 0 && cfg.t5_sequence_len % 8 == 0)
                         ? cfg.t5_sequence_len : 256;
        return std::make_unique<GenericDenoiser>(
            components, DenoiserSpec{ComponentType::TRANSFORMER, t5_seq, 4096, 768, 16,
                                     /*disable_cfg=*/true, /*timestep_scale=*/0.001f});
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
    d.latent_channels  = 32;    // VAE has 32 channels (packed to 128 for the transformer)
    d.default_scheduler = SchedulerType::FLOW_MATCH_EULER;

    d.has_common_dir = false;

    d.components = {
        {"text_encoder", ComponentType::TEXT_ENCODER, true,
         {"text_encoder/qwen3_text_encoder_prompt_embeds_matmulnbits.onnx",
          "text_encoder/model.onnx",
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
                           CLIPTokenizer&, CLIPTokenizer&, const SDConfig& cfg)
        -> std::unique_ptr<ITextEncoder> {
        // Derive model root from the text_encoder component path
        // (<root>/text_encoder/<file>.onnx -> parent->parent = <root>).
        std::string model_root;
        auto it = cfg.component_paths.find("text_encoder");
        if (it != cfg.component_paths.end()) {
            model_root = fs::path(it->second).parent_path().parent_path().string();
        }
        return std::make_unique<Flux2TextEncoder>(components, model_root, /*max_seq=*/256);
    };
    d.create_denoiser = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
                           ControlNetRunner*, const SDConfig& cfg)
        -> std::unique_ptr<IDenoiser> {
        // Model root for BatchNorm denorm stats (bn.running_x.safetensors):
        // any <root>/<comp>/... path -> parent->parent = <root> for the flat
        // (non-DD) layout, but transformer/vae use nested dd dirs, so prefer the
        // text_encoder path which is <root>/text_encoder/<file>.onnx.
        std::string model_root;
        auto it = cfg.component_paths.find("text_encoder");
        if (it != cfg.component_paths.end()) {
            model_root = fs::path(it->second).parent_path().parent_path().string();
        }
        return std::make_unique<Flux2Denoiser>(
            components, model_root, /*text_seq_len=*/256, /*text_embed_dim=*/7680);
    };
    d.create_vae_decoder = [](std::map<ComponentType, std::unique_ptr<OnnxModel>>& components)
        -> std::unique_ptr<IVaeDecoder> {
        return std::make_unique<Flux2VaeDecoder>(components);
    };

    return d;
}

REGISTER_VARIANT(make_flux1_schnell_descriptor)
REGISTER_VARIANT(make_flux2_klein_descriptor)

} // namespace sd_npu
