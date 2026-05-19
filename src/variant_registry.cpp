// variant_registry.cpp - VariantRegistry implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "variant_registry.h"
#include "i_text_encoder.h"
#include "i_vae_decoder.h"
#include "i_denoiser.h"
#include "scheduler.h"
#include "onnx_model.h"
#include "clip_tokenizer.h"
#include "controlnet_runner.h"
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

namespace sd_npu {

// ============================================================================
// Singleton
// ============================================================================

VariantRegistry& VariantRegistry::instance() {
    static VariantRegistry reg;
    return reg;
}

// ============================================================================
// Registration
// ============================================================================

void VariantRegistry::register_variant(VariantDescriptor desc) {
    std::cout << "[REGISTRY] Registered variant: " << desc.name << std::endl;
    descriptors_.push_back(std::move(desc));
}

// ============================================================================
// Detection — pick the descriptor with the highest score
// ============================================================================

const VariantDescriptor* VariantRegistry::detect(const fs::path& model_dir) const {
    const VariantDescriptor* best = nullptr;
    int best_score = 0;

    for (const auto& d : descriptors_) {
        if (!d.detect) continue;
        int score = d.detect(model_dir);
        if (score > best_score) {
            best_score = score;
            best = &d;
        }
    }

    if (best) {
        std::cout << "[REGISTRY] Auto-detected variant: " << best->name
                  << " (score=" << best_score << ")" << std::endl;
    }
    return best;
}

// ============================================================================
// Lookup by name / alias
// ============================================================================

const VariantDescriptor* VariantRegistry::find(const std::string& name) const {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    for (const auto& d : descriptors_) {
        // Check primary name
        std::string dname = d.name;
        std::transform(dname.begin(), dname.end(), dname.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (dname == lower) return &d;

        // Check aliases
        for (const auto& alias : d.aliases) {
            std::string a = alias;
            std::transform(a.begin(), a.end(), a.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (a == lower) return &d;
        }
    }
    return nullptr;
}

// ============================================================================
// Lookup by enum
// ============================================================================

const VariantDescriptor* VariantRegistry::find(ModelVariant variant) const {
    for (const auto& d : descriptors_) {
        if (d.variant == variant) return &d;
    }
    return nullptr;
}

// ============================================================================
// Component discovery — data-driven from descriptor
// ============================================================================

std::string VariantRegistry::find_first(const std::string& base,
                                         const std::vector<std::string>& candidates) {
    for (const auto& c : candidates) {
        auto p = (fs::path(base) / c).string();
        if (fs::exists(p)) return p;
    }
    return "";
}

void VariantRegistry::discover_components(const fs::path& model_dir,
                                           const VariantDescriptor& desc,
                                           SDConfig& config) const {
    std::string base = model_dir.string();

    // Resolve common/ and sub-model directories
    std::string common_path = base;
    std::string sub_path = base;
    if (desc.has_common_dir) {
        auto cp = (model_dir / "common").string();
        if (fs::exists(cp)) common_path = cp;
        if (!desc.sub_model_dir.empty()) {
            auto sp = (model_dir / desc.sub_model_dir).string();
            if (fs::exists(sp)) sub_path = sp;
        }
    }

    for (const auto& spec : desc.components) {
        std::string found;

        // Try sub-model path first (for transformer, controlnet, etc.)
        if (sub_path != base) {
            found = find_first(sub_path, spec.search_paths);
        }

        // Try common path (for shared components like VAE, text encoders)
        if (found.empty() && spec.search_common && common_path != base) {
            found = find_first(common_path, spec.search_paths);
        }

        // Try base path as fallback
        if (found.empty()) {
            found = find_first(base, spec.search_paths);
        }

        // For SD3-style layouts, also try with common/ and normal/ prefixes
        if (found.empty() && desc.has_common_dir) {
            // Try prefixed paths under base
            std::vector<std::string> prefixed;
            for (const auto& p : spec.search_paths) {
                if (spec.search_common) {
                    prefixed.push_back("common/" + p);
                }
                if (!desc.sub_model_dir.empty()) {
                    prefixed.push_back(desc.sub_model_dir + "/" + p);
                }
            }
            found = find_first(base, prefixed);
        }

        if (!found.empty()) {
            config.component_paths[spec.key] = found;
        } else if (spec.required) {
            std::cout << "[REGISTRY] WARNING: Required component '" << spec.key
                      << "' not found" << std::endl;
        }
    }

    // ControlNet discovery (dynamic, based on --controlnet flag)
    if (!config.controlnet_type.empty()) {
        std::cout << "[REGISTRY] ControlNet requested: type='" << config.controlnet_type << "'" << std::endl;
        
        std::string cn_dir = config.controlnet_type;
        std::transform(cn_dir.begin(), cn_dir.end(), cn_dir.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        cn_dir = "controlnet-" + cn_dir;

        std::vector<std::string> cn_paths = {
            cn_dir + "/dynamic/dd/replaced.onnx",
            cn_dir + "/dd/replaced.onnx"
        };

        std::string cn_found;
        if (sub_path != base) {
            cn_found = find_first(sub_path, cn_paths);
            if (!cn_found.empty()) {
                std::cout << "[REGISTRY] Found ControlNet in sub-path: " << cn_found << std::endl;
            }
        }
        if (cn_found.empty()) {
            cn_found = find_first(base, cn_paths);
            if (!cn_found.empty()) {
                std::cout << "[REGISTRY] Found ControlNet in base: " << cn_found << std::endl;
            }
        }
        if (cn_found.empty() && desc.has_common_dir && !desc.sub_model_dir.empty()) {
            std::vector<std::string> prefixed;
            for (const auto& p : cn_paths) {
                prefixed.push_back(desc.sub_model_dir + "/" + p);
            }
            cn_found = find_first(base, prefixed);
            if (!cn_found.empty()) {
                std::cout << "[REGISTRY] Found ControlNet in common dir: " << cn_found << std::endl;
            }
        }

        if (!cn_found.empty()) {
            config.component_paths["controlnet"] = cn_found;
            
            // ControlNet requires VAE encoder to process control images into latent space
            // Force VAE encoder discovery if not already found
            if (config.component_paths.find("vae_encoder") == config.component_paths.end()) {
                std::cout << "[REGISTRY] ControlNet enabled: forcing VAE encoder discovery" << std::endl;
                for (const auto& spec : desc.components) {
                    if (spec.key == "vae_encoder") {
                        std::string vae_enc_found;
                        if (sub_path != base) {
                            vae_enc_found = find_first(sub_path, spec.search_paths);
                        }
                        if (vae_enc_found.empty()) {
                            vae_enc_found = find_first(base, spec.search_paths);
                        }
                        if (vae_enc_found.empty() && desc.has_common_dir && !desc.sub_model_dir.empty()) {
                            std::vector<std::string> prefixed;
                            for (const auto& p : spec.search_paths) {
                                prefixed.push_back(desc.sub_model_dir + "/" + p);
                            }
                            vae_enc_found = find_first(base, prefixed);
                        }
                        if (!vae_enc_found.empty()) {
                            std::cout << "[REGISTRY] Found VAE encoder for ControlNet: " << vae_enc_found << std::endl;
                            config.component_paths[spec.key] = vae_enc_found;
                        } else {
                            std::cerr << "[REGISTRY] ERROR: VAE encoder required for ControlNet but not found" << std::endl;
                            throw std::runtime_error("VAE encoder not found (required for ControlNet '" + config.controlnet_type + "')");
                        }
                        break;
                    }
                }
            }
        } else {
            std::cerr << "[REGISTRY] ERROR: ControlNet type '" << config.controlnet_type 
                      << "' requested but not found in model directory" << std::endl;
            std::cerr << "[REGISTRY] Searched for: " << cn_dir << "/dynamic/dd/replaced.onnx" << std::endl;
            std::cerr << "[REGISTRY]            or: " << cn_dir << "/dd/replaced.onnx" << std::endl;
            throw std::runtime_error("ControlNet '" + config.controlnet_type + "' not found in model directory");
        }
    }

    std::cout << "[REGISTRY] Discovered " << config.component_paths.size()
              << " components for variant " << desc.name << std::endl;
}

// ============================================================================
// Apply defaults
// ============================================================================

void VariantRegistry::apply_defaults(const VariantDescriptor& desc,
                                      SDConfig& config) const {
    config.variant = desc.variant;
    config.width = desc.default_width;
    config.height = desc.default_height;
    config.num_inference_steps = desc.default_steps;
    config.guidance_scale = desc.default_guidance;
    config.scheduler_type = desc.default_scheduler;
}

// ============================================================================
// Create components — delegates to descriptor factory functions
// ============================================================================

VariantComponents VariantRegistry::create_components(
    const VariantDescriptor& desc,
    std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
    CLIPTokenizer& tokenizer,
    CLIPTokenizer& tokenizer_2,
    const SDConfig& config,
    ControlNetRunner* controlnet_runner) const {

    VariantComponents result;

    std::cout << "Creating variant components for: " << desc.name << std::endl;

    // Text encoder
    if (desc.create_encoder) {
        result.text_encoder = desc.create_encoder(components, tokenizer, tokenizer_2, config);
        auto required = result.text_encoder->required_components();
        result.required_models.insert(result.required_models.end(),
                                      required.begin(), required.end());
    }

    // VAE decoder
    if (desc.create_vae_decoder) {
        result.vae_decoder = desc.create_vae_decoder(components);
        auto required = result.vae_decoder->required_components();
        result.required_models.insert(result.required_models.end(),
                                      required.begin(), required.end());
    }

    // Scheduler — use config if already set by scheduler_config.json, else use descriptor default
    SchedulerType sched_type = config.scheduler_type;
    if (sched_type == SchedulerType::PNDM) {
        // PNDM is the SDConfig default; if not overridden, use variant's preferred scheduler
        sched_type = desc.default_scheduler;
    }
    result.scheduler = create_scheduler(sched_type, config.timestep_spacing);

    // Denoiser
    if (desc.create_denoiser) {
        result.denoiser = desc.create_denoiser(components, controlnet_runner, config);
        // The denoiser factory should push its own required models, but we also
        // infer the main model type from latent channels as a safety net
    }

    if (result.vae_decoder) {
        std::cout << "  VAE decoder: scaling_factor=" << result.vae_decoder->latent_scaling_factor() << std::endl;
    }
    if (result.denoiser) {
        std::cout << "  Denoiser: " << result.denoiser->latent_channels() << " latent channels" << std::endl;
    }

    return result;
}

} // namespace sd_npu
