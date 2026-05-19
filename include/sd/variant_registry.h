// variant_registry.h - Self-registering variant descriptor system
// Copyright (C) 2025 Advanced Micro Devices, Inc.
//
// Each model variant (SD1.5, SDXL, SD3, etc.) registers a VariantDescriptor
// that encapsulates all variant-specific knowledge: defaults, component
// discovery paths, auto-detection heuristics, and factory functions.
//
// Adding a new variant requires ZERO modifications to existing files:
//   1. Create a new header + source in src/variants/
//   2. Define the descriptor and call REGISTER_VARIANT()
//   3. Add the source file to CMakeLists.txt

#pragma once

#include "sd_types.h"
#include "variant_factory.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <filesystem>

namespace sd_npu {

// Forward declarations for types not needed in the header interface
class ControlNetRunner;

// ============================================================================
// ComponentSpec — describes one ONNX component to discover on disk
// ============================================================================

struct ComponentSpec {
    std::string key;                        // config key: "transformer", "unet", etc.
    ComponentType type;                     // enum value
    bool required = true;                   // false = optional (e.g. T5, VAE encoder)
    std::vector<std::string> search_paths;  // relative paths to try, in order
    bool search_common = false;             // also search under common/ (SD3 layout)
};

// ============================================================================
// VariantDescriptor — single source of truth for a model variant
// ============================================================================

struct VariantDescriptor {
    // ── Identity ──────────────────────────────────────────────
    std::string name;                       // primary name, e.g. "sd3.5"
    std::vector<std::string> aliases;       // alternative names for CLI parsing
    ModelVariant variant;

    // ── Defaults ──────────────────────────────────────────────
    int default_width  = 512;
    int default_height = 512;
    int default_steps  = 20;
    float default_guidance = 7.5f;
    int latent_channels = 4;
    SchedulerType default_scheduler = SchedulerType::EULER_DISCRETE;

    // ── Directory layout ──────────────────────────────────────
    bool has_common_dir = false;            // SD3-style common/ + normal/ split
    std::string sub_model_dir;              // "normal" for SD3, empty for flat layouts

    // ── Component discovery ───────────────────────────────────
    std::vector<ComponentSpec> components;

    // ── Auto-detection ────────────────────────────────────────
    // Returns a priority score (0 = no match, higher = better match).
    // The registry picks the descriptor with the highest score.
    using DetectFn = std::function<int(const std::filesystem::path&)>;
    DetectFn detect;

    // ── Factory functions ─────────────────────────────────────
    using EncoderFactory = std::function<std::unique_ptr<ITextEncoder>(
        std::map<ComponentType, std::unique_ptr<OnnxModel>>&,
        CLIPTokenizer&, CLIPTokenizer&, const SDConfig&)>;

    using DenoiserFactory = std::function<std::unique_ptr<IDenoiser>(
        std::map<ComponentType, std::unique_ptr<OnnxModel>>&,
        ControlNetRunner*, const SDConfig&)>;

    using VaeFactory = std::function<std::unique_ptr<IVaeDecoder>(
        std::map<ComponentType, std::unique_ptr<OnnxModel>>&)>;

    EncoderFactory  create_encoder;
    DenoiserFactory create_denoiser;
    VaeFactory      create_vae_decoder;
};

// ============================================================================
// VariantRegistry — global collection with auto-discovery
// ============================================================================

class VariantRegistry {
public:
    /// Singleton access.
    static VariantRegistry& instance();

    /// Register a variant descriptor (called from static initialisers).
    void register_variant(VariantDescriptor desc);

    /// Detect the best-matching variant for a model directory.
    /// Returns nullptr if nothing matches.
    const VariantDescriptor* detect(const std::filesystem::path& model_dir) const;

    /// Look up a variant by name or alias.
    /// Returns nullptr if not found.
    const VariantDescriptor* find(const std::string& name) const;

    /// Look up a variant by its enum value.
    /// Returns nullptr if not found.
    const VariantDescriptor* find(ModelVariant variant) const;

    /// Discover ONNX component paths for the given model directory and
    /// populate config.component_paths. Also handles ControlNet discovery.
    void discover_components(const std::filesystem::path& model_dir,
                             const VariantDescriptor& desc,
                             SDConfig& config) const;

    /// Apply variant defaults to config (width, height, steps, guidance, scheduler).
    /// Only sets values that are at their "unset" sentinel (e.g. -1).
    void apply_defaults(const VariantDescriptor& desc, SDConfig& config) const;

    /// Create variant-specific encoder, denoiser, VAE, and scheduler.
    VariantComponents create_components(
        const VariantDescriptor& desc,
        std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
        CLIPTokenizer& tokenizer,
        CLIPTokenizer& tokenizer_2,
        const SDConfig& config,
        ControlNetRunner* controlnet_runner = nullptr) const;

    /// All registered descriptors.
    const std::vector<VariantDescriptor>& all() const { return descriptors_; }

private:
    VariantRegistry() = default;
    std::vector<VariantDescriptor> descriptors_;

    /// Find the first existing file from a list of candidates.
    static std::string find_first(const std::string& base,
                                  const std::vector<std::string>& candidates);
};

// ============================================================================
// Auto-registration macro
// ============================================================================

// Usage (at file scope in a variant .cpp):
//   REGISTER_VARIANT(make_my_descriptor)
//
// where make_my_descriptor is a function returning VariantDescriptor.
// The macro creates a file-scope static bool whose initialiser calls
// register_variant(), ensuring registration happens before main().

#define REGISTER_VARIANT(desc_func)                                              \
    namespace {                                                                  \
    static const bool _registered_##desc_func = [] {                             \
        sd_npu::VariantRegistry::instance().register_variant(desc_func());       \
        return true;                                                             \
    }();                                                                         \
    }

} // namespace sd_npu
