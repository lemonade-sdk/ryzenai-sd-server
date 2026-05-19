// sd_types.h - Shared types, enums, and configuration for SD NPU pipeline
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <string>
#include <map>
#include <vector>

namespace sd_npu {

// ============================================================================
// Enums
// ============================================================================

// Model component types
enum class ComponentType {
    TEXT_ENCODER,
    TEXT_ENCODER_2,
    TEXT_ENCODER_3,
    VAE_ENCODER,
    VAE_DECODER,
    UNET,           // SD1.5, SDXL
    TRANSFORMER,    // SD3
    CONTROLNET
};

// Model variant
enum class ModelVariant {
    SD15,
    SD_TURBO,
    SDXL,
    SDXL_TURBO,
    SD3,
    SD35
};

// ControlNet type
enum class ControlNetType {
    NONE,
    CANNY,
    POSE,
    TILE,
    DEPTH,
    UNION,
    OUTPAINTING,
    INPAINTING,
    REMOVAL
};

inline ControlNetType controlnet_from_string(const std::string& s) {
    if (s == "canny" || s == "Canny")           return ControlNetType::CANNY;
    if (s == "pose" || s == "Pose")             return ControlNetType::POSE;
    if (s == "tile" || s == "Tile")             return ControlNetType::TILE;
    if (s == "depth" || s == "Depth")           return ControlNetType::DEPTH;
    if (s == "union" || s == "Union")           return ControlNetType::UNION;
    if (s == "outpainting" || s == "OutPainting") return ControlNetType::OUTPAINTING;
    if (s == "inpainting" || s == "InPainting") return ControlNetType::INPAINTING;
    if (s == "removal" || s == "Removal")       return ControlNetType::REMOVAL;
    return ControlNetType::NONE;
}

inline const char* controlnet_to_string(ControlNetType t) {
    switch (t) {
        case ControlNetType::CANNY:       return "Canny";
        case ControlNetType::POSE:        return "Pose";
        case ControlNetType::TILE:        return "Tile";
        case ControlNetType::DEPTH:       return "Depth";
        case ControlNetType::UNION:       return "Union";
        case ControlNetType::OUTPAINTING: return "OutPainting";
        case ControlNetType::INPAINTING:  return "InPainting";
        case ControlNetType::REMOVAL:     return "Removal";
        case ControlNetType::NONE:        return "None";
    }
    return "None";
}

// Scheduler type
enum class SchedulerType {
    PNDM,
    EULER_DISCRETE,
    FLOW_MATCH_EULER
};

// ============================================================================
// Variant utilities
// ============================================================================

// Parse variant string -> enum
inline ModelVariant variant_from_string(const std::string& s) {
    if (s == "sd15" || s == "sd1.5" || s == "1.5")  return ModelVariant::SD15;
    if (s == "sd-turbo" || s == "turbo")            return ModelVariant::SD_TURBO;
    if (s == "sdxl-turbo")                          return ModelVariant::SDXL_TURBO;
    if (s == "sdxl" || s == "sdxl-base" || s == "segmind-vega")
                                                    return ModelVariant::SDXL;
    if (s == "sd3")                                 return ModelVariant::SD3;
    if (s == "sd3.5" || s == "sd35")                return ModelVariant::SD35;
    return ModelVariant::SD15; // default to SD1.5
}

// Check if variant is SD3-family (SD3 or SD3.5 share the same pipeline)
inline bool is_sd3_family(ModelVariant v) {
    return v == ModelVariant::SD3 || v == ModelVariant::SD35;
}

inline const char* variant_to_string(ModelVariant v) {
    switch (v) {
        case ModelVariant::SD15:        return "sd15";
        case ModelVariant::SD_TURBO:    return "sd-turbo";
        case ModelVariant::SDXL:        return "sdxl";
        case ModelVariant::SDXL_TURBO:  return "sdxl-turbo";
        case ModelVariant::SD3:         return "sd3";
        case ModelVariant::SD35:        return "sd3.5";
    }
    return "unknown";
}

// ============================================================================
// Text Encoder Output
// ============================================================================

// Result of text encoding: embeddings + optional pooled embeddings
struct TextEncoderOutput {
    std::vector<float> embeddings;          // [batch, seq_len, embed_dim]
    std::vector<float> pooled_embeddings;   // [batch, pool_dim] (empty for SD1.5)
};

// ============================================================================
// Configuration
// ============================================================================

struct SDConfig {
    ModelVariant variant = ModelVariant::SD3;
    SchedulerType scheduler_type = SchedulerType::FLOW_MATCH_EULER;
    std::string timestep_spacing = "leading";  // "leading" or "trailing" spacing for schedulers
    int height = 1024;
    int width = 1024;
    int num_inference_steps = 28;
    float guidance_scale = 7.0f;
    int seed = 0;
    int num_images_per_prompt = 1;

    // Paths (using lemonade's multi-component pattern)
    std::map<std::string, std::string> component_paths;

    // Model identity (for auto-download from Hugging Face)
    std::string model_id;             // e.g., "amd/stable-diffusion-1.5-amdnpu"
    std::string revision;             // Git branch/tag/commit for HF download

    // Custom ops
    std::string custom_op_path;
    std::string dd_cache_path;

    // ControlNet (optional)
    std::string controlnet_type;      // empty = none; "Canny", "Pose", "Tile", etc.
    float controlnet_scale = 0.5f;
    std::string control_image_path;
    std::string control_mask_path;    // For outpainting/inpainting/removal
    std::vector<int> image_pads = {0, 0, 0, 0};  // [top, bottom, left, right] for outpainting

    // Image-to-image
    float strength = 0.3f;            // Img2img transformation strength (0.0-1.0)
    int img2img_start_step = 0;       // Start denoising from this step
    bool is_img2img = false;          // True when running img2img (needed to skip scale_initial_latents even when start_step==0)

    // SD3-specific
    int t5_sequence_len = 83;         // T5 tokenizer sequence length (77 or 83)
    bool dynamic_shape = false;       // Enable dynamic shape flow
    std::string dynamic_shape_file;   // JSON config for multiple resolutions

    // Execution provider
    bool force_cpu = false;           // Force CPU-only execution (disables DML/NPU)
    bool gpu = false;                 // Use DML/GPU instead of NPU
    bool enable_compile = false;      // Enable compile fusion runtime

    // Profiling
    bool enable_profile = false;      // Enable profiling measurement
    int profiling_rounds = 4;         // Number of profiling iterations

    // Output
    std::string output_path = "output.png"; // PNG, BMP, JPG, or raw RGB
    std::string output_dir = "generated_images"; // Output directory
    bool no_images = false;           // Skip saving images (profiling only)

    // Prompt input
    std::string prompt_file;          // Path to JSON file with list of prompts

    // Debug / logging
    bool verbose = false;
};

// ============================================================================
// HTTP API structures
// ============================================================================

struct ImageRequest {
    std::string prompt;
    std::string negative_prompt;
    int width = 512;
    int height = 512;
    int num_inference_steps = 20;
    float guidance_scale = 7.5f;
    int64_t seed = -1;
    int num_images_per_prompt = 1;
    std::string controlnet_type;           // For ControlNet
    std::string control_image_b64;         // Base64 encoded control image
    std::string control_mask_b64;          // Base64 encoded mask (outpainting/inpainting)
    float controlnet_conditioning_scale = 0.5f;
    float strength = 0.3f;                // Img2img strength
    std::vector<int> image_pads;           // [top,bottom,left,right] for outpainting
};

// Image generation response (OpenAI-compatible format)
struct ImageData {
    std::string b64_json;                  // Base64-encoded raw RGB data (H×W×3 uint8, row-major)
    std::string url;                       // URL to image (if serving files)
    std::string revised_prompt;            // Revised prompt (if applicable)
};

struct ImageResponse {
    // OpenAI-compatible fields
    int64_t created;                       // Unix timestamp
    std::vector<ImageData> data;           // Generated images
    
    // Extended metadata
    int width;
    int height;
    std::string format;                    // "png", "jpeg", "raw"
    double generation_time_ms;
    double text_encoding_time_ms;
    double denoising_time_ms;
    double vae_decoding_time_ms;
    int num_inference_steps;
    
    // Legacy fields for compatibility
    std::vector<uint8_t> image_data;       // Raw image bytes (for direct access)
    bool success = true;
    std::string error;                     // Error message if !success
    float inference_time_ms = 0.0f;        // Total inference time
};

} // namespace sd_npu
