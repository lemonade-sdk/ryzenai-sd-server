// sd_pipeline.cpp - Unified SD Pipeline implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "sd_pipeline.h"
#include "variant_registry.h"
#include "controlnet_runner.h"
#include <iostream>
#include <filesystem>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cmath>
#include <deque>
#include <vector>

namespace fs = std::filesystem;

// Diagnostic helpers: print min/max/mean/nonzero stats for a tensor.
// Guarded behind SD_ENABLE_DIAG — define it at compile time to enable.
// In release builds these compile to zero-cost no-ops (no scan, no output).
#ifdef SD_ENABLE_DIAG
static void dump_stats(const char* label, const std::vector<float>& v) {
    if (v.empty()) { std::cout << "[DIAG] " << label << ": EMPTY" << std::endl; return; }
    float mn = v[0], mx = v[0];
    double sum = 0;
    size_t nz = 0;
    for (auto x : v) { if (x < mn) mn = x; if (x > mx) mx = x; sum += x; if (x != 0.0f) nz++; }
    std::cout << "[DIAG] " << label << ": size=" << v.size()
              << " min=" << mn << " max=" << mx
              << " mean=" << (sum / v.size())
              << " nonzero=" << nz << "/" << v.size()
              << " first5=[" << v[0];
    for (size_t i = 1; i < std::min((size_t)5, v.size()); i++) std::cout << "," << v[i];
    std::cout << "]" << std::endl;
}
static void dump_stats_u8(const char* label, const std::vector<uint8_t>& v) {
    if (v.empty()) { std::cout << "[DIAG] " << label << ": EMPTY" << std::endl; return; }
    uint8_t mn = v[0], mx = v[0];
    double sum = 0;
    for (auto x : v) { if (x < mn) mn = x; if (x > mx) mx = x; sum += x; }
    std::cout << "[DIAG] " << label << ": size=" << v.size()
              << " min=" << (int)mn << " max=" << (int)mx
              << " mean=" << (sum / v.size()) << std::endl;
}
#else
static void dump_stats(const char*, const std::vector<float>&) {}
static void dump_stats_u8(const char*, const std::vector<uint8_t>&) {}
#endif

namespace sd_npu {

// ============================================================================
// Utility functions for base64 encoding and PNG encoding
// ============================================================================

namespace {
    static const char base64_chars[] = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string base64_encode(const uint8_t* data, size_t len) {
        const size_t out_len = ((len + 2) / 3) * 4;
        std::string ret(out_len, '\0');
        size_t out_pos = 0;

        size_t i = 0;
        while (i + 3 <= len) {
            uint32_t triple = (static_cast<uint32_t>(data[i])     << 16)
                            | (static_cast<uint32_t>(data[i + 1]) <<  8)
                            |  static_cast<uint32_t>(data[i + 2]);
            ret[out_pos++] = base64_chars[(triple >> 18) & 0x3F];
            ret[out_pos++] = base64_chars[(triple >> 12) & 0x3F];
            ret[out_pos++] = base64_chars[(triple >>  6) & 0x3F];
            ret[out_pos++] = base64_chars[ triple        & 0x3F];
            i += 3;
        }

        const size_t rem = len - i;
        if (rem == 1) {
            uint32_t triple = static_cast<uint32_t>(data[i]) << 16;
            ret[out_pos++] = base64_chars[(triple >> 18) & 0x3F];
            ret[out_pos++] = base64_chars[(triple >> 12) & 0x3F];
            ret[out_pos++] = '=';
            ret[out_pos++] = '=';
        } else if (rem == 2) {
            uint32_t triple = (static_cast<uint32_t>(data[i])     << 16)
                            | (static_cast<uint32_t>(data[i + 1]) <<  8);
            ret[out_pos++] = base64_chars[(triple >> 18) & 0x3F];
            ret[out_pos++] = base64_chars[(triple >> 12) & 0x3F];
            ret[out_pos++] = base64_chars[(triple >>  6) & 0x3F];
            ret[out_pos++] = '=';
        }

        return ret;
    }

    // CRC-32 table for PNG chunk checksums
    static uint32_t crc32_table[256];
    static bool crc32_table_init = false;

    void init_crc32_table() {
        if (crc32_table_init) return;
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int j = 0; j < 8; j++)
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            crc32_table[i] = c;
        }
        crc32_table_init = true;
    }

    uint32_t crc32(const uint8_t* data, size_t len) {
        init_crc32_table();
        uint32_t c = 0xFFFFFFFFu;
        for (size_t i = 0; i < len; i++)
            c = crc32_table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    }

    uint32_t adler32(const uint8_t* data, size_t len) {
        uint32_t a = 1, b = 0;
        for (size_t i = 0; i < len; i++) {
            a = (a + data[i]) % 65521;
            b = (b + a) % 65521;
        }
        return (b << 16) | a;
    }

    // Write big-endian uint32
    void write_be32(std::vector<uint8_t>& out, uint32_t v) {
        out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
        out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
        out.push_back(static_cast<uint8_t>((v >>  8) & 0xFF));
        out.push_back(static_cast<uint8_t>( v        & 0xFF));
    }

    void write_le16(std::vector<uint8_t>& out, uint16_t v) {
        out.push_back(static_cast<uint8_t>(v & 0xFF));
        out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    }

    // Append a PNG chunk: length + type + data + CRC
    void write_png_chunk(std::vector<uint8_t>& out, const char type[4],
                         const uint8_t* data, uint32_t len) {
        write_be32(out, len);
        size_t crc_start = out.size();
        out.insert(out.end(), type, type + 4);
        if (len > 0) out.insert(out.end(), data, data + len);
        uint32_t c = crc32(&out[crc_start], 4 + len);
        write_be32(out, c);
    }

    // Encode raw RGB data as a PNG file (uncompressed deflate blocks).
    // No external zlib dependency required.
    std::vector<uint8_t> encode_png(const uint8_t* rgb, int width, int height) {
        std::vector<uint8_t> png;
        // PNG signature
        const uint8_t sig[] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
        png.insert(png.end(), sig, sig + 8);

        // IHDR: width, height, 8-bit depth, color type 2 (RGB)
        uint8_t ihdr[13];
        ihdr[0]  = static_cast<uint8_t>((width  >> 24) & 0xFF);
        ihdr[1]  = static_cast<uint8_t>((width  >> 16) & 0xFF);
        ihdr[2]  = static_cast<uint8_t>((width  >>  8) & 0xFF);
        ihdr[3]  = static_cast<uint8_t>( width         & 0xFF);
        ihdr[4]  = static_cast<uint8_t>((height >> 24) & 0xFF);
        ihdr[5]  = static_cast<uint8_t>((height >> 16) & 0xFF);
        ihdr[6]  = static_cast<uint8_t>((height >>  8) & 0xFF);
        ihdr[7]  = static_cast<uint8_t>( height        & 0xFF);
        ihdr[8]  = 8;   // bit depth
        ihdr[9]  = 2;   // color type: RGB
        ihdr[10] = 0;   // compression
        ihdr[11] = 0;   // filter
        ihdr[12] = 0;   // interlace
        write_png_chunk(png, "IHDR", ihdr, 13);

        // Build filtered scanlines (filter byte 0 = None per row)
        size_t row_bytes = static_cast<size_t>(width) * 3;
        size_t raw_size = static_cast<size_t>(height) * (1 + row_bytes);
        std::vector<uint8_t> raw_data;
        raw_data.reserve(raw_size);
        for (int y = 0; y < height; y++) {
            raw_data.push_back(0); // filter: None
            raw_data.insert(raw_data.end(),
                            rgb + y * row_bytes,
                            rgb + y * row_bytes + row_bytes);
        }

        // Build zlib stream with uncompressed (stored) deflate blocks
        // zlib header: CM=8 (deflate), CINFO=7 (32K window), FCHECK
        std::vector<uint8_t> zlib_data;
        uint8_t cmf = 0x78;  // CM=8, CINFO=7
        uint8_t flg = 0x01;  // FCHECK=1 (makes (cmf*256+flg) % 31 == 0)
        zlib_data.push_back(cmf);
        zlib_data.push_back(flg);

        // Split raw data into stored deflate blocks (max 65535 bytes each)
        const size_t max_block = 65535;
        size_t offset = 0;
        while (offset < raw_data.size()) {
            size_t remaining = raw_data.size() - offset;
            size_t block_size = std::min(remaining, max_block);
            bool is_last = (offset + block_size >= raw_data.size());

            zlib_data.push_back(is_last ? 0x01 : 0x00); // BFINAL | BTYPE=00 (stored)
            uint16_t len = static_cast<uint16_t>(block_size);
            uint16_t nlen = static_cast<uint16_t>(~len);
            write_le16(zlib_data, len);
            write_le16(zlib_data, nlen);
            zlib_data.insert(zlib_data.end(),
                             raw_data.data() + offset,
                             raw_data.data() + offset + block_size);
            offset += block_size;
        }

        // Adler-32 checksum of uncompressed data
        uint32_t adler = adler32(raw_data.data(), raw_data.size());
        write_be32(zlib_data, adler);

        write_png_chunk(png, "IDAT", zlib_data.data(),
                        static_cast<uint32_t>(zlib_data.size()));

        // IEND
        write_png_chunk(png, "IEND", nullptr, 0);

        return png;
    }

}

// ============================================================================
// Construction / Destruction
// ============================================================================

SDPipeline::SDPipeline(const SDConfig& config)
    : config_(config),
      // WARNING-level ORT logging is very noisy on these DD-compiled models
      // (shape-merge fallback notices, EP-assignment info, deprecated session
      // option notices) even on fully successful runs -- none of it indicates
      // a real problem, so default to ERROR to keep normal-operation logs clean.
      env_(ORT_LOGGING_LEVEL_ERROR, "SDPipeline") {

    std::cout << "Initializing SD Pipeline..." << std::endl;
    std::cout << "  Variant: " << variant_to_string(config_.variant) << std::endl;
    std::cout << "  Resolution: " << config_.width << "x" << config_.height << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    
    setup_onnx_session_options();
    
    auto after_setup = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMING] Session options setup: " 
              << std::chrono::duration<double, std::milli>(after_setup - start).count() << " ms" << std::endl;
    
    load_components();
    
    auto after_load = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMING] Component loading: " 
              << std::chrono::duration<double, std::milli>(after_load - after_setup).count() << " ms" << std::endl;

    // Create variant-specific sub-modules using registry
    const auto* desc = VariantRegistry::instance().find(config_.variant);
    if (!desc) {
        throw std::runtime_error(std::string("No variant descriptor registered for: ") + 
                                 variant_to_string(config_.variant));
    }
    auto variant_components = VariantRegistry::instance().create_components(
        *desc, components_, tokenizer_, tokenizer_2_, config_,
        controlnet_runner_.get());
    
    text_encoder_ = std::move(variant_components.text_encoder);
    vae_decoder_ = std::move(variant_components.vae_decoder);
    denoiser_ = std::move(variant_components.denoiser);
    scheduler_ = std::move(variant_components.scheduler);
    
    auto after_factory = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMING] Variant components creation: " 
              << std::chrono::duration<double, std::milli>(after_factory - after_load).count() << " ms" << std::endl;

    std::cout << "Pipeline ready!" << std::endl;
}

SDPipeline::~SDPipeline() {
    components_.clear();
}

void SDPipeline::update_config(const SDConfig& new_config) {
    // Update generation parameters that don't require reloading models
    config_.num_inference_steps = new_config.num_inference_steps;
    config_.guidance_scale = new_config.guidance_scale;
    config_.seed = new_config.seed;
    config_.width = new_config.width;
    config_.height = new_config.height;
    config_.strength = new_config.strength;

    // Also update scheduler with new step count
    scheduler_->set_timesteps(config_.num_inference_steps);
}

// ============================================================================
// ONNX session options (base-level – per-model options in OnnxModel ctor)
// ============================================================================

void SDPipeline::setup_onnx_session_options() {
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);
}

// ============================================================================
// Component loading (same logic as old SDPipeline::load_components)
// ============================================================================

void SDPipeline::load_components() {
    load_onnx_models();
    load_scheduler_config();
    load_tokenizers();
}

// ── Load ONNX model files ────────────────────────────────────────────────────
void SDPipeline::load_onnx_models() {
    std::cout << "Loading model components..." << std::endl;

    // Build string→ComponentType mapping from registry descriptor
    std::map<std::string, ComponentType> key_to_type;
    const auto* desc = VariantRegistry::instance().find(config_.variant);
    if (desc) {
        for (const auto& spec : desc->components) {
            key_to_type[spec.key] = spec.type;
        }
    }

    // Resolve the ComponentType for a discovered component key.
    auto resolve_type = [&](const std::string& key, ComponentType& out) -> bool {
        auto it = key_to_type.find(key);
        if (it != key_to_type.end()) { out = it->second; return true; }
        // Fallback mapping for components not in descriptor
        if (key == "text_encoder") out = ComponentType::TEXT_ENCODER;
        else if (key == "text_encoder_2") out = ComponentType::TEXT_ENCODER_2;
        else if (key == "text_encoder_3") out = ComponentType::TEXT_ENCODER_3;
        else if (key == "unet") out = ComponentType::UNET;
        else if (key == "transformer") out = ComponentType::TRANSFORMER;
        else if (key == "vae_decoder") out = ComponentType::VAE_DECODER;
        else if (key == "vae_encoder") out = ComponentType::VAE_ENCODER;
        else return false;
        return true;
    };

    // Load priority: the RyzenAI NPU EP allocates a shared static instruction
    // buffer sized by the FIRST DynamicDispatch model loaded. The VAE decoder
    // needs a larger instruction buffer than the denoiser (unet/transformer),
    // so the VAE MUST load before the denoiser or the VAE fails with
    // "static instruction bo smaller than requested". This mirrors the GenAI-SD
    // python reference load order (vae_decoder before transformer).
    auto load_priority = [](ComponentType t) -> int {
        switch (t) {
            case ComponentType::TEXT_ENCODER:
            case ComponentType::TEXT_ENCODER_2:
            case ComponentType::TEXT_ENCODER_3: return 0;
            case ComponentType::VAE_ENCODER:    return 1;
            case ComponentType::VAE_DECODER:    return 2;
            case ComponentType::UNET:
            case ComponentType::TRANSFORMER:    return 3;
            default:                            return 2;
        }
    };

    // Collect and order components before loading.
    struct LoadEntry { std::string key; std::string path; ComponentType type; };
    std::vector<LoadEntry> load_order;
    for (const auto& [key, path] : config_.component_paths) {
        if (key == "controlnet") continue;  // handled separately below
        ComponentType type;
        if (!resolve_type(key, type)) {
            std::cout << "  Skipping unknown component: " << key << std::endl;
            continue;
        }
        load_order.push_back({key, path, type});
    }
    std::stable_sort(load_order.begin(), load_order.end(),
                     [&](const LoadEntry& a, const LoadEntry& b) {
                         return load_priority(a.type) < load_priority(b.type);
                     });

    for (const auto& entry : load_order) {
        std::cout << "  Loading " << entry.key << " from " << entry.path << std::endl;
        components_[entry.type] = std::make_unique<OnnxModel>(
            entry.path, session_options_, env_);
    }
    // ControlNet (optional)
    if (!config_.controlnet_type.empty() &&
        config_.component_paths.count("controlnet")) {
        std::cout << "Loading ControlNet: " << config_.controlnet_type << std::endl;
        components_[ComponentType::CONTROLNET] =
            std::make_unique<OnnxModel>(
                config_.component_paths.at("controlnet"),
                session_options_, env_);
        
        // Create ControlNet helper
        ControlNetType cn_type = controlnet_from_string(config_.controlnet_type);
        controlnet_runner_ = std::make_unique<ControlNetRunner>(
            components_[ComponentType::CONTROLNET].get(),
            cn_type);
        std::cout << "ControlNet runner initialized for " 
                  << controlnet_runner_->get_name() << std::endl;
        
        if (config_.component_paths.count("vae_encoder")) {
            components_[ComponentType::VAE_ENCODER] =
                std::make_unique<OnnxModel>(
                    config_.component_paths.at("vae_encoder"),
                    session_options_, env_);
        }
    }

    std::cout << components_.size() << " components loaded" << std::endl;
}

// ── Parse scheduler_config.json and set scheduler type / timestep spacing ───
void SDPipeline::load_scheduler_config() {
    std::string base_key;
    if (config_.component_paths.count("text_encoder"))     base_key = "text_encoder";
    else if (config_.component_paths.count("unet"))        base_key = "unet";
    else if (config_.component_paths.count("transformer")) base_key = "transformer";

    if (base_key.empty()) return;

    fs::path base_dir  = fs::path(config_.component_paths[base_key])
                             .parent_path().parent_path();
    fs::path sched_cfg = base_dir / "scheduler" / "scheduler_config.json";
    if (!fs::exists(sched_cfg)) return;

    std::cout << "Loading scheduler config from " << sched_cfg.string() << std::endl;
    std::ifstream f(sched_cfg);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    size_t pos = content.find("\"_class_name\"");
    if (pos != std::string::npos) {
        size_t start = content.find(":", pos);
        size_t vs = content.find("\"", start) + 1;
        size_t ve = content.find("\"", vs);
        std::string cls = content.substr(vs, ve - vs);
        std::cout << "  Scheduler class: " << cls << std::endl;

        SchedulerType det;
        if (cls.find("PNDM") != std::string::npos) {
            det = SchedulerType::PNDM;
            if (config_.variant == ModelVariant::SD15) {
                det = SchedulerType::EULER_DISCRETE;
                std::cout << "  Overriding PNDM to Euler Discrete for SD1.5" << std::endl;
            }
        } else if (cls.find("FlowMatch") != std::string::npos) {
            det = SchedulerType::FLOW_MATCH_EULER;
        } else if (cls.find("Euler") != std::string::npos) {
            det = SchedulerType::EULER_DISCRETE;
        } else {
            det = is_sd3_family(config_.variant) ? SchedulerType::FLOW_MATCH_EULER
                : (config_.variant == ModelVariant::SD15 ? SchedulerType::PNDM
                   : SchedulerType::EULER_DISCRETE);
        }
        config_.scheduler_type = det;
    }

    // Parse timestep_spacing if present (for EulerDiscrete)
    size_t spacing_pos = content.find("\"timestep_spacing\"");
    if (spacing_pos != std::string::npos) {
        size_t start = content.find(":", spacing_pos);
        size_t vs = content.find("\"", start) + 1;
        size_t ve = content.find("\"", vs);
        std::string spacing = content.substr(vs, ve - vs);
        config_.timestep_spacing = spacing;
        std::cout << "  Timestep spacing: " << spacing << std::endl;
    }
}

// ── Load CLIP tokenizers from the model directory ────────────────────────────
void SDPipeline::load_tokenizers() {
    std::string te_key;
    if (config_.component_paths.count("text_encoder"))     te_key = "text_encoder";
    else if (config_.component_paths.count("text_encoder_2")) te_key = "text_encoder_2";

    if (te_key.empty()) return;

    fs::path base_dir = fs::path(config_.component_paths[te_key])
                            .parent_path().parent_path();

    fs::path tok1 = base_dir / "tokenizer";
    if (fs::exists(tok1 / "vocab.json") && fs::exists(tok1 / "merges.txt")) {
        std::cout << "Loading tokenizer from " << tok1.string() << std::endl;
        tokenizer_.load((tok1 / "vocab.json").string(),
                        (tok1 / "merges.txt").string());
    }

    fs::path tok2 = base_dir / "tokenizer_2";
    if (fs::exists(tok2 / "vocab.json") && fs::exists(tok2 / "merges.txt")) {
        std::cout << "Loading tokenizer_2 from " << tok2.string() << std::endl;
        tokenizer_2_.load((tok2 / "vocab.json").string(),
                          (tok2 / "merges.txt").string());
    }
}

// ============================================================================
// generate() – orchestrate the full pipeline
// ============================================================================

ImageResponse SDPipeline::generate(
    const std::string& prompt,
    const std::string& negative_prompt,
    const std::vector<uint8_t>& control_image,
    const std::vector<uint8_t>& control_mask) {

    std::cout << "\nGenerating image..." << std::endl;
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Config: variant=" << variant_to_string(config_.variant)
              << " steps=" << config_.num_inference_steps
              << " cfg=" << config_.guidance_scale
              << " seed=" << config_.seed
              << " size=" << config_.width << "x" << config_.height << std::endl;
    std::cout.flush();

    ImageResponse response;
    response.success = true;

    // 0. Read model's compiled latent dimensions (DD models have fixed shapes)
    // Prefer the denoiser's declared latent channel count (authoritative per
    // variant: 16 for SD3/FLUX, 4 for SD1.5/SDXL). FLUX transformers take a
    // packed 3D input so the sample-shape probe below cannot recover this.
    int latent_ch = denoiser_ ? denoiser_->latent_channels()
                              : (is_sd3_family(config_.variant) ? 16 : 4);
    int latent_h  = config_.height / 8;
    int latent_w  = config_.width / 8;
    int output_h  = config_.height;
    int output_w  = config_.width;

    OnnxModel* unet_model = nullptr;
    if (components_.count(ComponentType::UNET))
        unet_model = components_[ComponentType::UNET].get();
    else if (components_.count(ComponentType::TRANSFORMER))
        unet_model = components_[ComponentType::TRANSFORMER].get();

    if (unet_model) {
        auto sample_shape = unet_model->get_input_shape(0);  // [batch, ch, h, w]
        if (sample_shape.size() >= 4) {
            if (sample_shape[1] > 0) latent_ch = (int)sample_shape[1];
            if (sample_shape[2] > 0) latent_h  = (int)sample_shape[2];
            if (sample_shape[3] > 0) latent_w  = (int)sample_shape[3];
            output_h = latent_h * 8;
            output_w = latent_w * 8;
        }
        std::cout << "  Model native latents: " << latent_ch << "x" << latent_h << "x" << latent_w
                  << " -> output " << output_h << "x" << output_w << std::endl;
    }

    // 1. Encode prompt
    auto gen_start = std::chrono::steady_clock::now();
    std::cout << "[PROGRESS] Encoding text prompt..." << std::endl;
    std::cout.flush();
    auto t0 = std::chrono::steady_clock::now();
    auto enc = text_encoder_->encode(prompt, negative_prompt);
    auto t1 = std::chrono::steady_clock::now();
    double text_enc_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[TIMING] Text encoding: " << text_enc_sec << " seconds" << std::endl;

    // 2. Initialise latents at model's native resolution
    std::vector<float> latents(latent_ch * latent_h * latent_w);

    // Process control image for ControlNet if provided
    std::vector<float> controlnet_cond;
    bool has_controlnet = components_.count(ComponentType::CONTROLNET) > 0;
    bool has_control_image = !control_image.empty();
    
    if (has_controlnet && has_control_image) {
        std::cout << "  Processing control image for ControlNet..." << std::endl;
        controlnet_cond = process_control_image(control_image, control_mask);
        if (controlnet_cond.empty()) {
            std::cout << "  WARNING: Failed to process control image, continuing without ControlNet" << std::endl;
        }
        // controlnet_cond stays single-batch here. CFG batching for ControlNet is
        // decided by ControlNetRunner::compute() from the denoiser's actual `batch`
        // (encoder-output-derived, not guidance_scale) -- gating duplication on
        // guidance_scale > 1.0 here used to desync from that and crash the NPU EP
        // on any ControlNet request with guidance_scale <= 1.0 (e.g. "pose", 0.0).
    }

    // For img2img: encode the input image and add noise
    // Note: img2img and ControlNet are mutually exclusive in this implementation
    // If ControlNet is active, control_image is used for ControlNet, not img2img
    bool is_img2img = !has_controlnet && has_control_image && components_.count(ComponentType::VAE_ENCODER);

    if (is_img2img) {
        std::cout << "  Using img2img mode with input image" << std::endl;

        // Encode input image through VAE encoder to get latents
        auto encoded_latents = encode_image_to_latents(control_image);

        // Forward-transform into whatever space denoise() actually expects
        // (no-op for most variants; FLUX.2-klein patchifies + BN-normalizes
        // raw VAE latents into its packed 128-channel token space here).
        encoded_latents = denoiser_->prepare_img2img_latents(
            encoded_latents, config_.height, config_.width);

        float strength = config_.strength;
        int n_steps = config_.num_inference_steps;

        // Distilled few-step ("turbo"/"lightning") models are configured with
        // a very small num_inference_steps (often 1) for fast txt2img. That
        // leaves no room to express partial img2img `strength`: start_step
        // below always clamps to 0, so the noise-blend sigma was always the
        // maximum noise level regardless of `strength` - the input image's
        // contribution was completely drowned out by noise, and img2img /
        // variations effectively degenerated into txt2img. Give img2img
        // requests enough step resolution to express `strength` meaningfully.
        // (Naive continuous sigma interpolation without also increasing the
        // step count was tried and rejected: it desyncs the noise-blend sigma
        // from the scheduler's per-step sigmas that scale_model_input()/
        // step() assume, producing numerically incorrect - garbage - output.
        // Actually bumping num_inference_steps keeps everything internally
        // consistent, at the cost of a few extra UNet calls for img2img.)
        //
        // SD_TURBO is excluded: empirically, its NPU-compiled UNet binary is
        // only numerically valid at its single native (sigma_max) timestep -
        // running it across a multi-step schedule (any other sigma) produces
        // NaN output, unlike SDXL_TURBO/other lightning variants which are
        // fine with up to ~4 steps. So SD_TURBO keeps the original 1-step
        // behavior (img2img signal is unavoidably weak for this variant).
        constexpr int kMinImg2ImgSteps = 4;
        if (n_steps < kMinImg2ImgSteps && config_.variant != ModelVariant::SD_TURBO) {
            n_steps = kMinImg2ImgSteps;
            config_.num_inference_steps = n_steps;
        }

        // start_step: skip the first (1-strength)*N steps, only denoise the last strength*N
        // steps. This integer index drives the denoising loop itself (how many UNet calls
        // actually run) and must stay clamped to a valid step in [0, n_steps-1].
        int start_step = n_steps - static_cast<int>(n_steps * strength);
        start_step = std::max(0, std::min(start_step, n_steps - 1));

        // Set up scheduler to get the sigma value at start_step. Some
        // variants (e.g. FLUX.2-klein) compute their own internal sigma
        // schedule inside denoise() and ignore this Scheduler entirely -
        // for those, ask the denoiser for the matching sigma instead so the
        // noise blend below lines up with what denoise() will integrate from.
        float sigma;
        if (denoiser_->has_custom_img2img_sigma()) {
            sigma = denoiser_->img2img_start_sigma(
                start_step, n_steps, config_.height, config_.width);
        } else {
            scheduler_->set_timesteps(n_steps);
            const auto& sigmas = scheduler_->sigmas();
            sigma = sigmas[start_step];
        }

        // Generate noise
        std::mt19937 gen(config_.seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> noise(latent_ch * latent_h * latent_w);
        for (auto& v : noise) v = dist(gen);

        // Add noise at the correct sigma level for the chosen start_step.
        // Formula differs by scheduler type:
        //   EulerDiscrete (SD1.5/SDXL): noisy = x0 + sigma * noise
        //     (sigmas are large, e.g. 14.6; scale_model_input divides by sqrt(sigma^2+1))
        //   FlowMatchEuler (SD3/SD3.5):  noisy = (1 - sigma) * x0 + sigma * noise
        //     (sigmas in [0,1]; linear interpolation between clean image and noise)
        bool is_flow_match = (config_.scheduler_type == SchedulerType::FLOW_MATCH_EULER);
        for (size_t i = 0; i < latents.size() && i < encoded_latents.size(); i++) {
            if (is_flow_match) {
                latents[i] = (1.0f - sigma) * encoded_latents[i] + sigma * noise[i];
            } else {
                latents[i] = encoded_latents[i] + noise[i] * sigma;
            }
        }

        // Tell the denoiser to start from start_step (skip steps 0..start_step-1)
        config_.img2img_start_step = start_step;
        config_.is_img2img = true;

        std::cout << "  img2img: strength=" << strength
                  << ", start_step=" << start_step << "/" << n_steps
                  << ", sigma=" << sigma << std::endl;
    } else {
        // Standard txt2img: pure random latents
        config_.img2img_start_step = 0;
        config_.is_img2img = false;
        std::mt19937 gen(config_.seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : latents) v = dist(gen);
    }

    dump_stats("initial_latents", latents);
    dump_stats("text_embeddings", enc.embeddings);

    // 3. Denoise (pass controlnet_cond if available)
    std::cout << "[PROGRESS] Denoising (" << config_.num_inference_steps << " steps)..." << std::endl;
    if (!controlnet_cond.empty()) {
        std::cout << "  Using ControlNet conditioning" << std::endl;
    }
    std::cout.flush();
    auto t2 = std::chrono::steady_clock::now();
    auto denoised = denoise(latents, enc.embeddings, enc.pooled_embeddings, controlnet_cond);
    auto t3 = std::chrono::steady_clock::now();
    double denoise_sec = std::chrono::duration<double>(t3 - t2).count();
    std::cout << "[TIMING] Denoising (" << config_.num_inference_steps << " steps): " << denoise_sec << " seconds" << std::endl;

    dump_stats("denoised_latents", denoised);

    // 4. VAE decode (use model's native resolution, not config)
    std::cout << "[PROGRESS] Decoding image with VAE (" << output_w << "x" << output_h << ")..." << std::endl;
    std::cout.flush();
    auto t4 = std::chrono::steady_clock::now();
    auto rgb_data = vae_decoder_->decode(denoised, output_h, output_w);
    auto t5 = std::chrono::steady_clock::now();
    double vae_sec = std::chrono::duration<double>(t5 - t4).count();
    double total_sec = std::chrono::duration<double>(t5 - gen_start).count();
    std::cout << "[TIMING] VAE decode: " << vae_sec << " seconds" << std::endl;

    dump_stats_u8("vae_rgb_data", rgb_data);

    std::cout << "[TIMING] Total generation: " << total_sec << " seconds "
              << "(encode=" << text_enc_sec << "s, denoise=" << denoise_sec
              << "s, vae=" << vae_sec << "s)" << std::endl;
    std::cout.flush();

    last_timing_ = {text_enc_sec, denoise_sec, vae_sec, total_sec, config_.num_inference_steps};

    // 5. Encode as PNG and base64 for OpenAI-compatible response
    auto png_data = encode_png(rgb_data.data(), output_w, output_h);
    std::string b64_json = base64_encode(png_data.data(), png_data.size());

    // 6. Populate ImageResponse with OpenAI-compatible format
    ImageData img_data;
    img_data.b64_json = b64_json;
    response.data.push_back(img_data);

    // Populate metadata
    response.created = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    response.width = output_w;
    response.height = output_h;
    response.format = "raw_rgb";
    response.generation_time_ms = total_sec * 1000.0;
    response.text_encoding_time_ms = text_enc_sec * 1000.0;
    response.denoising_time_ms = denoise_sec * 1000.0;
    response.vae_decoding_time_ms = vae_sec * 1000.0;
    response.num_inference_steps = config_.num_inference_steps;
    
    // Legacy fields for compatibility
    response.image_data = rgb_data;
    response.inference_time_ms = static_cast<float>(total_sec * 1000.0);

    std::cout << "[SUCCESS] Image generated: " << output_w << "x" << output_h 
              << " (base64 size: " << b64_json.length() << " chars)" << std::endl;
    
    return response;
}

// ============================================================================
// denoise() – the main denoising loop
// ============================================================================

std::vector<float> SDPipeline::denoise(
    const std::vector<float>& latents,
    const std::vector<float>& text_embeddings,
    const std::vector<float>& pooled_embeddings,
    const std::vector<float>& controlnet_cond) {

    std::cout << "Running denoising loop (" << config_.num_inference_steps
              << " steps)..." << std::endl;

    if (!denoiser_) {
        std::cout << "  WARNING: No denoiser loaded" << std::endl;
        return latents;
    }

    // Delegate to variant-specific denoiser
    return denoiser_->denoise(latents, text_embeddings, pooled_embeddings,
                              *scheduler_, config_, controlnet_cond);
}

// ============================================================================
// process_control_image - Encode control image to latent space for SD3 ControlNet
// ============================================================================

std::vector<float> SDPipeline::process_control_image(
    const std::vector<uint8_t>& image,
    const std::vector<uint8_t>& mask) {
    std::cout << "Processing control image with " << config_.controlnet_type << std::endl;
    
    // SD3 ControlNet expects control image in LATENT SPACE (16 channels for SD3)
    // NOT raw RGB! This is different from SD1.5 ControlNet which uses RGB.
    // We need to encode the control image through VAE encoder.
    
    if (!components_.count(ComponentType::VAE_ENCODER)) {
        std::cerr << "  ERROR: VAE encoder required for SD3 ControlNet but not found!" << std::endl;
        return {};
    }

    ControlNetType cn_type = controlnet_from_string(config_.controlnet_type);

    if (is_inpainting_controlnet(cn_type) && !mask.empty()) {
        // Mask-aware ControlNet (SD3 ControlNet-Inpainting): condition on
        // (masked-image latents [16ch] ++ downsampled mask [1ch]) = 17ch,
        // matching alimama's reference pipeline (prepare_image_with_mask):
        //   masked_image = image; masked_image[mask > 0.5] = -1 (black)
        //   image_latents = (vae.encode(masked_image) - shift) * scale
        //   mask = 1 - nearest_downsample(binarize(mask), latent_size)
        //   control_image = cat([image_latents, mask], dim=1)
        std::cout << "  Building mask-aware conditioning (masked-image latents + mask channel)..." << std::endl;

        int w = config_.width;
        int h = config_.height;
        size_t expected_size = static_cast<size_t>(w) * h * 3;
        if (image.size() != expected_size || mask.size() != expected_size) {
            std::cerr << "  ERROR: control image/mask size mismatch (expected "
                      << expected_size << " bytes each, got image=" << image.size()
                      << ", mask=" << mask.size() << ")" << std::endl;
            return {};
        }

        // Build the masked image: black out (0,0,0) any pixel whose mask
        // value is "on" (>127), matching the RGB->[-1,1] normalization that
        // maps 0 -> -1.0 inside encode_image_to_latents().
        std::vector<uint8_t> masked_image(image);
        for (int i = 0; i < w * h; i++) {
            if (mask[i * 3] > 127) {
                masked_image[i * 3 + 0] = 0;
                masked_image[i * 3 + 1] = 0;
                masked_image[i * 3 + 2] = 0;
            }
        }

        std::cout << "  Encoding masked image to latent space via VAE encoder..." << std::endl;
        // Unlike the plain control-image path below, the masked-image latents
        // for this ControlNet ARE shift-corrected, per the reference pipeline.
        auto image_latents = encode_image_to_latents(masked_image, /*apply_shift=*/true);
        if (image_latents.empty()) {
            std::cerr << "  ERROR: Failed to encode masked image to latents" << std::endl;
            return {};
        }

        int latent_h = h / 8;
        int latent_w = w / 8;

        // Downsample the mask to latent resolution (nearest-neighbor, matching
        // torch.nn.functional.interpolate's default mode) and invert so that
        // 1 = keep original content, 0 = region to inpaint.
        std::vector<float> mask_latent(static_cast<size_t>(latent_h) * latent_w);
        for (int ly = 0; ly < latent_h; ly++) {
            for (int lx = 0; lx < latent_w; lx++) {
                int sy = ly * 8;
                int sx = lx * 8;
                uint8_t mval = mask[(static_cast<size_t>(sy) * w + sx) * 3];
                mask_latent[static_cast<size_t>(ly) * latent_w + lx] = (mval > 127) ? 0.0f : 1.0f;
            }
        }

        std::vector<float> control_cond(image_latents.size() + mask_latent.size());
        std::copy(image_latents.begin(), image_latents.end(), control_cond.begin());
        std::copy(mask_latent.begin(), mask_latent.end(), control_cond.begin() + image_latents.size());

        std::cout << "  Mask-aware control conditioning built: " << image_latents.size()
                  << " image-latent values + " << mask_latent.size() << " mask values" << std::endl;
        return control_cond;
    }

    std::cout << "  Encoding control image to latent space via VAE encoder..." << std::endl;
    // ControlNet uses latents * scale (no shift subtraction), per HF reference
    auto latent_control = encode_image_to_latents(image, /*apply_shift=*/false);
    
    if (latent_control.empty()) {
        std::cerr << "  ERROR: Failed to encode control image to latents" << std::endl;
        return {};
    }
    
    std::cout << "  Control image encoded: " << latent_control.size() << " values" << std::endl;
    return latent_control;
}

// ============================================================================
// encode_image_to_latents - for img2img support
// ============================================================================

std::vector<float> SDPipeline::encode_image_to_latents(
    const std::vector<uint8_t>& rgb_image, bool apply_shift) {

    OnnxModel* vae_encoder = nullptr;
    if (components_.count(ComponentType::VAE_ENCODER)) {
        vae_encoder = components_[ComponentType::VAE_ENCODER].get();
    }

    if (!vae_encoder) {
        std::cout << "  WARNING: No VAE encoder available, returning empty latents" << std::endl;
        // Return zero latents as fallback
        int latent_ch = denoiser_->latent_channels();
        int latent_h = config_.height / 8;
        int latent_w = config_.width / 8;
        return std::vector<float>(latent_ch * latent_h * latent_w, 0.0f);
    }

    std::cout << "  Encoding input image to latents via VAE encoder..." << std::endl;

    // Preprocess: convert uint8 RGB [0,255] to float [-1, 1]
    // Input shape: [1, 3, H, W] for VAE encoder
    int h = config_.height;
    int w = config_.width;
    std::vector<float> input_tensor(1 * 3 * h * w);

    // Check if input image matches expected dimensions
    size_t expected_size = static_cast<size_t>(h * w * 3);
    if (rgb_image.size() != expected_size) {
        std::cout << "  WARNING: Input image size mismatch. Expected " << expected_size
                  << " but got " << rgb_image.size() << ". Using zero latents." << std::endl;
        int latent_ch = denoiser_->latent_channels();
        int latent_h = config_.height / 8;
        int latent_w = config_.width / 8;
        return std::vector<float>(latent_ch * latent_h * latent_w, 0.0f);
    }

    // Convert RGB [H, W, 3] uint8 to [1, 3, H, W] float normalized to [-1, 1]
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t src_idx = (y * w + x) * 3;
            // Channel order: R, G, B
            for (int c = 0; c < 3; c++) {
                size_t dst_idx = c * h * w + y * w + x;
                // Normalize: [0, 255] -> [0, 1] -> [-1, 1]
                input_tensor[dst_idx] = (rgb_image[src_idx + c] / 255.0f) * 2.0f - 1.0f;
            }
        }
    }

    // Run VAE encoder
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {1, 3, static_cast<int64_t>(h), static_cast<int64_t>(w)};

    // Check model's expected input type
    auto input_type = vae_encoder->get_input_type(0);
    std::vector<Ort::Value> inputs;
    std::vector<Ort::Float16_t> input_tensor_fp16;

    if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        // Convert float32 to float16 for NPU-optimized VAE encoder
        input_tensor_fp16.resize(input_tensor.size());
        for (size_t i = 0; i < input_tensor.size(); i++) {
            input_tensor_fp16[i] = Ort::Float16_t(input_tensor[i]);
        }
        auto input_tensor_ort = Ort::Value::CreateTensor<Ort::Float16_t>(
            mem_info, input_tensor_fp16.data(), input_tensor_fp16.size(),
            input_shape.data(), input_shape.size());
        inputs.push_back(std::move(input_tensor_ort));
    } else {
        // Use float32 directly (CPU or non-replaced models)
        auto input_tensor_ort = Ort::Value::CreateTensor<float>(
            mem_info, input_tensor.data(), input_tensor.size(),
            input_shape.data(), input_shape.size());
        inputs.push_back(std::move(input_tensor_ort));
    }

    auto& input_names = vae_encoder->get_input_names();
    auto& output_names = vae_encoder->get_output_names();

    std::vector<const char*> input_name_ptrs, output_name_ptrs;
    for (auto& n : input_names) input_name_ptrs.push_back(n.c_str());
    for (auto& n : output_names) output_name_ptrs.push_back(n.c_str());

    auto t0 = std::chrono::high_resolution_clock::now();
    auto outputs = vae_encoder->run(inputs, input_name_ptrs, output_name_ptrs);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto encode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "  VAE encode: " << encode_ms << " ms" << std::endl;

    if (outputs.empty()) {
        std::cout << "  WARNING: VAE encoder produced no output" << std::endl;
        int latent_ch = denoiser_->latent_channels();
        int latent_h = config_.height / 8;
        int latent_w = config_.width / 8;
        return std::vector<float>(latent_ch * latent_h * latent_w, 0.0f);
    }

    // Extract latents from output (handle both fp32 and fp16 outputs)
    auto& output = outputs[0];
    auto shape_info = output.GetTensorTypeAndShapeInfo();
    auto shape = shape_info.GetShape();
    auto elem_type = shape_info.GetElementType();
    size_t total_size = 1;
    for (auto s : shape) total_size *= s;

    std::vector<float> latents(total_size);
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const Ort::Float16_t* data = output.GetTensorData<Ort::Float16_t>();
        for (size_t i = 0; i < total_size; i++) {
            latents[i] = data[i].ToFloat();
        }
    } else {
        const float* data = output.GetTensorData<float>();
        std::copy(data, data + total_size, latents.begin());
    }

    // VAE encoder outputs:
    //   - Either mean+log_variance concatenated: [1, 2*latent_ch, H/8, W/8]
    //   - Or already-sampled latents: [1, latent_ch, H/8, W/8]
    int latent_ch = denoiser_->latent_channels();
    int latent_h = h / 8;
    int latent_w = w / 8;
    size_t latent_size = latent_ch * latent_h * latent_w;

    std::vector<float> latents_sampled;

    if (latents.size() >= latent_size * 2) {
        // DiagonalGaussianDistribution: output is [mean | logvar], take mean (first half)
        latents_sampled.assign(latents.begin(), latents.begin() + latent_size);
        std::cout << "  VAE encoder: DiagonalGaussian output, using mean" << std::endl;
    } else if (latents.size() >= latent_size) {
        // Already-sampled latents (some encoders output this directly)
        latents_sampled.assign(latents.begin(), latents.begin() + latent_size);
        std::cout << "  VAE encoder: direct latent output" << std::endl;
    } else {
        std::cerr << "  WARNING: VAE encoder output size mismatch! Expected " << latent_size
                  << " or " << (latent_size * 2) << ", got " << latents.size() << std::endl;
        return std::vector<float>(latent_size, 0.0f);
    }
    
    // Apply VAE scaling factor (per-variant, sourced from the loaded VAE decoder
    // so encode/decode always agree even for models without a hardcoded family).
    // img2img (apply_shift=true):  encode = (raw - shift) * scale  (inverse of decode: raw = latents/scale + shift)
    // ControlNet (apply_shift=false): encode = raw * scale          (per HF ControlNet reference)
    float scaling_factor = vae_decoder_->latent_scaling_factor();
    float shift_factor   = apply_shift ? vae_decoder_->latent_shift_factor() : 0.0f;
    for (auto& v : latents_sampled) {
        v = (v - shift_factor) * scaling_factor;
    }

    std::cout << "  Encoded latents shape: [";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "] -> sampled [1, " << latent_ch << ", " << latent_h << ", " << latent_w << "]" << std::endl;

    return latents_sampled;
}

} // namespace sd_npu
