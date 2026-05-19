// sd_pipeline.h - Unified Stable Diffusion Pipeline orchestrator
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "sd_types.h"
#include "onnx_model.h"
#include "scheduler.h"
#include "i_text_encoder.h"
#include "i_vae_decoder.h"
#include "i_denoiser.h"
#include "clip_tokenizer.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sd_npu {

// Forward declaration
class ControlNetRunner;

struct GenerationTiming {
    double text_encode_sec = 0;
    double denoise_sec = 0;
    double vae_decode_sec = 0;
    double total_sec = 0;
    int steps = 0;
};

class SDPipeline {
public:
    explicit SDPipeline(const SDConfig& config);
    ~SDPipeline();

    /// Generate an image from a text prompt. Returns ImageResponse with base64-encoded PNG and metadata.
    ImageResponse generate(
        const std::string& prompt,
        const std::string& negative_prompt = "",
        const std::vector<uint8_t>& control_image = {});

    /// Get timing from the last generate() call.
    const GenerationTiming& last_timing() const { return last_timing_; }

    /// Update generation config (steps, guidance_scale, seed) for next generation.
    /// Does not reload models - only updates parameters that don't require reload.
    void update_config(const SDConfig& new_config);

private:
    // ── Initialisation ────────────────────────────────────────
    void setup_onnx_session_options();
    void load_components();
    void load_onnx_models();
    void load_scheduler_config();
    void load_tokenizers();


    // ── Denoising loop ────────────────────────────────────────
    std::vector<float> denoise(
        const std::vector<float>& latents,
        const std::vector<float>& text_embeddings,
        const std::vector<float>& pooled_embeddings,
        const std::vector<float>& controlnet_cond = {});

    // ── ControlNet ────────────────────────────────────────────
    std::vector<float> process_control_image(const std::vector<uint8_t>& image);

    // ── Image-to-image support ───────────────────────────────
    std::vector<float> encode_image_to_latents(const std::vector<uint8_t>& rgb_image,
                                               bool apply_shift = true);

    // ── Owns all lifetime-sensitive buffers for one model call ──
    struct InputBuffers {
        // SDXL / SD15 buffers
        std::vector<int64_t>  ts_i64;
        std::vector<float>    ts_f32;
        std::vector<double>   ts_f64;
        std::vector<float>    time_ids_data;
        std::vector<int64_t>  time_ids_i64;
        // SD3 fp16 buffers
        std::vector<Ort::Float16_t> hs_fp16;
        std::vector<Ort::Float16_t> timestep_fp16;
        std::vector<Ort::Float16_t> ehs_fp16;
        std::vector<Ort::Float16_t> pooled_fp16;
        std::vector<std::vector<Ort::Float16_t>> ctrl_fp16;
        // Copy of latent_model_input (needed for two-pass when we take a single slice)
        std::vector<float> lmi;
        // text/pooled embedding copy for this pass
        std::vector<float> emb;
        std::vector<float> pool;
    };

    // ── Members ───────────────────────────────────────────────
    SDConfig config_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;

    std::map<ComponentType, std::unique_ptr<OnnxModel>> components_;

    CLIPTokenizer tokenizer_;
    CLIPTokenizer tokenizer_2_;

    // Sub-modules (created from factory)
    std::unique_ptr<ITextEncoder> text_encoder_;
    std::unique_ptr<IVaeDecoder>  vae_decoder_;
    std::unique_ptr<IDenoiser>    denoiser_;
    std::unique_ptr<Scheduler>    scheduler_;

    // ControlNet helper (optional, created if ControlNet is loaded)
    std::unique_ptr<ControlNetRunner> controlnet_runner_;

    GenerationTiming last_timing_;
};

} // namespace sd_npu

    