// flux2_denoiser.h - FLUX.2-klein transformer denoiser
// Copyright (C) 2025 Advanced Micro Devices, Inc.
//
// FLUX.2-klein uses a flow-matching transformer that operates on packed latent
// tokens. Unlike FLUX.1 (3-column ids, no batch dim), klein uses 4-column int64
// position ids WITH a batch dim, a Qwen3 sequence embedding (7680-dim), no
// pooled projection, no guidance, and no CFG.
//
// This denoiser is self-contained: it computes its own dynamic-shifting sigma
// schedule (empirical mu + exponential shift) and performs the flow-match Euler
// integration internally, ignoring the Scheduler passed by the pipeline.
//
// I/O contract:
//   - Input latents  : flat randn, interpreted as packed [128, latH, latW]
//                      (latH = latW = height/16). 524288 floats for 1024x1024.
//   - Return latents : unpacked VAE latents [32, H/8, W/8] with the BatchNorm
//                      denorm already applied (ready for the klein VAE ONNX).
//                      Denorm must happen in packed 128-channel space before
//                      unpatchify, so it is done here rather than in the VAE.

#pragma once

#include "i_denoiser.h"
#include "onnx_model.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sd_npu {

class Flux2Denoiser : public IDenoiser {
public:
    Flux2Denoiser(
        const std::map<ComponentType, std::unique_ptr<OnnxModel>>& components,
        const std::string& model_root,
        int text_seq_len = 256,
        int text_embed_dim = 7680);

    std::vector<float> denoise(
        const std::vector<float>& latents,
        const std::vector<float>& text_embeddings,
        const std::vector<float>& pooled_embeddings,
        Scheduler& scheduler,
        const SDConfig& config,
        const std::vector<float>& controlnet_cond = {}) override;

    std::vector<ComponentType> required_components() const override {
        return {ComponentType::TRANSFORMER};
    }

    // klein latent space: VAE has 32 channels; the pipeline sizes the initial
    // noise as latent_channels * (H/8) * (W/8). For klein that must equal
    // 128 * (H/16) * (W/16), which holds when latent_channels() == 32.
    int latent_channels() const override { return 32; }

private:
    OnnxModel* model_ = nullptr;
    int text_seq_len_   = 256;
    int text_embed_dim_ = 7680;

    // BatchNorm denorm statistics (128 packed channels), loaded from
    // bn.running_x.safetensors. Empty if not found.
    std::vector<float> bn_mean_;
    std::vector<float> bn_std_;   // sqrt(running_var + batch_norm_eps)

    static double compute_empirical_mu(int image_seq_len, int num_steps);
    void load_bn_stats(const std::string& model_root);
};

} // namespace sd_npu
