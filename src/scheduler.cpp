// scheduler.cpp - Noise scheduler implementations
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "scheduler.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace sd_npu {

// ============================================================================
// FlowMatchEulerDiscreteScheduler (SD3 / SD3.5)
// Reference: diffusers/schedulers/scheduling_flow_match_euler_discrete.py
// ============================================================================

FlowMatchEulerScheduler::FlowMatchEulerScheduler(float shift, int num_train_timesteps)
    : shift_(shift), num_train_timesteps_(num_train_timesteps) {}

void FlowMatchEulerScheduler::set_timesteps(int num_inference_steps) {
    num_inference_steps_ = num_inference_steps;
    int N = num_inference_steps;

    // Linearly spaced sigmas from 1.0 to 0.0 (N+1 values)
    std::vector<float> base_sigmas(N + 1);
    for (int i = 0; i <= N; i++) {
        base_sigmas[i] = 1.0f - (float)i / (float)N;
    }

    // Apply time shift: sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
    sigmas_.resize(N + 1);
    for (int i = 0; i <= N; i++) {
        float s = base_sigmas[i];
        sigmas_[i] = shift_ * s / (1.0f + (shift_ - 1.0f) * s);
    }

    // Timesteps = sigma * num_train_timesteps (only N values)
    timesteps_.resize(N);
    for (int i = 0; i < N; i++) {
        timesteps_[i] = sigmas_[i] * (float)num_train_timesteps_;
    }

    std::cout << "  FlowMatch schedule (shift=" << shift_ << ", steps=" << N << ")" << std::endl;
}

void FlowMatchEulerScheduler::scale_initial_latents(std::vector<float>& latents) const {
    // FlowMatch: init_noise_sigma = sigma_max
    float init_noise_sigma = sigmas_[0];
    if (init_noise_sigma != 1.0f) {
        std::cout << "  Scaling latents by init_noise_sigma=" << init_noise_sigma << std::endl;
        for (auto& v : latents) v *= init_noise_sigma;
    }
}

void FlowMatchEulerScheduler::scale_model_input(std::vector<float>& /*latents*/, int /*step*/) const {
    // FlowMatch: identity (no scaling)
}

void FlowMatchEulerScheduler::step(
    const std::vector<float>& noise_pred, int step, std::vector<float>& latents) {
    // sigma_next - sigma (negative, denoising goes from high to 0)
    float dt = sigmas_[step + 1] - sigmas_[step];
    for (size_t i = 0; i < latents.size(); i++) {
        latents[i] += noise_pred[i] * dt;
    }
}

// ============================================================================
// EulerDiscreteScheduler (SDXL, SD-Turbo)
// Reference: diffusers/schedulers/scheduling_euler_discrete.py
// ============================================================================

EulerDiscreteScheduler::EulerDiscreteScheduler(
    int num_train_timesteps, float beta_start, float beta_end, int steps_offset,
    const std::string& timestep_spacing)
    : num_train_timesteps_(num_train_timesteps),
      beta_start_(beta_start), beta_end_(beta_end),
      steps_offset_(steps_offset), timestep_spacing_(timestep_spacing) {}

void EulerDiscreteScheduler::set_timesteps(int num_inference_steps) {
    num_inference_steps_ = num_inference_steps;
    int N = num_inference_steps;

    // Compute betas (scaled_linear schedule)
    float sqrt_beta_start = std::sqrt(beta_start_);
    float sqrt_beta_end = std::sqrt(beta_end_);
    std::vector<float> betas(num_train_timesteps_);
    for (int i = 0; i < num_train_timesteps_; i++) {
        float t = (float)i / (float)(num_train_timesteps_ - 1);
        float sqrt_beta = sqrt_beta_start + (sqrt_beta_end - sqrt_beta_start) * t;
        betas[i] = sqrt_beta * sqrt_beta;
    }

    // Compute alphas and alphas_cumprod
    alphas_cumprod_.resize(num_train_timesteps_);
    alphas_cumprod_[0] = 1.0f - betas[0];
    for (int i = 1; i < num_train_timesteps_; i++) {
        alphas_cumprod_[i] = alphas_cumprod_[i-1] * (1.0f - betas[i]);
    }

    // Compute all sigmas: sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)
    std::vector<float> all_sigmas(num_train_timesteps_);
    for (int i = 0; i < num_train_timesteps_; i++) {
        all_sigmas[i] = std::sqrt((1.0f - alphas_cumprod_[i]) / alphas_cumprod_[i]);
    }

    // ══ Compute timesteps based on spacing ═════════════════════════════════
    std::vector<float> ts;

    if (timestep_spacing_ == "trailing") {
        float step_ratio = (float)num_train_timesteps_ / (float)N;
        for (int i = 0; i < N; i++) {
            float v = (float)num_train_timesteps_ - (float)i * step_ratio;
            ts.push_back(std::round(v) - 1.0f);
        }
    } else if (timestep_spacing_ == "linspace") {
        for (int i = N - 1; i >= 0; i--) {
            float v = (float)(num_train_timesteps_ - 1) * (float)i / (float)(N - 1 > 0 ? N - 1 : 1);
            ts.push_back(v);
        }
    } else {
        // "leading" (default)
        int step_ratio = num_train_timesteps_ / N;
        for (int i = N - 1; i >= 0; i--) {
            ts.push_back(std::round((float)i * (float)step_ratio) + (float)steps_offset_);
        }
    }

    // ══ Interpolate sigmas at timestep positions ═══════════════════════════
    timesteps_.resize(ts.size());
    sigmas_.resize(ts.size() + 1);

    for (size_t i = 0; i < ts.size(); i++) {
        timesteps_[i] = ts[i];
        int idx = (int)std::round(ts[i]);
        if (idx < 0) idx = 0;
        if (idx >= num_train_timesteps_) idx = num_train_timesteps_ - 1;
        sigmas_[i] = all_sigmas[idx];
    }
    sigmas_[ts.size()] = 0.0f;

    std::cout << "  EulerDiscrete schedule (spacing=" << timestep_spacing_
              << ", steps=" << N
              << ", timesteps=[";
    for (size_t i = 0; i < timesteps_.size(); i++) {
        if (i) std::cout << ", ";
        std::cout << (int)timesteps_[i];
    }
    std::cout << "], sigmas=[";
    for (size_t i = 0; i < sigmas_.size(); i++) {
        if (i) std::cout << ", ";
        std::cout << sigmas_[i];
    }
    std::cout << "])" << std::endl;
}

void EulerDiscreteScheduler::scale_initial_latents(std::vector<float>& latents) const {
    // init_noise_sigma = sqrt(sigma_max^2 + 1)
    float init_noise_sigma = std::sqrt(sigmas_[0] * sigmas_[0] + 1.0f);
    std::cout << "  Scaling latents by init_noise_sigma=" << init_noise_sigma
              << " (sigma_max=" << sigmas_[0] << ")" << std::endl;
    for (auto& v : latents) v *= init_noise_sigma;
}

void EulerDiscreteScheduler::scale_model_input(std::vector<float>& latents, int step) const {
    // latents = latents / sqrt(sigma^2 + 1)
    float sigma_t = sigmas_[step];
    float scale_factor = 1.0f / std::sqrt(sigma_t * sigma_t + 1.0f);
    for (auto& v : latents) v *= scale_factor;
}

void EulerDiscreteScheduler::step(
    const std::vector<float>& noise_pred, int step, std::vector<float>& latents) {
    // prev_sample = sample + model_output * (sigma_next - sigma)
    float dt = sigmas_[step + 1] - sigmas_[step];
    for (size_t i = 0; i < latents.size(); i++) {
        latents[i] += noise_pred[i] * dt;
    }
}

// ============================================================================
// PNDMScheduler (SD 1.5)
// Reference: diffusers/schedulers/scheduling_pndm.py with skip_prk_steps=True
// ============================================================================

PNDMScheduler::PNDMScheduler(int num_train_timesteps, float beta_start, float beta_end)
    : num_train_timesteps_(num_train_timesteps),
      beta_start_(beta_start), beta_end_(beta_end) {}

void PNDMScheduler::compute_alphas_and_sigmas() {
    float sqrt_beta_start = std::sqrt(beta_start_);
    float sqrt_beta_end = std::sqrt(beta_end_);
    std::vector<float> betas(num_train_timesteps_);
    for (int i = 0; i < num_train_timesteps_; i++) {
        float t = (float)i / (float)(num_train_timesteps_ - 1);
        float sqrt_beta = sqrt_beta_start + (sqrt_beta_end - sqrt_beta_start) * t;
        betas[i] = sqrt_beta * sqrt_beta;
    }

    alphas_cumprod_.resize(num_train_timesteps_);
    alphas_cumprod_[0] = 1.0f - betas[0];
    for (int i = 1; i < num_train_timesteps_; i++) {
        alphas_cumprod_[i] = alphas_cumprod_[i-1] * (1.0f - betas[i]);
    }
}

void PNDMScheduler::set_timesteps(int num_inference_steps) {
    num_inference_steps_ = num_inference_steps;
    int N = num_inference_steps;

    compute_alphas_and_sigmas();

    // Compute all sigmas from alphas_cumprod
    std::vector<float> all_sigmas(num_train_timesteps_ + 1);
    for (int i = 0; i < num_train_timesteps_; i++) {
        all_sigmas[i] = std::sqrt((1.0f - alphas_cumprod_[i]) / alphas_cumprod_[i]);
    }
    all_sigmas[num_train_timesteps_] = 0.0f;

    // PNDM: linspace timesteps, reversed
    timesteps_.resize(N);
    sigmas_.resize(N + 1);
    for (int i = 0; i < N; i++) {
        float t_float = ((float)(num_train_timesteps_ - 1) / (float)(N - 1)) * (float)(N - 1 - i);
        int t_idx = (int)std::round(t_float);
        if (t_idx < 0) t_idx = 0;
        if (t_idx >= num_train_timesteps_) t_idx = num_train_timesteps_ - 1;
        timesteps_[i] = (float)t_idx;
        sigmas_[i] = all_sigmas[t_idx];
    }
    sigmas_[N] = 0.0f;

    reset();
    std::cout << "  PNDM schedule (steps=" << N << ")" << std::endl;
}

void PNDMScheduler::scale_initial_latents(std::vector<float>& /*latents*/) const {
    // PNDM: no init_noise_sigma scaling
    std::cout << "  No init_noise_sigma scaling (PNDM)" << std::endl;
}

void PNDMScheduler::scale_model_input(std::vector<float>& /*latents*/, int /*step*/) const {
    // PNDM: identity (no scaling)
}

void PNDMScheduler::reset() {
    ets_buffer_.clear();
    ets_buffer_.resize(4);
    counter_ = 0;
}

void PNDMScheduler::step(
    const std::vector<float>& noise_pred, int step, std::vector<float>& latents) {

    size_t size = latents.size();

    if (ets_buffer_[0].empty()) {
        for (auto& buf : ets_buffer_) buf.resize(size, 0.0f);
    }

    // Store current model output in circular buffer
    int buffer_idx = counter_ % 4;
    ets_buffer_[buffer_idx] = noise_pred;

    // PLMS multi-step combination
    std::vector<float> combined_output(size);
    if (counter_ < 3) {
        // First 3 steps: 1st-order Euler
        combined_output = noise_pred;
    } else {
        // 4th order Adams-Bashforth:
        // (55*e_n - 59*e_{n-1} + 37*e_{n-2} - 9*e_{n-3}) / 24
        int idx0 = buffer_idx;
        int idx1 = (buffer_idx + 3) % 4;
        int idx2 = (buffer_idx + 2) % 4;
        int idx3 = (buffer_idx + 1) % 4;
        for (size_t i = 0; i < size; i++) {
            combined_output[i] = (55.0f * ets_buffer_[idx0][i]
                                - 59.0f * ets_buffer_[idx1][i]
                                + 37.0f * ets_buffer_[idx2][i]
                                -  9.0f * ets_buffer_[idx3][i]) / 24.0f;
        }
    }
    counter_++;

    // DDPM-style step using alphas_cumprod
    float sigma_t = sigmas_[step];
    float sigma_prev = sigmas_[step + 1];

    float alpha_prod_t      = 1.0f / (1.0f + sigma_t * sigma_t);
    float alpha_prod_t_prev = 1.0f / (1.0f + sigma_prev * sigma_prev);

    float beta_prod_t      = 1.0f - alpha_prod_t;
    float beta_prod_t_prev = 1.0f - alpha_prod_t_prev;

    float sample_coeff = std::sqrt(alpha_prod_t_prev / alpha_prod_t);
    float denom = std::sqrt(alpha_prod_t * beta_prod_t_prev)
                + std::sqrt(alpha_prod_t * beta_prod_t * alpha_prod_t_prev);

    for (size_t i = 0; i < size; i++) {
        latents[i] = sample_coeff * latents[i]
                   - (alpha_prod_t_prev - alpha_prod_t) * combined_output[i] / denom;
    }
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<Scheduler> create_scheduler(SchedulerType type, const std::string& timestep_spacing) {
    switch (type) {
        case SchedulerType::FLOW_MATCH_EULER:
            return std::make_unique<FlowMatchEulerScheduler>();
        case SchedulerType::EULER_DISCRETE:
            return std::make_unique<EulerDiscreteScheduler>(1000, 0.00085f, 0.012f, 1, timestep_spacing);
        case SchedulerType::PNDM:
            return std::make_unique<PNDMScheduler>();
    }
    return std::make_unique<EulerDiscreteScheduler>(1000, 0.00085f, 0.012f, 1, timestep_spacing);
}

} // namespace sd_npu
