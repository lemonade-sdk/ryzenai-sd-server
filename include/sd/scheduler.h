// scheduler.h - Noise schedulers for Stable Diffusion denoising
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "sd_types.h"
#include <vector>
#include <memory>
#include <cmath>

namespace sd_npu {

// ============================================================================
// Scheduler: computes timestep schedules and performs denoising steps
// ============================================================================

class Scheduler {
public:
    virtual ~Scheduler() = default;

    // Initialize the schedule for N inference steps
    virtual void set_timesteps(int num_inference_steps) = 0;

    // Get computed timesteps and sigmas
    const std::vector<float>& timesteps() const { return timesteps_; }
    const std::vector<float>& sigmas() const { return sigmas_; }

    // Scale latents by init_noise_sigma (called once before the loop)
    virtual void scale_initial_latents(std::vector<float>& latents) const = 0;

    // Scale model input at each step (e.g. EulerDiscrete divides by sqrt(sigma^2+1))
    virtual void scale_model_input(std::vector<float>& latents, int step) const = 0;

    // Perform one scheduler step: update latents in-place
    virtual void step(
        const std::vector<float>& noise_pred,
        int step,
        std::vector<float>& latents
    ) = 0;

    int num_inference_steps() const { return num_inference_steps_; }

protected:
    int num_inference_steps_ = 0;
    std::vector<float> timesteps_;
    std::vector<float> sigmas_;
};

// ============================================================================
// FlowMatchEulerDiscreteScheduler (SD3 / SD3.5)
// ============================================================================

class FlowMatchEulerScheduler : public Scheduler {
public:
    explicit FlowMatchEulerScheduler(float shift = 3.0f, int num_train_timesteps = 1000);

    void set_timesteps(int num_inference_steps) override;
    void scale_initial_latents(std::vector<float>& latents) const override;
    void scale_model_input(std::vector<float>& latents, int step) const override;
    void step(const std::vector<float>& noise_pred, int step, std::vector<float>& latents) override;

private:
    float shift_;
    int num_train_timesteps_;
};

// ============================================================================
// EulerDiscreteScheduler (SDXL, SD-Turbo)
// ============================================================================

class EulerDiscreteScheduler : public Scheduler {
public:
    explicit EulerDiscreteScheduler(int num_train_timesteps = 1000,
                                    float beta_start = 0.00085f,
                                    float beta_end = 0.012f,
                                    int steps_offset = 1,
                                    const std::string& timestep_spacing = "leading");

    void set_timesteps(int num_inference_steps) override;
    void scale_initial_latents(std::vector<float>& latents) const override;
    void scale_model_input(std::vector<float>& latents, int step) const override;
    void step(const std::vector<float>& noise_pred, int step, std::vector<float>& latents) override;

private:
    int num_train_timesteps_;
    float beta_start_;
    float beta_end_;
    int steps_offset_;
    std::string timestep_spacing_;
    std::vector<float> alphas_cumprod_;
};

// ============================================================================
// PNDMScheduler (SD 1.5)
// ============================================================================

class PNDMScheduler : public Scheduler {
public:
    explicit PNDMScheduler(int num_train_timesteps = 1000,
                           float beta_start = 0.00085f,
                           float beta_end = 0.012f);

    void set_timesteps(int num_inference_steps) override;
    void scale_initial_latents(std::vector<float>& latents) const override;
    void scale_model_input(std::vector<float>& latents, int step) const override;
    void step(const std::vector<float>& noise_pred, int step, std::vector<float>& latents) override;

    // Reset state between generations
    void reset();

private:
    void compute_alphas_and_sigmas();

    int num_train_timesteps_;
    float beta_start_;
    float beta_end_;
    std::vector<float> alphas_cumprod_;

    // Multi-step buffer for Adams-Bashforth
    std::vector<std::vector<float>> ets_buffer_;
    int counter_ = 0;
};

// ============================================================================
// Factory
// ============================================================================

// Create a scheduler from type enum
std::unique_ptr<Scheduler> create_scheduler(SchedulerType type, const std::string& timestep_spacing = "leading");

} // namespace sd_npu
