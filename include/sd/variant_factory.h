// variant_factory.h - VariantComponents struct used by the registry system
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "sd_types.h"
#include "i_text_encoder.h"
#include "i_vae_decoder.h"
#include "i_denoiser.h"
#include "scheduler.h"
#include "onnx_model.h"
#include "clip_tokenizer.h"
#include "controlnet_runner.h"
#include <memory>
#include <vector>
#include <map>

namespace sd_npu {

// ============================================================================
// VariantComponents: Bundle of variant-specific implementations
// ============================================================================

struct VariantComponents {
    std::unique_ptr<ITextEncoder> text_encoder;
    std::unique_ptr<IVaeDecoder>  vae_decoder;
    std::unique_ptr<IDenoiser>    denoiser;
    std::unique_ptr<Scheduler>    scheduler;
    
    // List of ComponentType entries that need to be loaded from ONNX
    std::vector<ComponentType> required_models;
};

} // namespace sd_npu
