// onnx_model.cpp - ONNX Runtime model wrapper implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "onnx_model.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace sd_npu {

namespace {

// The vendor RyzenAI NPU execution provider occasionally returns all-NaN
// tensors or a hard AIE command timeout on an otherwise-valid inference call
// on a long-lived session (observed alternating valid/NaN output across
// consecutive requests, and sporadic ERT_CMD_STATE_TIMEOUT / AIE_STREAM
// errors) -- this is inside the closed-source EP DLL and not something this
// codebase can fix at the source. Retrying the same Run() call a few times
// is a bounded mitigation; if it never recovers, callers get a clear error
// instead of silently returning a corrupted result.
constexpr int kMaxNpuRetries = 3;

bool output_has_nan(const std::vector<Ort::Value>& outputs) {
    for (const auto& out : outputs) {
        if (!out.IsTensor()) continue;
        auto info = out.GetTensorTypeAndShapeInfo();
        size_t count = info.GetElementCount();
        switch (info.GetElementType()) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                const float* data = out.GetTensorData<float>();
                for (size_t i = 0; i < count; i++) {
                    if (std::isnan(data[i])) return true;
                }
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
                const Ort::Float16_t* data = out.GetTensorData<Ort::Float16_t>();
                for (size_t i = 0; i < count; i++) {
                    if (std::isnan(data[i].ToFloat())) return true;
                }
                break;
            }
            default:
                break;
        }
    }
    return false;
}

bool is_transient_npu_error(const Ort::Exception& e) {
    std::string msg = e.what();
    return msg.find("TIMEOUT") != std::string::npos ||
           msg.find("AIE") != std::string::npos;
}

} // namespace

// ============================================================================
// OnnxModel Implementation
// ============================================================================

OnnxModel::OnnxModel(const std::string& model_path,
                     Ort::SessionOptions& /*base_options*/,
                     Ort::Env& env)
    : model_path_(model_path) {

    std::cout << "Loading ONNX model: " << model_path << std::endl;

    // Create fresh session options for this model
    Ort::SessionOptions model_options;
    model_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    // Check for external data file (.onnx.data)
    std::string data_path = model_path + ".data";
    bool has_external_data = fs::exists(data_path);
    if (has_external_data) {
        std::cout << "  Found external data: " << data_path << std::endl;
    }

    // Check for DD cache directory - MUST be set BEFORE registering custom ops
    fs::path model_dir = fs::path(model_path).parent_path();
    fs::path cache_dir = model_dir / "cache";
    fs::path dot_cache_dir = model_dir / ".cache";

    bool has_cache = fs::exists(cache_dir) || fs::exists(dot_cache_dir);
    if (has_cache) {
        fs::path actual_cache = fs::exists(cache_dir) ? cache_dir : dot_cache_dir;
        std::cout << "  Found DD cache: " << actual_cache.string() << std::endl;
        std::string dd_posix = actual_cache.string();
        std::replace(dd_posix.begin(), dd_posix.end(), '\\', '/');

        // RyzenAI 1.8 DynamicDispatch / NPU models require the RyzenAI plugin
        // execution provider ("RyzenAILightExecutionProvider") to be registered on
        // the OrtEnv and appended to the session — this mirrors the working GenAI-SD
        // python reference (config_session_options in src/utils/common.py).
        // Calling RegisterCustomOpsLibrary WITHOUT appending the EP crashes inside
        // onnxruntime_providers_ryzenai.dll during graph partitioning (0xC0000005).
        static const char* k_ep_name = "RyzenAILightExecutionProvider";
        static const wchar_t* k_ops_dll = L"onnxruntime_providers_ryzenai.dll";
        fs::path ops_path;
#ifdef _WIN32
        wchar_t exe_buf[MAX_PATH] = {};
        if (GetModuleFileNameW(nullptr, exe_buf, MAX_PATH) > 0)
            ops_path = fs::path(exe_buf).parent_path() / k_ops_dll;
        if (!fs::exists(ops_path))
            ops_path = L"C:/Program Files/RyzenAI/1.8.0/deployment/onnxruntime_providers_ryzenai.dll";
        if (fs::exists(ops_path)) {
            std::wstring ops_dir = ops_path.parent_path().wstring();
            AddDllDirectory(ops_dir.c_str());

            // Register the EP plugin library on the Env exactly once per process.
            static bool ep_registered = false;
            if (!ep_registered) {
                std::cout << "  Registering RyzenAI EP: " << ops_path.string() << std::endl;
                // Pre-load with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR so dependencies
                // (vaiml.dll, dyn_dispatch_core.dll, ...) resolve from the ops dir.
                HMODULE hOps = LoadLibraryExW(
                    ops_path.wstring().c_str(), nullptr,
                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
                if (!hOps)
                    hOps = LoadLibraryExW(ops_path.wstring().c_str(), nullptr,
                                          LOAD_WITH_ALTERED_SEARCH_PATH);
                if (!hOps)
                    std::cerr << "  Warning: failed to pre-load RyzenAI EP DLL (error "
                              << GetLastError() << ")" << std::endl;
                // chdir to the DLL dir during registration (matches python register_ep).
                wchar_t prev_cwd[MAX_PATH] = {};
                GetCurrentDirectoryW(MAX_PATH, prev_cwd);
                SetCurrentDirectoryW(ops_dir.c_str());
                try {
                    env.RegisterExecutionProviderLibrary(k_ep_name, ops_path.wstring());
                    ep_registered = true;
                } catch (const std::exception& e) {
                    std::cerr << "  Warning: RegisterExecutionProviderLibrary failed: "
                              << e.what() << std::endl;
                }
                SetCurrentDirectoryW(prev_cwd);
            }

            // Append the RyzenAI EP to this session with DynamicDispatch options.
            try {
                std::vector<Ort::ConstEpDevice> ryzen_devices;
                for (const auto& dev : env.GetEpDevices()) {
                    const char* name = dev.EpName();
                    if (name && std::string(name) == k_ep_name)
                        ryzen_devices.push_back(dev);
                }
                if (!ryzen_devices.empty()) {
                    std::unordered_map<std::string, std::string> ep_options = {
                        {"dd_cache", dd_posix},
                        {"onnx_custom_ops_const_key", ""},
                        {"compile_fusion_rt", "0"},
                    };
                    if (has_external_data) {
                        // Pass external_data_file as a namespaced EP option
                        // (arrives as "ep.ryzenailightexecutionprovider.external_data_file").
                        // The unprefixed SessionOptions.AddConfigEntry form is deprecated
                        // and crashes (0xC0000005) during graph partitioning for large
                        // external-data models such as the SD3 transformer.
                        ep_options["external_data_file"] = data_path;
                    }
                    std::cout << "  Appending RyzenAI EP (" << ryzen_devices.size()
                              << " device(s)) with dd_cache=" << dd_posix << std::endl;
                    model_options.AppendExecutionProvider_V2(env, ryzen_devices, ep_options);
                } else {
                    std::cerr << "  Warning: no RyzenAI EP devices found after registration"
                              << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "  Warning: AppendExecutionProvider_V2 failed: "
                          << e.what() << std::endl;
            }

            // Also register the custom ops library (com.ryzenai CPU fallback ops).
            model_options.RegisterCustomOpsLibrary(ops_path.wstring().c_str());
        }
#endif
    }

    // Create session
    auto start_session = std::chrono::high_resolution_clock::now();
#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env, wpath.c_str(), model_options);
#else
    session_ = std::make_unique<Ort::Session>(env, model_path.c_str(), model_options);
#endif
    auto end_session = std::chrono::high_resolution_clock::now();
    std::cout << "  [TIMING] Session creation: " 
              << std::chrono::duration<double, std::milli>(end_session - start_session).count() << " ms" << std::endl;

    // Get input names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_->GetInputCount();
    for (size_t i = 0; i < num_inputs; i++) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        input_names_.push_back(name.get());
    }

    // Get output names
    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.push_back(name.get());
    }

    std::cout << "Model loaded with " << num_inputs << " inputs, "
              << num_outputs << " outputs" << std::endl;
}

std::vector<int64_t> OnnxModel::get_input_shape(size_t index) const {
    auto type_info = session_->GetInputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

ONNXTensorElementDataType OnnxModel::get_input_type(size_t index) const {
    auto type_info = session_->GetInputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetElementType();
}

std::vector<int64_t> OnnxModel::get_output_shape(size_t index) const {
    auto type_info = session_->GetOutputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

std::vector<Ort::Value> OnnxModel::run(
    const std::vector<Ort::Value>& inputs,
    const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names) {

    for (int attempt = 1; attempt <= kMaxNpuRetries; attempt++) {
        std::vector<Ort::Value> outputs;
        try {
            outputs = session_->Run(
                Ort::RunOptions{nullptr},
                input_names.data(),
                inputs.data(),
                inputs.size(),
                output_names.data(),
                output_names.size()
            );
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error during inference (" << model_path_ << "): "
                      << e.what() << std::endl;
            if (is_transient_npu_error(e) && attempt < kMaxNpuRetries) {
                std::cerr << "  [WARN] Transient RyzenAI NPU EP fault -- retrying inference "
                          << attempt << "/" << kMaxNpuRetries << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            throw;
        } catch (const std::exception& e) {
            std::cerr << "Error during inference (" << model_path_ << "): " << e.what() << std::endl;
            throw;
        }

        if (!output_has_nan(outputs)) {
            return outputs;
        }

        if (attempt < kMaxNpuRetries) {
            std::cerr << "  [WARN] " << model_path_ << ": ONNX Runtime output contained NaN "
                      << "(likely a transient RyzenAI NPU EP fault) -- retrying inference "
                      << attempt << "/" << kMaxNpuRetries << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }

        throw std::runtime_error(
            "ONNX Runtime output contained NaN after " + std::to_string(kMaxNpuRetries) +
            " attempts (model: " + model_path_ + ") -- likely an unrecoverable RyzenAI NPU EP fault");
    }

    throw std::runtime_error("ONNX Runtime inference failed after retries (model: " + model_path_ + ")");
}

std::vector<Ort::Value> OnnxModel::run(const std::vector<Ort::Value>& inputs) {
    std::vector<const char*> in_names, out_names;
    for (const auto& n : input_names_)  in_names.push_back(n.c_str());
    for (const auto& n : output_names_) out_names.push_back(n.c_str());
    return run(inputs, in_names, out_names);
}

} // namespace sd_npu
