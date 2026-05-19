// onnx_model.cpp - ONNX Runtime model wrapper implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "onnx_model.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <chrono>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace sd_npu {

// ============================================================================
// Global custom ops path
// ============================================================================

static std::string g_custom_ops_path;

void set_custom_ops_path(const std::string& path) {
    g_custom_ops_path = path;
}

const std::string& get_custom_ops_path() {
    return g_custom_ops_path;
}

// ============================================================================
// OnnxModel Implementation
// ============================================================================

OnnxModel::OnnxModel(const std::string& model_path,
                     Ort::SessionOptions& /*base_options*/,
                     Ort::Env& env) {

    std::cout << "Loading ONNX model: " << model_path << std::endl;

    // Create fresh session options for this model
    Ort::SessionOptions model_options;
    model_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    // Check for external data file (.onnx.data)
    std::string data_path = model_path + ".data";
    if (fs::exists(data_path)) {
        std::cout << "  Found external data: " << data_path << std::endl;
        model_options.AddConfigEntry("external_data_file", data_path.c_str());
    }

    // Check for DD cache directory - MUST be set BEFORE registering custom ops
    fs::path model_dir = fs::path(model_path).parent_path();
    fs::path cache_dir = model_dir / "cache";
    fs::path dot_cache_dir = model_dir / ".cache";

    bool has_cache = fs::exists(cache_dir) || fs::exists(dot_cache_dir);
    if (has_cache) {
        std::cout << "  Found DD cache: " << model_dir.string() << std::endl;
        std::string dd_posix = model_dir.string();
        std::replace(dd_posix.begin(), dd_posix.end(), '\\', '/');
        model_options.AddConfigEntry("dd_cache", dd_posix.c_str());
        model_options.AddConfigEntry("onnx_custom_ops_const_key", model_path.c_str());
    }

    // Register custom ops library only for NPU models (those with DD cache).
    // CPU-only models (text encoders) don't need custom ops, and loading the
    // VitisAI EP for them adds tens of seconds of unnecessary overhead.
    if (has_cache && !g_custom_ops_path.empty() && fs::exists(g_custom_ops_path)) {
        std::cout << "  Registering custom ops: " << g_custom_ops_path << std::endl;
        auto start_custom_ops = std::chrono::high_resolution_clock::now();
#ifdef _WIN32
        // Add the custom ops directory to DLL search path so dependencies are found
        std::wstring wide_ops_path(g_custom_ops_path.begin(), g_custom_ops_path.end());
        auto ops_dir = fs::path(g_custom_ops_path).parent_path().wstring();
        SetDllDirectoryW(ops_dir.c_str());
        AddDllDirectory(ops_dir.c_str());
        HMODULE hMod = LoadLibraryW(wide_ops_path.c_str());
        if (!hMod) {
            // Try LoadLibraryExW with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
            hMod = LoadLibraryExW(wide_ops_path.c_str(), nullptr,
                                  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        }
        if (!hMod) {
            std::cerr << "  Warning: LoadLibraryW failed for custom ops (error "
                      << GetLastError() << ")" << std::endl;
        }
        model_options.RegisterCustomOpsLibrary(wide_ops_path.c_str());
        SetDllDirectoryW(nullptr);  // restore default search order
#else
        std::wstring wide_path(g_custom_ops_path.begin(), g_custom_ops_path.end());
        model_options.RegisterCustomOpsLibrary(wide_path.c_str());
#endif
        auto end_custom_ops = std::chrono::high_resolution_clock::now();
        std::cout << "  [TIMING] Custom ops registration: " 
                  << std::chrono::duration<double, std::milli>(end_custom_ops - start_custom_ops).count() << " ms" << std::endl;
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

    try {
        return session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            inputs.data(),
            inputs.size(),
            output_names.data(),
            output_names.size()
        );
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error during inference: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        throw;
    }
}

std::vector<Ort::Value> OnnxModel::run(const std::vector<Ort::Value>& inputs) {
    std::vector<const char*> in_names, out_names;
    for (const auto& n : input_names_)  in_names.push_back(n.c_str());
    for (const auto& n : output_names_) out_names.push_back(n.c_str());
    return run(inputs, in_names, out_names);
}

} // namespace sd_npu
