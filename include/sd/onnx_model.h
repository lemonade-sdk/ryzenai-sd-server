// onnx_model.h - ONNX Runtime model wrapper with DD/NPU support
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

namespace sd_npu {

// Global custom ops path (set once by SDPipeline, used by all OnnxModel instances)
void set_custom_ops_path(const std::string& path);
const std::string& get_custom_ops_path();

// Single ONNX model wrapper with DD cache and custom ops support
class OnnxModel {
public:
    OnnxModel(const std::string& model_path,
              Ort::SessionOptions& options,
              Ort::Env& env);

    // Run inference with explicit input/output names
    std::vector<Ort::Value> run(
        const std::vector<Ort::Value>& inputs,
        const std::vector<const char*>& input_names,
        const std::vector<const char*>& output_names
    );

    // Run inference using all registered input/output names
    std::vector<Ort::Value> run(const std::vector<Ort::Value>& inputs);

    const std::vector<std::string>& get_input_names() const { return input_names_; }
    const std::vector<std::string>& get_output_names() const { return output_names_; }

    // Get input shape for a given input index (dynamic dims returned as -1)
    std::vector<int64_t> get_input_shape(size_t index) const;

    // Get input element type
    ONNXTensorElementDataType get_input_type(size_t index) const;

    // Get number of outputs
    size_t get_num_outputs() const { return output_names_.size(); }

    // Get output shape for a given output index (dynamic dims returned as -1)
    std::vector<int64_t> get_output_shape(size_t index) const;

private:
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

} // namespace sd_npu
