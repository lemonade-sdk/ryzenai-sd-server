// http_server.h - HTTP server for SD inference with OpenAI-compatible API
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "sd_pipeline.h"
#include <string>
#include <memory>
#include <mutex>
#include <thread>

namespace sd_npu {

// HTTP server with OpenAI-compatible image generation API
class SDServer {
public:
    SDServer(int port, const SDConfig& model_config, const std::string& model_path = "");
    ~SDServer();
    
    void start();    // Start server in background thread
    void stop();     // Stop server
    void run();      // Run server (blocking)
    
    // Load a new model (thread-safe). Returns empty string on success, error message on failure.
    std::string load_model(const std::string& model_path);

    // Get the currently loaded model path
    std::string current_model_path() const { return current_model_path_; }

private:
    int port_;
    SDConfig config_;
    std::unique_ptr<SDPipeline> pipeline_;
    std::mutex pipeline_mutex_;  // protects pipeline_ during model switches
    std::string current_model_path_;
    bool running_ = false;
    std::unique_ptr<std::thread> server_thread_;
};

} // namespace sd_npu
