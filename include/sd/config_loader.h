// config_loader.h - Configuration file loader
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <string>
#include <map>
#include <optional>

namespace sd_npu {

struct ServerConfig {
    // Paths
    std::string onnxruntime_root;
    std::string custom_ops_dll;
    std::string dd_root;
    std::string dd_plugins_root;
    std::string opencv_root;
    std::string lib_directory;  // Additional lib directory to copy
    
    // Models
    std::string default_model_path;
    std::string models_directory;
    
    // Server
    int default_port = 8080;
    std::string host = "0.0.0.0";
    
    // Generation defaults
    std::string default_prompt;
    std::string default_negative_prompt;
    int default_width = 512;
    int default_height = 512;
    int default_steps = 20;
    float default_guidance_scale = 7.0f;
    std::string default_scheduler = "euler";
};

class ConfigLoader {
public:
    // Load configuration from JSON file
    static std::optional<ServerConfig> load(const std::string& config_path);
    
    // Save configuration to JSON file
    static bool save(const std::string& config_path, const ServerConfig& config);
    
    // Get default config file path (looks in current dir, exe dir, etc.)
    static std::string get_default_config_path();
    
private:
    static std::string trim(const std::string& str);
    static std::map<std::string, std::string> parse_simple_json(const std::string& content);
};

} // namespace sd_npu
