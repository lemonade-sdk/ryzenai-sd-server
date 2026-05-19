// config_loader.cpp - Configuration file loader implementation
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "config_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace sd_npu {

std::string ConfigLoader::trim(const std::string& str) {
    auto start = str.find_first_not_of(" \t\r\n\"");
    auto end = str.find_last_not_of(" \t\r\n\",");
    if (start == std::string::npos) return "";
    return str.substr(start, end - start + 1);
}

std::optional<ServerConfig> ConfigLoader::load(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "[CONFIG] Could not open config file: " << config_path << std::endl;
        return std::nullopt;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    ServerConfig config;
    
    // Simple JSON parser (manual for no dependencies)
    // Format: "key": "value"
    auto find_value = [&](const std::string& key) -> std::string {
        std::string search = "\"" + key + "\"";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return "";
        
        pos = content.find(":", pos);
        if (pos == std::string::npos) return "";
        
        size_t start = content.find_first_not_of(" \t\r\n", pos + 1);
        if (start == std::string::npos) return "";
        
        size_t end;
        if (content[start] == '"') {
            // String value
            start++;
            end = content.find('"', start);
        } else {
            // Number or boolean
            end = content.find_first_of(",\n}", start);
        }
        
        if (end == std::string::npos) return "";
        return trim(content.substr(start, end - start));
    };
    
    auto find_int = [&](const std::string& key, int default_val) -> int {
        std::string val = find_value(key);
        if (val.empty()) return default_val;
        try {
            return std::stoi(val);
        } catch (...) {
            return default_val;
        }
    };
    
    auto find_float = [&](const std::string& key, float default_val) -> float {
        std::string val = find_value(key);
        if (val.empty()) return default_val;
        try {
            return std::stof(val);
        } catch (...) {
            return default_val;
        }
    };
    
    // Parse paths
    config.onnxruntime_root = find_value("onnxruntime_root");
    config.custom_ops_dll = find_value("custom_ops_dll");
    config.dd_root = find_value("dd_root");
    config.dd_plugins_root = find_value("dd_plugins_root");
    config.opencv_root = find_value("opencv_root");
    config.lib_directory = find_value("lib_directory");
    
    // Parse models
    config.default_model_path = find_value("default_model_path");
    config.models_directory = find_value("models_directory");
    
    // Parse server
    config.default_port = find_int("default_port", 8080);
    config.host = find_value("host");
    if (config.host.empty()) config.host = "0.0.0.0";
    
    // Parse generation defaults
    config.default_prompt = find_value("default_prompt");
    config.default_negative_prompt = find_value("default_negative_prompt");
    config.default_width = find_int("default_width", 512);
    config.default_height = find_int("default_height", 512);
    config.default_steps = find_int("default_steps", 0);
    config.default_guidance_scale = find_float("default_guidance_scale", 7.0f);
    config.default_scheduler = find_value("default_scheduler");
    if (config.default_scheduler.empty()) config.default_scheduler = "euler";
    
    std::cout << "[CONFIG] Loaded configuration from: " << config_path << std::endl;
    return config;
}

bool ConfigLoader::save(const std::string& config_path, const ServerConfig& config) {
    std::ofstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "[CONFIG] Could not write config file: " << config_path << std::endl;
        return false;
    }
    
    file << "{\n";
    file << "  \"paths\": {\n";
    file << "    \"onnxruntime_root\": \"" << config.onnxruntime_root << "\",\n";
    file << "    \"custom_ops_dll\": \"" << config.custom_ops_dll << "\",\n";
    file << "    \"dd_root\": \"" << config.dd_root << "\",\n";
    file << "    \"dd_plugins_root\": \"" << config.dd_plugins_root << "\",\n";
    file << "    \"opencv_root\": \"" << config.opencv_root << "\",\n";
    file << "    \"lib_directory\": \"" << config.lib_directory << "\"\n";
    file << "  },\n";
    file << "  \"models\": {\n";
    file << "    \"default_model_path\": \"" << config.default_model_path << "\",\n";
    file << "    \"models_directory\": \"" << config.models_directory << "\"\n";
    file << "  },\n";
    file << "  \"server\": {\n";
    file << "    \"default_port\": " << config.default_port << ",\n";
    file << "    \"host\": \"" << config.host << "\"\n";
    file << "  },\n";
    file << "  \"generation\": {\n";
    file << "    \"default_prompt\": \"" << config.default_prompt << "\",\n";
    file << "    \"default_negative_prompt\": \"" << config.default_negative_prompt << "\",\n";
    file << "    \"default_width\": " << config.default_width << ",\n";
    file << "    \"default_height\": " << config.default_height << ",\n";
    file << "    \"default_steps\": " << config.default_steps << ",\n";
    file << "    \"default_guidance_scale\": " << config.default_guidance_scale << ",\n";
    file << "    \"default_scheduler\": \"" << config.default_scheduler << "\"\n";
    file << "  }\n";
    file << "}\n";
    
    std::cout << "[CONFIG] Saved configuration to: " << config_path << std::endl;
    return true;
}

std::string ConfigLoader::get_default_config_path() {
    // Check current directory
    if (fs::exists("config.json")) {
        return "config.json";
    }
    
    // Check executable directory
#ifdef _WIN32
    char exe_path[MAX_PATH];
    GetModuleFileNameA(NULL, exe_path, MAX_PATH);
    fs::path exe_dir = fs::path(exe_path).parent_path();
    fs::path config_in_exe_dir = exe_dir / "config.json";
    if (fs::exists(config_in_exe_dir)) {
        return config_in_exe_dir.string();
    }
#endif
    
    // Check parent directory (for build/Release structure)
    if (fs::exists("../config.json")) {
        return "../config.json";
    }
    
    if (fs::exists("../../config.json")) {
        return "../../config.json";
    }
    
    // Default to current directory
    return "config.json";
}

} // namespace sd_npu
