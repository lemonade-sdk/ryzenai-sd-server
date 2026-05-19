// http_server.cpp - HTTP server implementation with OpenAI-compatible API
// Copyright (C) 2025 Advanced Micro Devices, Inc.

#include "http_server.h"
#include "variant_registry.h"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <regex>
#include <random>

// cpp-httplib - single-header HTTP server library
// Download from: https://github.com/yhirose/cpp-httplib/blob/master/httplib.h
// Place httplib.h in include/ directory
#ifdef __has_include
#  if __has_include(<httplib.h>)
#    include <httplib.h>
#    define HAS_HTTPLIB
#  endif
#endif

#ifndef HAS_HTTPLIB
#  ifdef _MSC_VER
#    pragma message("httplib.h not found - HTTP server functionality will be disabled")
#    pragma message("Download from: https://github.com/yhirose/cpp-httplib")
#  else
#    warning "httplib.h not found - HTTP server functionality will be disabled"
#    warning "Download from: https://github.com/yhirose/cpp-httplib"
#  endif
#endif

// Base64 decoding and encoding helpers
namespace {
    // Base64 decode lookup table
    static const unsigned char base64_decode_table[256] = {
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,62,64,64,64,63,52,53,54,55,56,57,58,59,60,61,64,64,64,64,64,64,
        64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,64,64,64,64,64,
        64,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64
    };

    std::vector<uint8_t> base64_decode(const std::string& encoded) {
        std::vector<uint8_t> result;
        if (encoded.empty()) return result;

        size_t in_len = encoded.size();
        size_t i = 0;
        size_t j = 0;
        unsigned char char_array_4[4], char_array_3[3];

        result.reserve((in_len / 4) * 3);

        while (in_len-- && encoded[i] != '=') {
            if (base64_decode_table[static_cast<unsigned char>(encoded[i])] == 64) {
                i++;
                continue;
            }

            char_array_4[j++] = encoded[i++];
            if (j == 4) {
                for (j = 0; j < 4; j++)
                    char_array_4[j] = base64_decode_table[static_cast<unsigned char>(char_array_4[j])];

                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

                for (j = 0; j < 3; j++)
                    result.push_back(char_array_3[j]);
                j = 0;
            }
        }

        if (j) {
            for (size_t k = j; k < 4; k++)
                char_array_4[k] = 0;

            for (size_t k = 0; k < 4; k++)
                char_array_4[k] = base64_decode_table[static_cast<unsigned char>(char_array_4[k])];

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (size_t k = 0; k < j - 1; k++)
                result.push_back(char_array_3[k]);
        }

        return result;
    }

    // Escape a string for safe embedding in a JSON string value
    std::string json_escape(const std::string& s) {
        std::string result;
        result.reserve(s.size() + 16);
        for (char c : s) {
            switch (c) {
                case '\\': result += "\\\\"; break;
                case '"':  result += "\\\""; break;
                case '\n': result += "\\n";  break;
                case '\r': result += "\\r";  break;
                case '\t': result += "\\t";  break;
                default:   result += c;      break;
            }
        }
        return result;
    }

    // Parse base64-encoded raw RGB image data
    // Expected format: base64(RGB bytes) with separate width/height parameters
    // Or: raw binary RGB data (no decoding needed)
    struct DecodedImage {
        std::vector<uint8_t> rgb_data;
        int width = 0;
        int height = 0;
        bool valid = false;
    };

    // For backwards compatibility: try to decode as base64 first, fallback to raw
    DecodedImage decode_image_data(const std::vector<uint8_t>& image_data, int expected_w, int expected_h) {
        DecodedImage result;
        
        // If it looks like base64 (printable ASCII), decode it
        bool is_base64 = !image_data.empty() && 
                        std::all_of(image_data.begin(), image_data.begin() + std::min((size_t)100, image_data.size()),
                                   [](unsigned char c) { return std::isprint(c) || c == '\n' || c == '\r'; });
        
        if (is_base64) {
            std::string b64_str(image_data.begin(), image_data.end());
            result.rgb_data = base64_decode(b64_str);
        } else {
            // Already raw RGB
            result.rgb_data.assign(image_data.begin(), image_data.end());
        }
        
        // Validate size matches expected dimensions
        size_t expected_size = static_cast<size_t>(expected_w * expected_h * 3);
        if (result.rgb_data.size() == expected_size) {
            result.width = expected_w;
            result.height = expected_h;
            result.valid = true;
        } else {
            std::cerr << "Warning: Image data size mismatch. Expected " << expected_size 
                     << " bytes for " << expected_w << "x" << expected_h 
                     << " but got " << result.rgb_data.size() << " bytes" << std::endl;
        }
        
        return result;
    }

    // Parse "size" field (e.g., "1024x1024") into width and height
    std::pair<int, int> parse_size(const std::string& size_str, int default_w, int default_h) {
        auto x_pos = size_str.find('x');
        if (x_pos != std::string::npos) {
            try {
                int w = std::stoi(size_str.substr(0, x_pos));
                int h = std::stoi(size_str.substr(x_pos + 1));
                return {w, h};
            } catch (...) {}
        }
        return {default_w, default_h};
    }

    // Parse sd_cpp_extra_args embedded in prompt by Lemonade
    // Format: <sd_cpp_extra_args>{"steps":20,"cfg_scale":7.0,"seed":42}</sd_cpp_extra_args>
    struct ExtraArgs {
        int steps = -1;           // -1 means use default
        float cfg_scale = -1.0f;  // -1 means use default
        int seed = -1;            // -1 means use default
        std::string clean_prompt; // prompt with extra_args tag removed
    };

    ExtraArgs parse_extra_args(const std::string& prompt) {
        ExtraArgs args;
        args.clean_prompt = prompt;

        // Look for <sd_cpp_extra_args>JSON</sd_cpp_extra_args>
        std::regex tag_regex(R"(<sd_cpp_extra_args>(.*?)</sd_cpp_extra_args>)");
        std::smatch match;

        if (std::regex_search(prompt, match, tag_regex)) {
            std::string json_str = match[1].str();

            // Remove the tag from prompt
            args.clean_prompt = std::regex_replace(prompt, tag_regex, "");
            // Trim trailing whitespace
            while (!args.clean_prompt.empty() && std::isspace(args.clean_prompt.back())) {
                args.clean_prompt.pop_back();
            }

            // Simple JSON parsing for steps, cfg_scale, seed
            // Parse "steps":N
            std::regex steps_regex(R"("steps"\s*:\s*(\d+))");
            if (std::regex_search(json_str, match, steps_regex)) {
                args.steps = std::stoi(match[1].str());
            }

            // Parse "cfg_scale":N.N
            std::regex cfg_regex(R"("cfg_scale"\s*:\s*([\d.]+))");
            if (std::regex_search(json_str, match, cfg_regex)) {
                args.cfg_scale = std::stof(match[1].str());
            }

            // Parse "seed":N
            std::regex seed_regex(R"("seed"\s*:\s*(-?\d+))");
            if (std::regex_search(json_str, match, seed_regex)) {
                args.seed = std::stoi(match[1].str());
            }
        }

        return args;
    }
}

namespace sd_npu {

SDServer::SDServer(int port, const SDConfig& model_config, const std::string& model_path) 
    : port_(port), config_(model_config) {
    
    std::cout << "Initializing SD Server on port " << port_ << std::endl;

    // Store normalized model path so load_model() can skip reload for same model
    if (!model_path.empty()) {
        current_model_path_ = std::filesystem::path(model_path).lexically_normal().string();
    }
    
    // Create pipeline
    auto start = std::chrono::high_resolution_clock::now();
    pipeline_ = std::make_unique<SDPipeline>(model_config);
    auto end = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Pipeline loaded in " << load_ms << " ms" << std::endl;
}

SDServer::~SDServer() {
    stop();
}

void SDServer::start() {
    if (running_) {
        return;
    }
    
    running_ = true;
    server_thread_ = std::make_unique<std::thread>([this]() {
        this->run();
    });
}

void SDServer::stop() {
    if (running_) {
        std::cout << "Stopping server..." << std::endl;
        running_ = false;
        if (server_thread_ && server_thread_->joinable()) {
            server_thread_->join();
        }
    }
}

std::string SDServer::load_model(const std::string& model_path) {
    namespace fs = std::filesystem;

    // Normalize path for consistent comparison (forward vs backslash, trailing slashes, etc.)
    std::string normalized_path = fs::path(model_path).lexically_normal().string();

    if (!fs::exists(normalized_path)) {
        return "Model path does not exist: " + model_path;
    }

    // If same model is already loaded, skip
    if (normalized_path == current_model_path_ && pipeline_) {
        std::cout << "[LOAD] Model already loaded: " << normalized_path << std::endl;
        return "";
    }

    std::cout << "[LOAD] Loading model from: " << normalized_path << std::endl;

    try {
        // Detect variant using registry
        auto& registry = VariantRegistry::instance();
        const VariantDescriptor* desc = registry.detect(fs::path(normalized_path));

        if (!desc) {
            return "Could not auto-detect variant for: " + normalized_path;
        }
        ModelVariant variant = desc->variant;

        // Build new config, preserving execution-provider settings from current config
        SDConfig new_config;
        new_config.variant = variant;
        new_config.custom_op_path = config_.custom_op_path;
        new_config.dd_cache_path = config_.dd_cache_path;
        new_config.force_cpu = config_.force_cpu;
        new_config.gpu = config_.gpu;
        new_config.enable_compile = config_.enable_compile;
        new_config.verbose = config_.verbose;

        // Set variant-specific defaults from registry
        new_config.width = desc->default_width;
        new_config.height = desc->default_height;
        new_config.num_inference_steps = desc->default_steps;
        new_config.guidance_scale = desc->default_guidance;

        // Discover ONNX components using registry
        registry.discover_components(fs::path(normalized_path), *desc, new_config);

        if (new_config.component_paths.empty()) {
            return "No ONNX model components found in: " + normalized_path;
        }

        // Create new pipeline (this is the slow part - model loading)
        auto start = std::chrono::high_resolution_clock::now();
        auto new_pipeline = std::make_unique<SDPipeline>(new_config);
        auto end = std::chrono::high_resolution_clock::now();
        auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Swap pipeline under lock
        {
            std::lock_guard<std::mutex> lock(pipeline_mutex_);
            pipeline_ = std::move(new_pipeline);
            config_ = new_config;
            current_model_path_ = normalized_path;
        }

        std::cout << "[LOAD] Model loaded in " << load_ms << " ms: "
                  << variant_to_string(variant) << " from " << normalized_path << std::endl;
        return "";

    } catch (const std::exception& e) {
        return std::string("Failed to load model: ") + e.what();
    }
}

void SDServer::run() {
#ifdef HAS_HTTPLIB
    std::cout << "Starting SD Server on port " << port_ << "..." << std::endl;
    
    httplib::Server svr;
    
    // Enable CORS
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization"}
    });
    
    // Handle CORS preflight for all paths
    svr.Options(".*", [](const httplib::Request& /*req*/, httplib::Response& res) {
        res.status = 204;
    });
    
    // Health check endpoint
    svr.Get("/health", [this](const httplib::Request& /*req*/, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        std::ostringstream resp;
        resp << R"({"status":"ok","model_path":")" << json_escape(current_model_path_) << R"("})";
        res.set_content(resp.str(), "application/json");
    });
    
    svr.Get("/", [](const httplib::Request& /*req*/, httplib::Response& res) {
        res.set_content(R"({"status":"ok","service":"sd-npu-server"})", "application/json");
    });

    // Model info endpoint - returns currently loaded model path
    svr.Get("/v1/internal/model_info", [this](const httplib::Request& /*req*/, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        std::ostringstream resp;
        resp << R"({"model_path":")" << json_escape(current_model_path_)
             << R"(","variant":")" << variant_to_string(config_.variant)
             << R"(","width":)" << config_.width
             << R"(,"height":)" << config_.height
             << R"(,"steps":)" << config_.num_inference_steps
             << R"(,"guidance_scale":)" << config_.guidance_scale
             << R"(})";
        res.set_content(resp.str(), "application/json");
    });

    // Model load endpoint - switch to a different model
    svr.Post("/v1/internal/load", [this](const httplib::Request& req, httplib::Response& res) {
        // Parse model_path from request body JSON
        std::string model_path;
        auto path_pos = req.body.find("\"model_path\"");
        if (path_pos != std::string::npos) {
            auto colon_pos = req.body.find(":", path_pos);
            if (colon_pos != std::string::npos) {
                auto q1 = req.body.find("\"", colon_pos + 1);
                if (q1 != std::string::npos) {
                    // Find closing quote, skipping escaped characters
                    size_t q2 = q1 + 1;
                    while (q2 < req.body.size()) {
                        if (req.body[q2] == '\\') { q2 += 2; continue; }
                        if (req.body[q2] == '"') break;
                        q2++;
                    }
                    if (q2 < req.body.size()) {
                        std::string raw = req.body.substr(q1 + 1, q2 - q1 - 1);
                        // Unescape JSON string (\/ -> /, \\ -> \, etc.)
                        model_path.reserve(raw.size());
                        for (size_t k = 0; k < raw.size(); k++) {
                            if (raw[k] == '\\' && k + 1 < raw.size()) {
                                model_path += raw[k + 1];
                                k++;
                            } else {
                                model_path += raw[k];
                            }
                        }
                    }
                }
            }
        }

        if (model_path.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"model_path is required","type":"invalid_request_error"}})", "application/json");
            return;
        }

        std::string error = load_model(model_path);
        if (!error.empty()) {
            res.status = 500;
            std::ostringstream err;
            err << R"({"error":{"message":")" << error << R"(","type":"server_error"}})";
            res.set_content(err.str(), "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        std::ostringstream resp;
        resp << R"({"status":"ok","model_path":")" << json_escape(current_model_path_)
             << R"(","variant":")" << variant_to_string(config_.variant)
             << R"("})";
        res.set_content(resp.str(), "application/json");
    });
    
    // OpenAI-compatible image generation endpoint
    svr.Post("/v1/images/generations", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            // Parse JSON request
            auto json_str = req.body;

            // Simple JSON parsing (in production, use nlohmann/json)
            std::string prompt;
            std::string negative_prompt = "";

            // Extract prompt
            // Look for "prompt":"..." or "prompt": "..."
            auto prompt_pos = json_str.find("\"prompt\"");
            if (prompt_pos != std::string::npos) {
                // Find the colon after "prompt"
                auto colon_pos = json_str.find(":", prompt_pos);
                if (colon_pos != std::string::npos) {
                    // Find the opening quote of the value
                    auto quote1 = json_str.find("\"", colon_pos + 1);
                    if (quote1 != std::string::npos) {
                        // Find the closing quote, properly skipping escaped quotes
                        auto quote2 = quote1 + 1;
                        while (quote2 < json_str.length()) {
                            if (json_str[quote2] == '\\') {
                                quote2 += 2; // skip escaped char entirely
                                continue;
                            }
                            if (json_str[quote2] == '\"') break;
                            quote2++;
                        }
                        if (quote2 < json_str.length()) {
                            // Extract and unescape the JSON string value
                            std::string raw = json_str.substr(quote1 + 1, quote2 - quote1 - 1);
                            // Unescape \" -> " and \\ -> backslash
                            prompt.reserve(raw.size());
                            for (size_t i = 0; i < raw.size(); i++) {
                                if (raw[i] == '\\' && i + 1 < raw.size()) {
                                    char next = raw[i + 1];
                                    if (next == '"' || next == '\\' || next == '/') {
                                        prompt += next; i++; continue;
                                    }
                                    if (next == 'n') { prompt += '\n'; i++; continue; }
                                    if (next == 't') { prompt += '\t'; i++; continue; }
                                }
                                prompt += raw[i];
                            }
                        }
                    }
                }
            }

            if (prompt.empty()) {
                prompt = "An astronaut riding a green horse";
                std::cout << "[WARNING] No prompt found in request, using default" << std::endl;
            }

            // Parse size field from JSON (e.g., "size":"1024x768")
            std::string size_str;
            auto size_pos = json_str.find("\"size\"");
            if (size_pos != std::string::npos) {
                auto s_colon = json_str.find(":", size_pos);
                if (s_colon != std::string::npos) {
                    auto sq1 = json_str.find("\"", s_colon + 1);
                    if (sq1 != std::string::npos) {
                        auto sq2 = json_str.find("\"", sq1 + 1);
                        if (sq2 != std::string::npos) {
                            size_str = json_str.substr(sq1 + 1, sq2 - sq1 - 1);
                        }
                    }
                }
            }
            auto [req_width, req_height] = parse_size(size_str, config_.width, config_.height);

            // Parse extra args from prompt (Lemonade embeds steps, cfg_scale, seed)
            auto extra = parse_extra_args(prompt);
            std::string clean_prompt = extra.clean_prompt;

            // Apply extra args to a local copy of config for this request
            SDConfig req_config = config_;
            req_config.width = req_width;
            req_config.height = req_height;
            if (extra.steps > 0) {
                req_config.num_inference_steps = extra.steps;
            }
            if (extra.cfg_scale >= 0.0f) {
                req_config.guidance_scale = extra.cfg_scale;
            }
            if (extra.seed >= 0) {
                req_config.seed = extra.seed;
            } else {
                // seed=-1 or not provided: use random seed from hardware random device
                std::random_device rd;
                req_config.seed = (int)(rd() % 2147483647);
            }

            std::cout << "[REQUEST] Generating image:" << std::endl;
            std::cout << "  prompt: \"" << clean_prompt << "\"" << std::endl;
            std::cout << "  size: " << req_config.width << "x" << req_config.height << std::endl;
            std::cout << "  steps: " << req_config.num_inference_steps
                      << ", cfg_scale: " << req_config.guidance_scale
                      << ", seed: " << req_config.seed << std::endl;
            std::cout.flush();  // Force output immediately

            auto start = std::chrono::high_resolution_clock::now();

            // Update pipeline config and generate under lock
            ImageResponse response;
            {
                std::lock_guard<std::mutex> lock(pipeline_mutex_);
                pipeline_->update_config(req_config);
                response = pipeline_->generate(clean_prompt, negative_prompt);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            if (!response.success) {
                std::ostringstream err;
                err << R"({"error":{"message":")" << response.error << R"("}})";
                res.status = 500;
                res.set_content(err.str(), "application/json");
                return;
            }

            std::cout << "[RESPONSE] Generated in " << gen_ms << " ms ("
                      << response.width << "x" << response.height << ", "
                      << "base64 length: " << (response.data.empty() ? 0 : response.data[0].b64_json.length())
                      << " chars)" << std::endl;

            // Build OpenAI-compatible response with timing info
            std::ostringstream resp;
            resp << R"({"created":)" << response.created
                 << R"(,"width":)"  << response.width
                 << R"(,"height":)" << response.height
                 << R"(,"format":"raw_rgb")"
                 << R"(,"data":[)";
            for (size_t i = 0; i < response.data.size(); i++) {
                if (i > 0) resp << ",";
                resp << R"({"b64_json":")" << response.data[i].b64_json << R"("})";
            }
            resp << R"(],"timing":{"total_ms":)" << response.generation_time_ms
                 << R"(,"denoise_ms":)" << response.denoising_time_ms
                 << R"(,"text_encode_ms":)" << response.text_encoding_time_ms
                 << R"(,"vae_decode_ms":)" << response.vae_decoding_time_ms
                 << R"(,"steps":)" << response.num_inference_steps
                 << R"(}})";

            res.set_content(resp.str(), "application/json");

        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Generation failed: " << e.what() << std::endl;

            std::string error_response = R"({
                "error": {
                    "message": ")" + std::string(e.what()) + R"(",
                    "type": "server_error"
                }
            })";

            res.status = 500;
            res.set_content(error_response, "application/json");
        }
    });

    // OpenAI-compatible image editing endpoint (multipart/form-data)
    svr.Post("/v1/images/edits", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::cout << "[REQUEST] Image edit request received" << std::endl;

            // Parse multipart form data
            std::string prompt;
            std::string size_str;
            int n = 1;
            std::vector<uint8_t> input_image_data;
            std::vector<uint8_t> mask_data;

            // httplib provides parsed multipart form data
            if (req.form.has_field("prompt")) {
                prompt = req.form.get_field("prompt");
            }
            if (req.form.has_field("n")) {
                n = std::stoi(req.form.get_field("n"));
            }
            if (req.form.has_field("size")) {
                size_str = req.form.get_field("size");
            }
            if (req.form.has_file("image[]")) {
                auto file = req.form.get_file("image[]");
                input_image_data.assign(file.content.begin(), file.content.end());
            }
            if (req.form.has_file("mask")) {
                auto file = req.form.get_file("mask");
                mask_data.assign(file.content.begin(), file.content.end());
            }

            if (prompt.empty()) {
                res.status = 400;
                res.set_content(R"({"error":{"message":"prompt is required","type":"invalid_request_error"}})", "application/json");
                return;
            }

            // Parse extra args from prompt (Lemonade embeds steps, cfg_scale, seed)
            auto extra = parse_extra_args(prompt);
            std::string clean_prompt = extra.clean_prompt;

            // Parse size
            auto [req_width, req_height] = parse_size(size_str, config_.width, config_.height);

            // Apply extra args to a local copy of config for this request
            SDConfig req_config = config_;
            req_config.width = req_width;
            req_config.height = req_height;
            if (extra.steps > 0) {
                req_config.num_inference_steps = extra.steps;
            }
            if (extra.cfg_scale >= 0.0f) {
                req_config.guidance_scale = extra.cfg_scale;
            }
            if (extra.seed >= 0) {
                req_config.seed = extra.seed;
            } else {
                req_config.seed = (int)(std::chrono::steady_clock::now().time_since_epoch().count() % 2147483647);
            }

            std::cout << "  prompt: \"" << clean_prompt << "\"" << std::endl;
            std::cout << "  size: " << req_config.width << "x" << req_config.height << std::endl;
            std::cout << "  steps: " << req_config.num_inference_steps
                      << ", cfg_scale: " << req_config.guidance_scale
                      << ", seed: " << req_config.seed << std::endl;

            // Decode input image if provided (expects base64-encoded raw RGB)
            std::vector<uint8_t> control_rgb;
            if (!input_image_data.empty()) {
                auto decoded = decode_image_data(input_image_data, req_config.width, req_config.height);
                if (decoded.valid) {
                    control_rgb = std::move(decoded.rgb_data);
                    std::cout << "  input image: " << decoded.width << "x" << decoded.height << " (raw RGB)" << std::endl;
                } else {
                    std::cout << "  Warning: Failed to decode input image data" << std::endl;
                }
            }

            // Decode mask if provided (expects base64-encoded raw RGB)
            std::vector<uint8_t> mask_rgb;
            if (!mask_data.empty()) {
                auto decoded = decode_image_data(mask_data, req_config.width, req_config.height);
                if (decoded.valid) {
                    mask_rgb = std::move(decoded.rgb_data);
                    std::cout << "  mask: " << decoded.width << "x" << decoded.height << " (raw RGB)" << std::endl;
                } else {
                    std::cout << "  Warning: Failed to decode mask data" << std::endl;
                }
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Update pipeline config and generate under lock
            std::vector<ImageData> all_images;
            int resp_width = req_config.width, resp_height = req_config.height;
            {
                std::lock_guard<std::mutex> lock(pipeline_mutex_);
                pipeline_->update_config(req_config);

                // Generate all requested images
                for (int i = 0; i < n; i++) {
                    // Vary seed for each image
                    SDConfig img_config = req_config;
                    img_config.seed = req_config.seed + i;
                    pipeline_->update_config(img_config);

                    auto img_response = pipeline_->generate(clean_prompt, "", control_rgb);
                    if (img_response.success && !img_response.data.empty()) {
                        all_images.insert(all_images.end(), img_response.data.begin(), img_response.data.end());
                        resp_width  = img_response.width;
                        resp_height = img_response.height;
                    }
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "[RESPONSE] Generated " << n << " image(s) in " << gen_ms << " ms" << std::endl;

            // Build OpenAI-compatible response
            std::ostringstream response;
            response << R"({"created":)" << std::chrono::system_clock::now().time_since_epoch().count()
                     << R"(,"width":)"   << resp_width
                     << R"(,"height":)"  << resp_height
                     << R"(,"format":"raw_rgb")"
                     << R"(,"data":[)";
            for (size_t i = 0; i < all_images.size(); i++) {
                if (i > 0) response << ",";
                response << R"({"b64_json":")" << all_images[i].b64_json << R"("})";
            }
            response << "]}";

            res.set_content(response.str(), "application/json");

        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Image edit failed: " << e.what() << std::endl;

            std::string error_response = R"({
                "error": {
                    "message": ")" + std::string(e.what()) + R"(",
                    "type": "server_error"
                }
            })";

            res.status = 500;
            res.set_content(error_response, "application/json");
        }
    });

    // OpenAI-compatible image variations endpoint (multipart/form-data)
    svr.Post("/v1/images/variations", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::cout << "[REQUEST] Image variation request received" << std::endl;

            // Parse multipart form data
            std::string prompt;
            std::string size_str;
            int n = 1;
            std::vector<uint8_t> input_image_data;

            if (req.form.has_field("prompt")) {
                prompt = req.form.get_field("prompt");
            }
            if (req.form.has_field("n")) {
                n = std::stoi(req.form.get_field("n"));
            }
            if (req.form.has_field("size")) {
                size_str = req.form.get_field("size");
            }
            if (req.form.has_file("image[]")) {
                auto file = req.form.get_file("image[]");
                input_image_data.assign(file.content.begin(), file.content.end());
            }

            if (input_image_data.empty()) {
                res.status = 400;
                res.set_content(R"({"error":{"message":"image is required","type":"invalid_request_error"}})", "application/json");
                return;
            }

            // Parse size
            auto [req_width, req_height] = parse_size(size_str, config_.width, config_.height);

            SDConfig req_config = config_;
            req_config.width = req_width;
            req_config.height = req_height;

            // Parse generation parameters from form fields
            if (req.form.has_field("num_inference_steps")) {
                try { req_config.num_inference_steps = std::stoi(req.form.get_field("num_inference_steps")); } catch (...) {}
            }
            if (req.form.has_field("guidance_scale")) {
                try { req_config.guidance_scale = std::stof(req.form.get_field("guidance_scale")); } catch (...) {}
            }
            if (req.form.has_field("seed")) {
                try { req_config.seed = std::stoi(req.form.get_field("seed")); } catch (...) {}
            }
            if (req.form.has_field("strength")) {
                try { req_config.strength = std::stof(req.form.get_field("strength")); } catch (...) {}
            }

            // Parse extra args embedded in prompt (Lemonade embeds steps, cfg_scale, seed)
            auto extra = parse_extra_args(prompt.empty() ? "" : prompt);
            std::string clean_prompt = extra.clean_prompt.empty() ? "image variation" : extra.clean_prompt;
            if (extra.steps > 0)        req_config.num_inference_steps = extra.steps;
            if (extra.cfg_scale >= 0.f) req_config.guidance_scale = extra.cfg_scale;
            if (extra.seed >= 0)        req_config.seed = extra.seed;

            std::cout << "  prompt: \"" << clean_prompt << "\"" << std::endl;
            std::cout << "  size: " << req_config.width << "x" << req_config.height << std::endl;
            std::cout << "  steps: " << req_config.num_inference_steps
                      << ", cfg_scale: " << req_config.guidance_scale
                      << ", seed: " << req_config.seed
                      << ", strength: " << req_config.strength << std::endl;
            std::cout << "  n: " << n << std::endl;

            // Decode input image (expects base64-encoded raw RGB)
            std::vector<uint8_t> control_rgb;
            auto decoded = decode_image_data(input_image_data, req_config.width, req_config.height);
            if (decoded.valid) {
                control_rgb = std::move(decoded.rgb_data);
                std::cout << "  input image: " << decoded.width << "x" << decoded.height << " (raw RGB)" << std::endl;
            } else {
                res.status = 400;
                res.set_content(R"({"error":{"message":"failed to decode image - expected base64 raw RGB matching size","type":"invalid_request_error"}})", "application/json");
                return;
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Generate variations under lock
            std::vector<ImageData> all_variations;
            int resp_width = req_config.width, resp_height = req_config.height;
            {
                std::lock_guard<std::mutex> lock(pipeline_mutex_);
                for (int i = 0; i < n; i++) {
                    // Use different seed for each variation
                    SDConfig img_config = req_config;
                    img_config.seed = req_config.seed + i;
                    pipeline_->update_config(img_config);

                    auto img_response = pipeline_->generate(clean_prompt, "", control_rgb);
                    if (img_response.success && !img_response.data.empty()) {
                        all_variations.insert(all_variations.end(), img_response.data.begin(), img_response.data.end());
                        resp_width  = img_response.width;
                        resp_height = img_response.height;
                    }
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "[RESPONSE] Generated " << n << " variation(s) in " << gen_ms << " ms" << std::endl;

            // Build OpenAI-compatible response
            std::ostringstream response;
            response << R"({"created":)" << std::chrono::system_clock::now().time_since_epoch().count()
                     << R"(,"width":)"   << resp_width
                     << R"(,"height":)"  << resp_height
                     << R"(,"format":"raw_rgb")"
                     << R"(,"data":[)";
            for (size_t i = 0; i < all_variations.size(); i++) {
                if (i > 0) response << ",";
                response << R"({"b64_json":")" << all_variations[i].b64_json << R"("})";
            }
            response << "]}";

            res.set_content(response.str(), "application/json");

        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Image variation failed: " << e.what() << std::endl;

            std::string error_response = R"({
                "error": {
                    "message": ")" + std::string(e.what()) + R"(",
                    "type": "server_error"
                }
            })";

            res.status = 500;
            res.set_content(error_response, "application/json");
        }
    });

    std::cout << "Server ready at http://127.0.0.1:" << port_ << std::endl;
    std::cout << "Endpoints:" << std::endl;
    std::cout << "  GET  /health                    - Health check" << std::endl;
    std::cout << "  POST /v1/images/generations     - Generate images (OpenAI-compatible)" << std::endl;
    std::cout << "  POST /v1/images/edits           - Edit images (OpenAI-compatible)" << std::endl;
    std::cout << "  POST /v1/images/variations      - Generate variations (OpenAI-compatible)" << std::endl;
    std::cout << std::endl;
    
    // Listen (this blocks until server stops)
    if (!svr.listen("127.0.0.1", port_)) {
        std::cerr << "Failed to start server on port " << port_ << std::endl;
        running_ = false;
    }
#else
    std::cerr << "ERROR: httplib.h not found - cannot start HTTP server" << std::endl;
    std::cerr << "Download from: https://github.com/yhirose/cpp-httplib" << std::endl;
    std::cerr << "Place httplib.h in the include/ directory" << std::endl;
    running_ = false;
#endif
}

} // namespace sd_npu
