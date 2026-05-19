// main.cpp - CLI entry point for SD NPU Server
// Copyright (C) 2025 Advanced Micro Devices, Inc.
//
// Usage:
//   sd_npu_server --model-path <path> [options]
//   sd_npu_server --server --model-path <path> [--port 8080]
//
// Options are modelled after GenAI-SD pipeline_configs.yaml.

#include "sd_model.h"
#include "http_server.h"
#include "config_loader.h"
#include "variant_registry.h"
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <thread>
#include <chrono>
#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

// ── Usage / help ────────────────────────────────────────────────────────────

static void print_help() {
    std::cout << R"(
sd_npu_server - Stable Diffusion on AMD NPU via ONNX Runtime

USAGE:
  sd_npu_server --model-path <dir> [options]
  sd_npu_server --server --model-path <dir> [--port 8080]

REQUIRED:
  --model-path, -m <dir>      Base directory containing ONNX model components
                              (auto-discovers unet/transformer, vae_decoder, etc.)
  --config, -f <path>         Path to config.json file
                              [default: auto-discover in current/exe/parent dir]

MODEL OPTIONS:
  --variant, -v <name>        Model variant: sd15, sdxl, sd3, sd35
                              (auto-detected from directory if not specified)
  --model-id <id>             HuggingFace model ID for auto-download
                              (e.g. amd/stable-diffusion-1.5-amdnpu)
  --revision <rev>            Git branch/tag/commit for HF download [default: main]
  --custom-ops, -c <path>     Path to onnx_custom_ops.dll

DD ENVIRONMENT:
  --dd-root <path>            DD root path for DynamicDispatch
  --dd-plugins-root <path>    DD plugins path (set as DD_ROOT / DD_PLUGINS_ROOT env vars)

GENERATION OPTIONS:
  --prompt, -p <text>         Text prompt
                              [default: "An astronaut riding a green horse"]
  --negative-prompt, --n-prompt <text>
                              Negative prompt [default: ""]
  --prompt-file <path>        JSON file with prompts (overrides --prompt)
  --width, -W <int>           Image width  [default: auto based on variant]
  --height, -H <int>          Image height [default: auto based on variant]
  --num-inference-steps, -n <int>
                              Number of denoising steps [default: auto based on variant]
  --guidance-scale, -g <float>
                              Classifier-free guidance scale [default: 7.0]
  --seed, -s <int>            Random seed [default: 0]
  --num-images, --num-images-per-prompt <int>
                              Number of images to generate [default: 1]
  --strength <float>          Img2img denoising strength (0.0-1.0) [default: 0.8]
  --scheduler <name>          Scheduler type: euler, euler_a, pndm, ddim, lms [default: auto]

CONTROLNET OPTIONS:
  --controlnet, -C <type>     ControlNet type: None, Canny, Pose, Tile, Depth,
                              Union, OutPainting, InPainting, Removal [default: None]
  --control-image <path>      Path to control/input image
  --control-mask <path>       Path to control mask image (for inpainting/outpainting)
  --controlnet-scale <float>  ControlNet conditioning scale [default: 0.5]
  --image-pads <t,r,b,l>     Comma-separated padding values for outpainting

SD3/SD3.5 OPTIONS:
  --t5-sequence-len <int>     T5 text encoder sequence length [default: 77]
  --dynamic-shape, -ds        Enable dynamic shape inference
  --dynamic-shape-file <path> JSON config file for dynamic shapes (implies --dynamic-shape)

EXECUTION PROVIDER OPTIONS:
  --force-cpu                 Force CPU execution provider
  --gpu                       Use GPU execution provider
  --enable-compile            Enable model compilation/caching

PROFILING:
  --enable-profile, -P        Enable profiling output
  --profiling-rounds <int>    Number of profiling iterations [default: 1]

OUTPUT:
  --output, -o <path>         Output file path (.png, .bmp, .jpg, or .raw) [default: output.png]
  --output-dir <dir>          Directory for output images [default: current dir]
  --no-images                 Skip saving images (profiling/benchmarking only)

SERVER MODE:
  --server                    Start HTTP server instead of single generation
  --port, --listen-port <int> HTTP server port [default: 8080]

FLAGS:
  --verbose, -V               Enable verbose/debug output

EXAMPLES:
  # Generate with SDXL model (auto-detected)
  sd_npu_server -m C:/models/amd_sdxl-base-amdnpu -p "A sunset over mountains"

  # SD3 with custom steps, guidance, and dynamic shapes
  sd_npu_server -m C:/models/sd3 -v sd3 -n 8 -g 4.5 --dynamic-shape

  # SDXL Turbo (1 step, no guidance)
  sd_npu_server -m C:/models/sdxl_turbo -v sdxl -n 1 -g 0.0 -W 512 -H 512

  # ControlNet Canny with custom strength
  sd_npu_server -m C:/models/sd15 -C Canny --control-image input.png --controlnet-scale 0.7

  # Start HTTP server
  sd_npu_server --server -m C:/models/sd3 --port 9090
)";
}

// ── Argument parsing ────────────────────────────────────────────────────────

struct CliArgs {
    // Configuration
    std::string config_file;       // Path to config.json (empty = auto-discover)
    
    // Required
    std::string model_path;

    // Model
    std::string model_id;          // HuggingFace model ID for auto-download
    std::string revision;          // Git branch/tag/commit for HF download
    std::string variant_str;       // empty = auto-detect
    std::string custom_ops_path = "C:/Program Files/RyzenAI/1.7.1/deployment/onnx_custom_ops.dll";
    // DD env defaults (absolute paths)
    std::string dd_root = "C:/Program Files/RyzenAI/1.7.1/deployment";
    std::string dd_plugins_root = "C:/Program Files/RyzenAI/1.7.1/deployment";

    // Generation
    std::string prompt = "An astronaut riding a green horse";
    std::string negative_prompt;
    std::string prompt_file;       // Path to JSON file with list of prompts
    int width = -1;                // -1 = auto
    int height = -1;
    int num_inference_steps = -1;  // -1 = auto
    float guidance_scale = -1.0f;   // -1 = auto based on variant
    int seed = 0;
    int num_images = 1;
    float strength = 0.3f;        // Img2img transformation strength

    // ControlNet
    std::string controlnet;        // empty = none
    std::string control_image;
    std::string control_mask;      // For outpainting/inpainting/removal
    float controlnet_scale = 0.5f;
    std::vector<int> image_pads;   // [top, bottom, left, right] for outpainting

    // SD3-specific
    int t5_sequence_len = 83;      // T5 tokenizer sequence length (77 or 83)
    bool dynamic_shape = false;    // Enable dynamic shape flow
    std::string dynamic_shape_file; // JSON config for multiple resolutions

    // Execution
    bool force_cpu = false;        // Force CPU-only execution
    bool gpu = false;              // Use DML/GPU
    bool enable_compile = false;   // Enable compile fusion runtime

    // Profiling
    bool enable_profile = false;   // Enable profiling measurement
    int profiling_rounds = 4;      // Number of profiling iterations

    // Output
    std::string output = "output.png";
    std::string output_dir = "generated_images";
    bool no_images = false;        // Skip saving images (profiling only)

    // Scheduler
    std::string scheduler;         // Explicit scheduler choice

    // Server
    bool server_mode = false;
    int port = 8080;

    // Flags
    bool help = false;
    bool verbose = false;
};

static bool next_arg(int i, int argc, const char* name) {
    if (i + 1 >= argc) {
        std::cerr << "Error: " << name << " requires a value\n";
        return false;
    }
    return true;
}

static CliArgs parse_args(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        // Helper: consume next token as a string value
        auto str_val = [&](std::string& field) -> bool {
            if (!next_arg(i, argc, arg.c_str())) return false;
            field = argv[++i];
            return true;
        };
        auto int_val = [&](int& field) -> bool {
            if (!next_arg(i, argc, arg.c_str())) return false;
            field = std::atoi(argv[++i]);
            return true;
        };
        auto float_val = [&](float& field) -> bool {
            if (!next_arg(i, argc, arg.c_str())) return false;
            field = (float)std::atof(argv[++i]);
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            a.help = true;
        }
        else if (arg == "--config" || arg == "-f")           { str_val(a.config_file); }
        else if (arg == "--model-path" || arg == "-m")       { str_val(a.model_path); }
        else if (arg == "--model-id")                        { str_val(a.model_id); }
        else if (arg == "--revision")                        { str_val(a.revision); }
        else if (arg == "--variant" || arg == "-v")          { str_val(a.variant_str); }
        else if (arg == "--custom-ops" || arg == "-c")       { str_val(a.custom_ops_path); }
        else if (arg == "--dd-root")                         { str_val(a.dd_root); }
        else if (arg == "--dd-plugins-root")                 { str_val(a.dd_plugins_root); }
        else if (arg == "--prompt" || arg == "-p")           { str_val(a.prompt); }
        else if (arg == "--negative-prompt" || arg == "--n-prompt") { str_val(a.negative_prompt); }
        else if (arg == "--prompt-file" || arg == "--prompt-file-path") { str_val(a.prompt_file); }
        else if (arg == "--width" || arg == "-W")            { int_val(a.width); }
        else if (arg == "--height" || arg == "-H")           { int_val(a.height); }
        else if (arg == "--num-inference-steps" || arg == "-n") { int_val(a.num_inference_steps); }
        else if (arg == "--guidance-scale" || arg == "-g")   { float_val(a.guidance_scale); }
        else if (arg == "--seed" || arg == "-s")             { int_val(a.seed); }
        else if (arg == "--num-images" || arg == "--num-images-per-prompt") { int_val(a.num_images); }
        else if (arg == "--strength")                        { float_val(a.strength); }
        else if (arg == "--controlnet" || arg == "-C") {
            if (str_val(a.controlnet) &&
                (a.controlnet == "None" || a.controlnet == "none"))
                a.controlnet.clear();
        }
        else if (arg == "--control-image" || arg == "--control-image-path") { str_val(a.control_image); }
        else if (arg == "--control-mask" || arg == "--control-mask-path")   { str_val(a.control_mask); }
        else if (arg == "--controlnet-scale" || arg == "--controlnet-conditioning-scale") {
            float_val(a.controlnet_scale);
        }
        else if (arg == "--image-pads") {
            if (next_arg(i, argc, arg.c_str())) {
                std::string pads_str = argv[++i];
                std::istringstream ss(pads_str);
                std::string token;
                a.image_pads.clear();
                while (std::getline(ss, token, ','))
                    a.image_pads.push_back(std::atoi(token.c_str()));
            }
        }
        else if (arg == "--t5-sequence-len")                 { int_val(a.t5_sequence_len); }
        else if (arg == "--dynamic-shape" || arg == "-ds")   { a.dynamic_shape = true; }
        else if (arg == "--dynamic-shape-file") {
            if (str_val(a.dynamic_shape_file)) a.dynamic_shape = true;
        }
        else if (arg == "--force-cpu")                       { a.force_cpu = true; }
        else if (arg == "--gpu")                             { a.gpu = true; }
        else if (arg == "--enable-compile")                  { a.enable_compile = true; }
        else if (arg == "--enable-profile" || arg == "-P")   { a.enable_profile = true; }
        else if (arg == "--profiling-rounds")                { int_val(a.profiling_rounds); }
        else if (arg == "--scheduler")                       { str_val(a.scheduler); }
        else if (arg == "--output" || arg == "-o")           { str_val(a.output); }
        else if (arg == "--output-dir")                      { str_val(a.output_dir); }
        else if (arg == "--no-images")                       { a.no_images = true; }
        else if (arg == "--server")                          { a.server_mode = true; }
        else if (arg == "--port" || arg == "--listen-port")  { int_val(a.port); }
        else if (arg == "--verbose" || arg == "-V")          { a.verbose = true; }
        else {
            std::cerr << "Warning: unknown argument '" << arg << "'\n";
        }
    }
    return a;
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    if (args.help) {
        print_help();
        return 0;
    }

    if (args.model_path.empty()) {
        print_help();
        return 1;
    }

    try {
        // Detect or set variant using registry
        sd_npu::ModelVariant variant;
        const sd_npu::VariantDescriptor* desc = nullptr;
        auto& registry = sd_npu::VariantRegistry::instance();

        if (!args.variant_str.empty()) {
            desc = registry.find(args.variant_str);
            if (!desc) {
                throw std::runtime_error("Unknown variant: " + args.variant_str);
            }
            variant = desc->variant;
        } else {
            desc = registry.detect(fs::path(args.model_path));
            if (!desc) {
                throw std::runtime_error("Could not auto-detect variant for: " + args.model_path);
            }
            variant = desc->variant;
        }

        // Save CLI-provided values (non-sentinel values were explicitly provided)
        int cli_width = args.width;
        int cli_height = args.height;
        int cli_steps = args.num_inference_steps;
        float cli_guidance = args.guidance_scale;
        std::string cli_prompt = args.prompt;
        std::string cli_negative_prompt = args.negative_prompt;

        // Set variant-specific defaults from registry
        args.width = desc->default_width;
        args.height = desc->default_height;
        args.num_inference_steps = desc->default_steps;
        args.guidance_scale = desc->default_guidance;
        
        std::cout << "[DEBUG] After variant defaults: steps=" << args.num_inference_steps 
                  << ", guidance=" << args.guidance_scale << std::endl;

        // Load configuration file - config can override variant defaults
        std::string config_path = args.config_file.empty() ? 
                                 sd_npu::ConfigLoader::get_default_config_path() : 
                                 args.config_file;
        
        auto config_opt = sd_npu::ConfigLoader::load(config_path);
        if (config_opt.has_value()) {
            auto& cfg = config_opt.value();
            std::cout << "[CONFIG] Loaded config from: " << config_path << std::endl;
            
            // Apply config defaults for paths (command-line args override)
            if (args.custom_ops_path == "C:/Program Files/RyzenAI/1.7.1/deployment/onnx_custom_ops.dll" && 
                !cfg.custom_ops_dll.empty()) {
                args.custom_ops_path = cfg.custom_ops_dll;
            }
            if (args.dd_root == "C:/Program Files/RyzenAI/1.7.1/deployment" && 
                !cfg.dd_root.empty()) {
                args.dd_root = cfg.dd_root;
            }
            if (args.dd_plugins_root == "C:/Program Files/RyzenAI/1.7.1/deployment" && 
                !cfg.dd_plugins_root.empty()) {
                args.dd_plugins_root = cfg.dd_plugins_root;
            }
            
            // Apply config defaults for server
            if (args.port == 8080 && cfg.default_port != 8080) {
                args.port = cfg.default_port;
            }
            
            // Config file overrides variant defaults
            if (cfg.default_width > 0) {
                args.width = cfg.default_width;
            }
            if (cfg.default_height > 0) {
                args.height = cfg.default_height;
            }
            if (cfg.default_steps > 0) {
                args.num_inference_steps = cfg.default_steps;
                std::cout << "[DEBUG] Config override: steps=" << cfg.default_steps << std::endl;
            }
            // Only apply config.json guidance if the variant default is non-zero.
            // Turbo variants explicitly require guidance=0; overriding that produces
            // all-white images because CFG is incompatible with distilled Turbo models.
            bool variant_requires_no_cfg = (variant == sd_npu::ModelVariant::SD_TURBO ||
                                            variant == sd_npu::ModelVariant::SDXL_TURBO);
            if (cfg.default_guidance_scale > 0.0f && !variant_requires_no_cfg) {
                args.guidance_scale = cfg.default_guidance_scale;
            }
            if (!cfg.default_prompt.empty()) {
                args.prompt = cfg.default_prompt;
            }
            if (!cfg.default_negative_prompt.empty()) {
                args.negative_prompt = cfg.default_negative_prompt;
            }
        } else {
            std::cout << "[CONFIG] No config file found, using variant defaults" << std::endl;
        }

        // CLI args override everything (restore explicitly-provided CLI values)
        if (cli_width >= 0) args.width = cli_width;
        if (cli_height >= 0) args.height = cli_height;
        if (cli_steps >= 0) args.num_inference_steps = cli_steps;
        if (cli_guidance >= 0.0f) args.guidance_scale = cli_guidance;
        if (cli_prompt != "An astronaut riding a green horse") args.prompt = cli_prompt;
        if (!cli_negative_prompt.empty()) args.negative_prompt = cli_negative_prompt;
        
        std::cout << "[DEBUG] Final: steps=" << args.num_inference_steps << ", guidance=" << args.guidance_scale << std::endl;

        // Build config
        sd_npu::SDConfig config;
        config.variant             = variant;
        config.width               = args.width;
        config.height              = args.height;
        config.num_inference_steps = args.num_inference_steps;
        config.guidance_scale      = args.guidance_scale;
        config.seed                = args.seed;
        config.num_images_per_prompt = args.num_images;
        config.custom_op_path      = args.custom_ops_path;
        config.controlnet_type     = args.controlnet;
        config.controlnet_scale    = args.controlnet_scale;
        config.control_image_path  = args.control_image;
        config.output_path         = args.output;
        config.model_id            = args.model_id;
        config.revision            = args.revision;
        config.control_mask_path   = args.control_mask;
        config.image_pads          = args.image_pads;
        config.strength            = args.strength;
        config.t5_sequence_len     = args.t5_sequence_len;
        config.dynamic_shape       = args.dynamic_shape;
        config.dynamic_shape_file  = args.dynamic_shape_file;
        config.force_cpu           = args.force_cpu;
        config.gpu                 = args.gpu;
        config.enable_compile      = args.enable_compile;
        config.enable_profile      = args.enable_profile;
        config.profiling_rounds    = args.profiling_rounds;
        config.output_dir          = args.output_dir;
        config.no_images           = args.no_images;
        config.prompt_file         = args.prompt_file;
        config.verbose             = args.verbose;

        // Discover components using registry
        registry.discover_components(fs::path(args.model_path), *desc, config);

        // Generate timestamped output filename if using default
        if (args.output == "output.png") {
            auto now = std::chrono::system_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
            std::string variant_str = sd_npu::variant_to_string(variant);
            // Remove spaces from variant name
            variant_str.erase(std::remove(variant_str.begin(), variant_str.end(), ' '), variant_str.end());
            args.output = variant_str + "_" + std::to_string(config.width) + "x" + std::to_string(config.height) 
                        + "_" + std::to_string(timestamp) + ".png";
            config.output_path = args.output;
        }

        // Print configuration
        std::cout << "========================================\n";
        std::cout << "SD NPU Server\n";
        std::cout << "========================================\n";
        std::cout << "Model path:    " << args.model_path << "\n";
        std::cout << "Variant:       " << sd_npu::variant_to_string(variant) << "\n";
        std::cout << "Resolution:    " << config.width << "x" << config.height << "\n";
        std::cout << "Steps:         " << config.num_inference_steps << "\n";
        std::cout << "Guidance:      " << config.guidance_scale << "\n";
        std::cout << "Seed:          " << config.seed << "\n";
        std::cout << "Custom ops:    " << config.custom_op_path << "\n";
        if (!config.controlnet_type.empty()) {
            std::cout << "ControlNet:    " << config.controlnet_type
                      << " (scale=" << config.controlnet_scale << ")\n";
            std::cout << "Control image: " << config.control_image_path << "\n";
        }
        std::cout << "Components:    ";
        for (const auto& [name, path] : config.component_paths)
            std::cout << name << " ";
        std::cout << "\n";
        std::cout << "========================================\n";

        if (config.component_paths.empty()) {
            std::cerr << "Error: no ONNX model components found in " << args.model_path << "\n";
            return 1;
        }

        // Set DD environment variables before model loading (needed for both
        // server mode and generation mode so NPU DynamicDispatch finds its libs).
        // Priority: exe-local lib/ (if it exists) > config.json / CLI args > hardcoded default.
        std::string dd_root, dd_plugins;
    #ifdef _WIN32
        char exe_path[MAX_PATH];
        GetModuleFileNameA(NULL, exe_path, MAX_PATH);
        fs::path exe_dir = fs::path(exe_path).parent_path();
        fs::path exe_lib = exe_dir / "lib";

        if (fs::exists(exe_lib / "transaction" / "stx")) {
            // Prefer exe-local lib when it has been deployed by CMake
            dd_root = exe_lib.string();
            dd_plugins = (exe_lib / "transaction" / "stx").string();
        } else {
            // Fall back to args (from config.json or CLI)
            dd_root = args.dd_root;
            dd_plugins = args.dd_plugins_root;
        }
        
        SetEnvironmentVariableA("DD_ROOT", dd_root.c_str());
        SetEnvironmentVariableA("DD_PLUGINS_ROOT", dd_plugins.c_str());
    #else
        char exe_path[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (len != -1) {
            exe_path[len] = '\0';
            fs::path exe_dir = fs::path(exe_path).parent_path();
            fs::path exe_lib = exe_dir / "lib";

            if (fs::exists(exe_lib / "transaction" / "stx")) {
                dd_root = exe_lib.string();
                dd_plugins = (exe_lib / "transaction" / "stx").string();
            } else {
                dd_root = args.dd_root;
                dd_plugins = args.dd_plugins_root;
            }
        } else {
            dd_root = args.dd_root;
            dd_plugins = args.dd_plugins_root;
        }
        
        setenv("DD_ROOT", dd_root.c_str(), 1);
        setenv("DD_PLUGINS_ROOT", dd_plugins.c_str(), 1);
    #endif
        std::cout << "DD_ROOT=" << dd_root << std::endl;
        std::cout << "DD_PLUGINS_ROOT=" << dd_plugins << std::endl;

        // ── Server mode ─────────────────────────────────────────────────
        if (args.server_mode) {
            std::cout << "Starting HTTP server on port " << args.port << "...\n";
            sd_npu::SDServer server(args.port, config, args.model_path);
            server.start();
            std::cout << "Press Ctrl+C to stop\n";
            while (true)
                std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // ── Generation mode ─────────────────────────────────────────────
        std::cout << "\nCreating pipeline...\n";

        auto t0 = std::chrono::high_resolution_clock::now();
        sd_npu::SDPipeline pipeline(config);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "Pipeline ready (" << load_ms << " ms)\n";

        for (int img = 0; img < args.num_images; img++) {
            // Per-image seed
            config.seed = args.seed + img;

            std::string out_path = args.output;
            if (args.num_images > 1) {
                // Insert index before extension: output_0.png, output_1.png…
                auto dot = out_path.rfind('.');
                auto base_name = (dot != std::string::npos) ? out_path.substr(0, dot) : out_path;
                auto ext = (dot != std::string::npos) ? out_path.substr(dot) : ".png";
                out_path = base_name + "_" + std::to_string(img) + ext;
            }

            std::cout << "\n── Image " << (img + 1) << "/" << args.num_images << " ──\n";
            auto gen_start = std::chrono::high_resolution_clock::now();
            auto response = pipeline.generate(args.prompt, args.negative_prompt);
            auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - gen_start).count();

            if (!response.success) {
                std::cerr << "Error: " << response.error << "\n";
                continue;
            }

            std::cout << "Generated in " << gen_ms << " ms\n";
            std::cout << "Image: " << response.width << "x" << response.height 
                      << " (" << response.format << ")\n";
            std::cout << "Timing breakdown:\n"
                      << "  Text encoding: " << response.text_encoding_time_ms << " ms\n"
                      << "  Denoising (" << response.num_inference_steps << " steps): " 
                      << response.denoising_time_ms << " ms\n"
                      << "  VAE decoding: " << response.vae_decoding_time_ms << " ms\n"
                      << "  Total: " << response.generation_time_ms << " ms\n";

            if (!response.data.empty() && !response.data[0].b64_json.empty()) {
                std::cout << "Base64 image size: " << response.data[0].b64_json.length() << " chars\n";
                
                // Save raw RGB data (use Python for PNG/JPG conversion)
                if (!response.image_data.empty()) {
                    std::ofstream f(out_path, std::ios::binary);
                    f.write(reinterpret_cast<const char*>(response.image_data.data()), 
                            response.image_data.size());
                    std::cout << "Saved raw RGB: " << out_path << " (" 
                              << response.width << "x" << response.height << ", " 
                              << response.image_data.size() << " bytes)\n";
                    std::cout << "Use Python to convert: PIL.Image.fromarray(data.reshape(" 
                              << response.height << "," << response.width << ",3), 'RGB').save('out.png')\n";
                }
            } else {
                std::cout << "Warning: No image data in response\n";
            }
        }

        std::cout << "\nDone.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
