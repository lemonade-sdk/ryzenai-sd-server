cmake_minimum_required(VERSION 3.20)
project(ryzenai-sd-server VERSION 1.7.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

message(STATUS "Building Ryzen AI SD Server")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find dependencies
find_package(Threads REQUIRED)

# ── ONNX Runtime ─────────────────────────────────────────────────────────────
# Headers are vendored in include/onnxruntime/ — no SDK needed for compilation.
# Only the import library (.lib) and runtime DLLs are required externally.
#
# ONNXRUNTIME_LIB_DIR: directory containing onnxruntime.lib (and DLLs to deploy)
#   Priority: -DONNXRUNTIME_LIB_DIR > ONNXRUNTIME_LIB_DIR env var > platform default
if(NOT DEFINED ONNXRUNTIME_LIB_DIR)
    if(DEFINED ENV{ONNXRUNTIME_LIB_DIR})
        set(ONNXRUNTIME_LIB_DIR "$ENV{ONNXRUNTIME_LIB_DIR}" CACHE PATH
            "Directory containing onnxruntime.lib and runtime DLLs")
    elseif(WIN32)
        set(ONNXRUNTIME_LIB_DIR "C:/Program Files/RyzenAI/1.7.1/onnxruntime/lib" CACHE PATH
            "Directory containing onnxruntime.lib and runtime DLLs")
    elseif(UNIX AND NOT APPLE)
        set(ONNXRUNTIME_LIB_DIR "/opt/ryzenai/1.7.1/onnxruntime/lib" CACHE PATH
            "Directory containing libonnxruntime.so")
    endif()
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIB_DIR}")
    message(FATAL_ERROR "ONNX Runtime lib directory not found at: ${ONNXRUNTIME_LIB_DIR}\n"
        "Set -DONNXRUNTIME_LIB_DIR=<path> to the directory containing onnxruntime.lib / libonnxruntime.so")
endif()

find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_LIB_DIR}" REQUIRED NO_DEFAULT_PATH)
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")

# Optional: directory containing extra DLLs to deploy alongside the executable
# (e.g., RyzenAI deployment DLLs, DynamicDispatch libs)
set(RUNTIME_DLLS_DIR "" CACHE PATH "Directory with additional DLLs to copy next to the executable")
set(LIB_DIRECTORY "" CACHE PATH "Additional lib directory tree to copy (preserves subdirectory structure)")

# ── Download header-only libraries if not present ────────────────────────────

# cpp-httplib
set(HTTPLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/cpp-httplib")
if(NOT EXISTS "${HTTPLIB_DIR}/httplib.h")
    message(STATUS "Downloading cpp-httplib...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.26.0/httplib.h"
        "${HTTPLIB_DIR}/httplib.h"
        SHOW_PROGRESS
    )
endif()

# stb_image (single-header PNG/JPEG/BMP decoder)
set(STB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/stb")
if(NOT EXISTS "${STB_DIR}/stb_image.h")
    message(STATUS "Downloading stb_image.h...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/nothings/stb/f0569113c93ad095470c54bf34a17b36646bbbb5/stb_image.h"
        "${STB_DIR}/stb_image.h"
        SHOW_PROGRESS
    )
endif()

# ── Source files ─────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/onnx_model.cpp
    src/scheduler.cpp
    src/sd_pipeline.cpp
    src/clip_tokenizer.cpp
    src/http_server.cpp
    src/config_loader.cpp
    src/variant_registry.cpp
    src/controlnet_runner.cpp
    # Variant implementations
    src/variants/generic_denoiser.cpp    # unified denoiser for SD1.5 / SDXL / SD3
    src/variants/sd15_text_encoder.cpp
    src/variants/sd15_vae_decoder.cpp
    src/variants/sdxl_text_encoder.cpp
    src/variants/sdxl_vae_decoder.cpp
    src/variants/sd3_text_encoder.cpp
    src/variants/sd3_vae_decoder.cpp
    # Variant registrations (self-registering descriptors)
    src/variants/sd15_variant.cpp
    src/variants/sdxl_variant.cpp
    src/variants/sd3_variant.cpp
)

# Create executable
add_executable(ryzenai-sd-server ${SOURCES})

# Include directories
# - include/sd/          : project headers
# - include/onnxruntime/ : vendored ONNX Runtime headers (no SDK install needed)
# - external/cpp-httplib/ : auto-downloaded HTTP library
target_include_directories(ryzenai-sd-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/sd
    ${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime
    ${HTTPLIB_DIR}
    ${STB_DIR}
)

# Link libraries
target_link_libraries(ryzenai-sd-server PRIVATE
    ${ONNXRUNTIME_LIB}
    Threads::Threads
)

# Pass version info from CMake to C++ as compile definitions
target_compile_definitions(ryzenai-sd-server PRIVATE
    CPPHTTPLIB_THREAD_POOL_COUNT=8
    RYZENAI_SD_SERVER_VERSION="${PROJECT_VERSION}"
)

# ── Windows-specific settings ────────────────────────────────────────────────
if(WIN32)
    # Target Windows 10 or later (required for cpp-httplib v0.26.0)
    target_compile_definitions(ryzenai-sd-server PRIVATE
        _WIN32_WINNT=0x0A00
        _CRT_SECURE_NO_WARNINGS
    )
    target_link_libraries(ryzenai-sd-server PRIVATE winhttp ws2_32)

    # Compiler warnings
    target_compile_options(ryzenai-sd-server PRIVATE /W4 /WX-)

    # Copy ONNX Runtime DLLs to output directory
    # Look for DLLs in the lib dir itself and a sibling bin/ directory
    set(ONNXRT_DLL_DIRS "${ONNXRUNTIME_LIB_DIR}")
    get_filename_component(ONNXRT_BIN_DIR "${ONNXRUNTIME_LIB_DIR}/../bin" ABSOLUTE)
    if(EXISTS "${ONNXRT_BIN_DIR}")
        list(APPEND ONNXRT_DLL_DIRS "${ONNXRT_BIN_DIR}")
    endif()

    foreach(dll_dir ${ONNXRT_DLL_DIRS})
        file(GLOB _dlls "${dll_dir}/*.dll")
        foreach(dll ${_dlls})
            add_custom_command(TARGET ryzenai-sd-server POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${dll}"
                    $<TARGET_FILE_DIR:ryzenai-sd-server>
                VERBATIM
            )
        endforeach()
    endforeach()

    # Copy additional runtime DLLs if specified (e.g., RyzenAI custom ops)
    if(RUNTIME_DLLS_DIR AND EXISTS "${RUNTIME_DLLS_DIR}")
        file(GLOB EXTRA_DLLS "${RUNTIME_DLLS_DIR}/*.dll")
        list(LENGTH EXTRA_DLLS EXTRA_DLL_COUNT)
        if(EXTRA_DLL_COUNT GREATER 0)
            add_custom_command(TARGET ryzenai-sd-server POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "Deploying ${EXTRA_DLL_COUNT} additional DLL(s)..."
                VERBATIM
            )
            foreach(dll ${EXTRA_DLLS})
                add_custom_command(TARGET ryzenai-sd-server POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${dll}"
                        $<TARGET_FILE_DIR:ryzenai-sd-server>
                    VERBATIM
                )
            endforeach()
        endif()
    endif()

    # Copy additional lib directory if specified (e.g., GenAI-SD/lib for DD runtime)
    if(LIB_DIRECTORY AND EXISTS "${LIB_DIRECTORY}")
        file(GLOB_RECURSE LIB_FILES "${LIB_DIRECTORY}/*")
        list(LENGTH LIB_FILES LIB_FILE_COUNT)

        if(LIB_FILE_COUNT GREATER 0)
            add_custom_command(TARGET ryzenai-sd-server POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "Deploying lib directory..."
                VERBATIM
            )

            foreach(lib_file ${LIB_FILES})
                file(RELATIVE_PATH rel_path "${LIB_DIRECTORY}" "${lib_file}")
                get_filename_component(rel_dir "${rel_path}" DIRECTORY)

                if(rel_dir)
                    add_custom_command(TARGET ryzenai-sd-server POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E make_directory
                            "$<TARGET_FILE_DIR:ryzenai-sd-server>/lib/${rel_dir}"
                        VERBATIM
                    )
                endif()

                add_custom_command(TARGET ryzenai-sd-server POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${lib_file}"
                        "$<TARGET_FILE_DIR:ryzenai-sd-server>/lib/${rel_path}"
                    VERBATIM
                )
            endforeach()
        endif()
    endif()

    # Set console subsystem explicitly
    set_target_properties(ryzenai-sd-server PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:CONSOLE"
    )

elseif(UNIX AND NOT APPLE)
    target_compile_options(ryzenai-sd-server PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(ryzenai-sd-server PRIVATE dl)

    # RPATH so shared libs next to the binary are found
    target_link_options(ryzenai-sd-server PRIVATE "-Wl,--disable-new-dtags")
    set_target_properties(ryzenai-sd-server PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )

    # Copy extra runtime shared libraries next to the binary
    if(RUNTIME_DLLS_DIR AND EXISTS "${RUNTIME_DLLS_DIR}")
        add_custom_command(TARGET ryzenai-sd-server POST_BUILD
            COMMAND cp -a "${RUNTIME_DLLS_DIR}"/. $<TARGET_FILE_DIR:ryzenai-sd-server>/
            COMMENT "Copying additional runtime libraries"
        )
    endif()
endif()

# Installation rules
install(TARGETS ryzenai-sd-server
    RUNTIME DESTINATION bin
)

# ── Print build information ──────────────────────────────────────────────────
message(STATUS "==========================================================")
message(STATUS "Ryzen AI SD Server Configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  ORT lib dir:       ${ONNXRUNTIME_LIB_DIR}")
message(STATUS "  Output directory:  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "==========================================================")
