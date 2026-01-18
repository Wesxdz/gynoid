# Default Paphos project starter
# flecs/PyTorch with AOT Inductor pipeline

cmake_minimum_required(VERSION 3.18)
project(project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# add_definitions(AOTInductor_CUDA)
add_executable(${PROJECT_NAME} src/main.cpp src/mel_spec_render.cpp src/glad.c src/vision_processor.cpp src/x11_outline.cpp src/dino_embedder.cpp)
# deps/tracy-0.13.1/public/TracyClient.cpp

include_directories("src")
include_directories("deps/mel_spec/src")
include_directories("deps/dinov2.cpp")
include_directories("deps/dinov2.cpp/ggml/include")
include_directories("deps/mel_spec/deps/glad/include")
include_directories("deps/spdlog/include")
include_directories("deps/mel_spec/deps/fpng/src")
include_directories("deps/json-3.11.2/single_include")

add_definitions(-DTRACY_ENABLE)
add_subdirectory("deps/tracy-0.13.1")

# Flecs
include_directories("deps/flecs-4.1.1/include")
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build Flecs as a shared library" FORCE)
add_subdirectory("deps/flecs-4.1.1")

# Flecs custom build configuration - must match defines in main.cpp
target_compile_definitions(flecs PUBLIC
    FLECS_CUSTOM_BUILD
    FLECS_ALERTS
    FLECS_APP
    FLECS_CPP
    FLECS_DOC
    FLECS_JSON
    FLECS_HTTP
    FLECS_LOG
    FLECS_META
    FLECS_METRICS
    FLECS_MODULE
    FLECS_OS_API_IMPL
    FLECS_PERF_TRACE
    FLECS_PIPELINE
    FLECS_REST
    FLECS_PARSER
    FLECS_QUERY_DSL
    FLECS_SCRIPT
    FLECS_SYSTEM
    FLECS_STATS
    FLECS_TIMER
    FLECS_UNITS
)

add_subdirectory("deps/spdlog")

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("deps/glfw")

include_directories("deps/raylib")

# GLAD
add_library(glad STATIC
    deps/glad/src/glad.c
)
target_include_directories(glad PUBLIC deps/glad/include)

# NanoVG
add_library(nanovg STATIC
    deps/nanovg/src/nanovg.c
)
target_include_directories(nanovg PUBLIC deps/nanovg/src)
target_compile_definitions(nanovg PUBLIC NANOVG_GL2_IMPLEMENTATION)

# fpng
add_library(fpng STATIC
    deps/mel_spec/deps/fpng/src/fpng.cpp
)
target_include_directories(fpng PUBLIC deps/mel_spec/deps/fpng/src)
set_target_properties(fpng PROPERTIES CXX_STANDARD 11)
target_compile_options(fpng PRIVATE -msse4.1 -mpclmul)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LibVNCServer)

# DINOv2.cpp / ggml for visual embeddings
# Enable CUDA for ggml (much faster inference)
set(GGML_CUDA ON CACHE BOOL "Enable CUDA for ggml" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures" FORCE)
add_subdirectory("deps/dinov2.cpp/ggml")
add_library(dinov2 STATIC deps/dinov2.cpp/dinov2.cpp)
target_include_directories(dinov2 PUBLIC deps/dinov2.cpp deps/dinov2.cpp/ggml/include)
target_compile_definitions(dinov2 PUBLIC GGML_USE_CUDA)
target_link_libraries(dinov2 PUBLIC ggml ${OpenCV_LIBS})
set_target_properties(dinov2 PROPERTIES CXX_STANDARD 20)

# add_subdirectory("deps/libssh2-1.11.1")
find_package(libssh2 REQUIRED CONFIG)
# include_directories("deps/libssh2-1.11.1/include")

# PYTHON QUERY SERVER ----
# Add spatialindex library
include_directories("deps/spatialindex-src-2.1.0/include")
add_subdirectory("deps/spatialindex-src-2.1.0")
# Add LMDB library
include_directories("deps/liblmdb")
add_library(lmdb STATIC deps/liblmdb/mdb.c deps/liblmdb/midl.c)
set_target_properties(lmdb PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
# Build query server shared library (contains all query/socket/spatial logic)
add_library(query_server SHARED src/query_server.cpp)
target_link_libraries(query_server flecs spatialindex lmdb pthread)
set_property(TARGET query_server PROPERTY CXX_STANDARD 20)
# ----

target_link_libraries(${PROJECT_NAME} flecs glfw glad nanovg fpng X11 Xi Xtst SDL2 SDL2_image ${OpenCV_LIBS} query_server Tracy::TracyClient libssh2::libssh2 dinov2)

# Add audio stream client executable
add_executable(audio_stream_client src/audio_stream_client.c)
target_link_libraries(audio_stream_client
    "${CMAKE_SOURCE_DIR}/deps/mel_spec/build/libaudio_stream.so"
    "${CMAKE_SOURCE_DIR}/deps/mel_spec/build/libmelspec.so"
)

# Link mel_spec .so libraries
link_directories("${CMAKE_SOURCE_DIR}/deps/mel_spec/build")
link_directories("${CMAKE_SOURCE_DIR}/deps/mel_spec/build/deps/libsoundio")

# Add mel_spec dependencies
target_link_libraries(${PROJECT_NAME}
    "${CMAKE_SOURCE_DIR}/deps/mel_spec/build/libaudio_stream.so"
    "${CMAKE_SOURCE_DIR}/deps/mel_spec/build/libmelspec.so"
)

if(LibVNCServer_FOUND)
    target_link_libraries(${PROJECT_NAME} LibVNCServer::vncclient)
    target_link_libraries(${PROJECT_NAME} LibVNCServer::vncserver)
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
