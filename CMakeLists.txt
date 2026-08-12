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
add_executable(${PROJECT_NAME} src/main.cpp src/mel_spec_render.cpp src/glad.c src/vision_processor.cpp src/x11_outline.cpp src/dino_embedder.cpp src/thorn_vfx.cpp src/tradewinds.cpp src/stl_mesh.cpp src/panel3d.cpp src/screen3d.cpp)
# deps/tracy-0.13.1/public/TracyClient.cpp

target_include_directories(${PROJECT_NAME} PRIVATE "deps/cppzmq")

include_directories(deps/msgpack-c/include)
add_subdirectory("deps/msgpack-c")

# build libzmq from source
set(WITH_DOC OFF CACHE BOOL "" FORCE)
set(ZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED ON CACHE BOOL "" FORCE)
set(BUILD_STATIC OFF CACHE BOOL "" FORCE)
add_subdirectory("deps/libzmq")
target_include_directories(${PROJECT_NAME} PRIVATE "deps/libzmq/include")

include_directories("src")
include_directories("deps/mel_spec/src")
include_directories("deps/dinov2.cpp")
include_directories("deps/dinov2.cpp/ggml/include")
include_directories("deps/mel_spec/deps/glad/include")
include_directories("deps/spdlog/include")
include_directories("deps/mel_spec/deps/fpng/src")
include_directories("deps/json-3.11.2/single_include")
include_directories("deps/enet-1.3.18/include")

add_definitions(-DTRACY_ENABLE)
add_subdirectory("deps/tracy-0.13.1")

# Yoga (flexbox layout). Point at the yoga/ subdir rather than the repo root so
# we skip its tests target (which wants gtest). yogacore exports the repo root
# as a PUBLIC include dir, so <yoga/Yoga.h> resolves for anything linking it.
add_subdirectory("deps/yoga-3.2.1/yoga")

# Flecs
include_directories("deps/flecs-4.1.1/include")
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build Flecs as a shared library" FORCE)
add_subdirectory("deps/flecs-4.1.1")

# Flecs custom build configuration
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
add_subdirectory("deps/enet-1.3.18")

include_directories("deps/raylib")

# GLAD. glad_compat/include supplies <glad/gl.h> for render3d, which expects the
# glad 2.x header name, plus the two instancing entrypoints this 0.1.36 gl=3.0
# loader omits -- see deps/glad_compat/include/glad/gl.h.
add_library(glad STATIC
    deps/glad/src/glad.c
    deps/glad_compat/src/gl_compat.c
)
target_include_directories(glad PUBLIC deps/glad/include deps/glad_compat/include)

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

# mel_spec libraries (audio streaming and mel spectrogram generation)
add_subdirectory("deps/mel_spec")

# glm (header-only). The light distribution has no top-level CMakeLists, so the
# INTERFACE target render3d links against is declared here.
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/deps/glm")

# render3d -- flecs module for the rasterized 3D scene drawn into the Droid
# panel. Links flecs/glfw/glad/glm, all of which exist above.
add_subdirectory("deps/render3d")

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

target_link_libraries(${PROJECT_NAME} flecs enet glfw glad nanovg fpng X11 Xi Xtst SDL2 SDL2_image ${OpenCV_LIBS} query_server Tracy::TracyClient libssh2::libssh2 dinov2 msgpack-cxx libzmq yogacore render3d)

# Add audio stream client executable
add_executable(audio_stream_client src/audio_stream_client.c)
target_link_libraries(audio_stream_client audio_stream melspec)

# Add mel_spec dependencies
target_link_libraries(${PROJECT_NAME} audio_stream melspec)

if(LibVNCServer_FOUND)
    target_link_libraries(${PROJECT_NAME} LibVNCServer::vncclient)
    target_link_libraries(${PROJECT_NAME} LibVNCServer::vncserver)
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
