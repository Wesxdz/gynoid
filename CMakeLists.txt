# Default Paphos project starter
# flecs/PyTorch with AOT Inductor pipeline

cmake_minimum_required(VERSION 3.18)
project(project)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# add_definitions(AOTInductor_CUDA)
add_executable(${PROJECT_NAME} src/main.cpp src/mel_spec_render.cpp src/glad.c src/vision_processor.cpp)
include_directories("src")
include_directories("deps/mel_spec/src")
include_directories("deps/mel_spec/deps/glad/include")
include_directories("deps/spdlog/include")

# Flecs
include_directories("deps/flecs-4.1.1/include")
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build Flecs as a shared library" FORCE)
add_subdirectory("deps/flecs-4.1.1")

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

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LibVNCServer)

target_link_libraries(${PROJECT_NAME} flecs glfw glad nanovg X11 Xi Xtst SDL2 SDL2_image ${OpenCV_LIBS})

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

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
