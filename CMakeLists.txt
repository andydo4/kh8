cmake_minimum_required(VERSION 3.20)
project(fsr2_vulkan_min CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FFX_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/FidelityFX-SDK-v1")

if(NOT EXISTS "${FFX_SDK_DIR}/sdk")
    message(FATAL_ERROR "FidelityFX SDK not found at ${FFX_SDK_DIR}")
endif()

find_package(Vulkan REQUIRED)

# Collect all FSR2 backend sources
file(GLOB FFX_VK_BACKEND_SOURCES
    "${FFX_SDK_DIR}/sdk/src/backends/vk/*.cpp"
)

file(GLOB FFX_SHARED_BACKEND_SOURCES
    "${FFX_SDK_DIR}/sdk/src/backends/shared/*.cpp"
)

file(GLOB FFX_FSR2_SOURCES
    "${FFX_SDK_DIR}/sdk/src/components/fsr2/*.cpp"
)

add_executable(fsr2_vulkan_min
    main.cpp
    ${FFX_VK_BACKEND_SOURCES}
    ${FFX_SHARED_BACKEND_SOURCES}
    ${FFX_FSR2_SOURCES}
)

target_include_directories(fsr2_vulkan_min PRIVATE
  ${Vulkan_INCLUDE_DIRS}
  ${FFX_SDK_DIR}/sdk/include
  ${FFX_SDK_DIR}/sdk/src/backends/vk
  ${FFX_SDK_DIR}/sdk/src/backends/shared
  ${FFX_SDK_DIR}/sdk/src/components
  ${FFX_SDK_DIR}/sdk/build/src/backends/shaders/vk
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(fsr2_vulkan_min PRIVATE
  ${Vulkan_LIBRARIES}
)

if(UNIX AND NOT APPLE)
  target_link_libraries(fsr2_vulkan_min PRIVATE pthread dl)
endif()

target_compile_definitions(fsr2_vulkan_min PRIVATE
  STB_IMAGE_IMPLEMENTATION
  STB_IMAGE_WRITE_IMPLEMENTATION
  FFX_CPU
)