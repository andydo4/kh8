
cmake_minimum_required(VERSION 3.20)
project(fsr2_vulkan_min CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FFX_SDK_DIR "/home/amd-knights/code/FidelityFX-SDK-v1")

find_package(Vulkan REQUIRED)

add_executable(fsr2_vulkan_min main.cpp)

target_include_directories(fsr2_vulkan_min PRIVATE
  ${Vulkan_INCLUDE_DIRS}
  ${FFX_SDK_DIR}/sdk/include
  ${FFX_SDK_DIR}/sdk/src/backends/vk
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
)