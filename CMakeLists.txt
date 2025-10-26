cmake_minimum_required(VERSION 3.20)
project(fsr2_vulkan_min CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- EDIT THIS: path to your FidelityFX SDK clone ----
#   git clone https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git
set(FFX_SDK_DIR "/abs/path/to/FidelityFX-SDK")

find_package(Vulkan REQUIRED)

add_executable(fsr2_vulkan_min main.cpp
  third_party/stb_image.h
  third_party/stb_image_write.h
)

target_include_directories(fsr2_vulkan_min PRIVATE
  ${Vulkan_INCLUDE_DIRS}
  ${FFX_SDK_DIR}/sdk/include          # FFX host APIs (ffx_fsr2.h, etc.)
  ${FFX_SDK_DIR}/sdk/src/backends/vk  # Vulkan backend helpers
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(fsr2_vulkan_min PRIVATE
  ${Vulkan_LIBRARIES}
)

# On Linux you may also need pthread and dl
if(UNIX AND NOT APPLE)
  target_link_libraries(fsr2_vulkan_min PRIVATE pthread dl)
endif()

# Define to make stb headers compile in this TU
target_compile_definitions(fsr2_vulkan_min PRIVATE
  STB_IMAGE_IMPLEMENTATION
  STB_IMAGE_WRITE_IMPLEMENTATION
)
