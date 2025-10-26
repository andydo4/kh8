cmake_minimum_required(VERSION 3.20)
project(fsr2_vulkan_min CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ✅ Path to the FidelityFX SDK (inside your repo)
set(FFX_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/FidelityFX-SDK-v1")

find_package(Vulkan REQUIRED)

# ✅ Include the actual FSR2 source files so the linker finds ffx* symbols
file(GLOB FFX_SOURCES
  "${FFX_SDK_DIR}/sdk/src/ffx_fsr2.cpp"
  "${FFX_SDK_DIR}/sdk/src/backends/vk/ffx_vk.cpp"
)

# ✅ Build the executable including FSR2 source files
add_executable(fsr2_vulkan_min main.cpp ${FFX_SOURCES})

# ✅ Include directories
target_include_directories(fsr2_vulkan_min PRIVATE
  ${Vulkan_INCLUDE_DIRS}
  ${FFX_SDK_DIR}/sdk/include
  ${FFX_SDK_DIR}/sdk/src/backends/vk
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# ✅ Link Vulkan and system libs
target_link_libraries(fsr2_vulkan_min PRIVATE
  ${Vulkan_LIBRARIES}
)

if(UNIX AND NOT APPLE)
  target_link_libraries(fsr2_vulkan_min PRIVATE pthread dl)
endif()

# ✅ Preprocessor definitions for stb
target_compile_definitions(fsr2_vulkan_min PRIVATE
  STB_IMAGE_IMPLEMENTATION
  STB_IMAGE_WRITE_IMPLEMENTATION
)
