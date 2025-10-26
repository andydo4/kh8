find_package(Vulkan REQUIRED)

set(FSR2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/FSR2-standalone")

file(GLOB FSR2_SOURCES
  "${FSR2_DIR}/src/ffx_fsr2.cpp"
  "${FSR2_DIR}/src/ffx_vk.cpp"
)

add_executable(fsr2_vulkan_min
  main.cpp
  ${FSR2_SOURCES}
)

target_include_directories(fsr2_vulkan_min PRIVATE
  ${Vulkan_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${FSR2_DIR}/include
  ${FSR2_DIR}/src
  ${FSR2_DIR}/build/src/ffx_vk   # generated shader header lives here
)

target_link_libraries(fsr2_vulkan_min PRIVATE ${Vulkan_LIBRARIES} pthread dl)
