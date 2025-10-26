#!/bin/bash
# Test script to verify FSR2 pipeline setup

echo "================================================"
echo "FSR2 Pipeline Setup Verification"
echo "================================================"

ERRORS=0
WARNINGS=0

# Check Python
echo ""
echo "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found"
    ERRORS=$((ERRORS + 1))
else
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "✓ $PYTHON_VERSION"
fi

# Check required Python packages
echo ""
echo "Checking Python packages..."

# Check OpenCV
if python3 -c "import cv2" 2>/dev/null; then
    CV_VERSION=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null)
    echo "✓ opencv (cv2): $CV_VERSION"

    # Check if contrib modules available
    if python3 -c "import cv2; cv2.optflow" 2>/dev/null; then
        echo "  ✓ opencv-contrib modules available"
    else
        echo "  ⚠ opencv-contrib not found (optional, using Farneback instead)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "❌ opencv (cv2) not found"
    echo "  Install: pip install opencv-contrib-python"
    ERRORS=$((ERRORS + 1))
fi

# Check NumPy
if python3 -c "import numpy" 2>/dev/null; then
    NP_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
    echo "✓ numpy: $NP_VERSION"
else
    echo "❌ numpy not found"
    echo "  Install: pip install numpy"
    ERRORS=$((ERRORS + 1))
fi

# Check PyTorch (optional but listed in requirements)
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo "✓ torch: $TORCH_VERSION"

    # Check CUDA/ROCm availability
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo "  ✓ ROCm/CUDA GPU detected: $GPU_NAME"
    else
        echo "  ⚠ No GPU detected in PyTorch (CPU only)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "⚠ torch not found (optional)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check CMake
echo ""
echo "Checking build tools..."
if ! command -v cmake &> /dev/null; then
    echo "❌ cmake not found"
    echo "  Install: sudo apt install cmake"
    ERRORS=$((ERRORS + 1))
else
    CMAKE_VERSION=$(cmake --version | head -n1)
    echo "✓ $CMAKE_VERSION"

    # Check version is 3.20+
    CMAKE_VER_NUM=$(cmake --version | head -n1 | grep -oP '\d+\.\d+' | head -n1)
    if command -v bc &> /dev/null && [ "$(echo "$CMAKE_VER_NUM >= 3.20" | bc -l 2>/dev/null)" == "1" ]; then
        echo "  ✓ Version 3.20+ required for FSR2"
    else
        echo "  ⚠ Version $CMAKE_VER_NUM may be too old (need 3.20+)"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

# Check C++ compiler
if ! command -v g++ &> /dev/null; then
    echo "❌ g++ not found"
    echo "  Install: sudo apt install build-essential"
    ERRORS=$((ERRORS + 1))
else
    GCC_VERSION=$(g++ --version | head -n1)
    echo "✓ $GCC_VERSION"
fi

# Check make
if ! command -v make &> /dev/null; then
    echo "❌ make not found"
    echo "  Install: sudo apt install build-essential"
    ERRORS=$((ERRORS + 1))
else
    MAKE_VERSION=$(make --version | head -n1)
    echo "✓ $MAKE_VERSION"
fi

# Check Vulkan
echo ""
echo "Checking Vulkan..."
if ! command -v vulkaninfo &> /dev/null; then
    echo "❌ vulkaninfo not found"
    echo "  Install: sudo apt install vulkan-tools libvulkan-dev"
    ERRORS=$((ERRORS + 1))
else
    echo "✓ vulkaninfo found"

    # Check for Vulkan devices
    DEVICE_COUNT=$(vulkaninfo --summary 2>/dev/null | grep -c "deviceName" || echo "0")
    if [ "$DEVICE_COUNT" -gt 0 ]; then
        echo "✓ Vulkan devices found: $DEVICE_COUNT"
        vulkaninfo --summary 2>/dev/null | grep "deviceName" | sed 's/^/  /' || echo "  (could not list devices)"

        # Check API version
        API_VERSION=$(vulkaninfo --summary 2>/dev/null | grep "apiVersion" | head -n1 | awk '{print $3}' || echo "unknown")
        echo "  API Version: $API_VERSION"
    else
        echo "❌ No Vulkan devices found"
        echo "  Check GPU drivers are installed"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Check for Vulkan development files
if [ -f "/usr/include/vulkan/vulkan.h" ] || [ -f "/usr/local/include/vulkan/vulkan.h" ]; then
    echo "✓ Vulkan headers found"
else
    echo "❌ Vulkan headers not found"
    echo "  Install: sudo apt install libvulkan-dev"
    ERRORS=$((ERRORS + 1))
fi

# Check FFmpeg
echo ""
echo "Checking FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠ ffmpeg not found (needed for final video encoding)"
    echo "  Install: sudo apt install ffmpeg"
    WARNINGS=$((WARNINGS + 1))
else
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | cut -d' ' -f3)
    echo "✓ ffmpeg: $FFMPEG_VERSION"
fi

if ! command -v ffprobe &> /dev/null; then
    echo "⚠ ffprobe not found (used for video info)"
    WARNINGS=$((WARNINGS + 1))
else
    echo "✓ ffprobe found"
fi

# Check FidelityFX SDK
echo ""
echo "Checking FidelityFX SDK..."
if [ -f "CMakeLists.txt" ]; then
    echo "✓ CMakeLists.txt found"

    # Extract FFX SDK path
    FFX_PATH=$(grep "set(FFX_SDK_DIR" CMakeLists.txt | cut -d'"' -f2 2>/dev/null)

    if [ -z "$FFX_PATH" ]; then
        echo "❌ Could not find FFX_SDK_DIR in CMakeLists.txt"
        ERRORS=$((ERRORS + 1))
    elif [ -d "$FFX_PATH" ]; then
        echo "✓ FidelityFX SDK path: $FFX_PATH"

        # Check for FSR2 headers
        if [ -f "$FFX_PATH/sdk/include/FidelityFX/host/ffx_fsr2.h" ]; then
            echo "  ✓ FSR2 host headers found"
        else
            echo "  ❌ FSR2 headers not found at expected location"
            echo "    Expected: $FFX_PATH/sdk/include/FidelityFX/host/ffx_fsr2.h"
            ERRORS=$((ERRORS + 1))
        fi

        # Check for Vulkan backend
        if [ -f "$FFX_PATH/sdk/include/FidelityFX/host/backends/vk/ffx_vk.h" ]; then
            echo "  ✓ FSR2 Vulkan backend found"
        else
            echo "  ⚠ FSR2 Vulkan backend not found at expected location"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo "❌ FidelityFX SDK not found at: $FFX_PATH"
        echo "  Clone v1.1.2 with: git clone --branch v1.1.2 https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git FidelityFX-SDK-v1"
        echo "  Then update FFX_SDK_DIR in CMakeLists.txt"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "❌ CMakeLists.txt not found in current directory"
    ERRORS=$((ERRORS + 1))
fi

# Check input video
echo ""
echo "Checking input files..."
if [ -f "input480.mp4" ]; then
    SIZE=$(du -h input480.mp4 | cut -f1)
    echo "✓ input480.mp4 found ($SIZE)"

    if command -v ffprobe &> /dev/null; then
        RES=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 input480.mp4 2>/dev/null)
        FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 input480.mp4 2>/dev/null)
        DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 input480.mp4 2>/dev/null)

        if [ -n "$RES" ]; then
            echo "  Resolution: $RES"
        fi
        if [ -n "$FPS" ]; then
            echo "  Frame rate: $FPS"
        fi
        if [ -n "$DURATION" ]; then
            echo "  Duration: ${DURATION}s"
        fi
    fi
else
    echo "⚠ input480.mp4 not found"
    echo "  This file is needed to run the pipeline"
    echo "  Copy your input video as: input480.mp4"
    WARNINGS=$((WARNINGS + 1))
fi

# Check required source files
echo ""
echo "Checking source files..."
REQUIRED_FILES=("run_pipeline.py" "main.cpp" "CMakeLists.txt")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "❌ $file missing"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check stb headers
echo ""
echo "Checking third-party libraries..."
if [ -d "third_party" ]; then
    echo "✓ third_party/ directory exists"

    if [ -f "third_party/stb_image.h" ]; then
        echo "  ✓ stb_image.h found"
    else
        echo "  ❌ stb_image.h missing"
        echo "    Download: wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -O third_party/stb_image.h"
        ERRORS=$((ERRORS + 1))
    fi

    if [ -f "third_party/stb_image_write.h" ]; then
        echo "  ✓ stb_image_write.h found"
    else
        echo "  ❌ stb_image_write.h missing"
        echo "    Download: wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h -O third_party/stb_image_write.h"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "❌ third_party/ directory not found"
    echo "  Create with: mkdir -p third_party"
    echo "  Then download stb headers"
    ERRORS=$((ERRORS + 1))
fi

# Check disk space
echo ""
echo "Checking system resources..."
AVAILABLE=$(df -h . | awk 'NR==2 {print $4}')
echo "Disk space available: $AVAILABLE"

if [ -f "input480.mp4" ]; then
    VIDEO_SIZE=$(stat --format=%s "input480.mp4" 2>/dev/null || stat -f%z "input480.mp4" 2>/dev/null || echo "0")
    if [ "$VIDEO_SIZE" -gt 0 ]; then
        NEEDED_MB=$((VIDEO_SIZE * 5 / 1024 / 1024))
        echo "Estimated space needed: ~${NEEDED_MB}MB (5x video size)"

        # Check if enough space (rough check)
        AVAIL_KB=$(df . | awk 'NR==2 {print $4}')
        NEEDED_KB=$((NEEDED_MB * 1024))
        if [ "$AVAIL_KB" -lt "$NEEDED_KB" ]; then
            echo "⚠ May not have enough disk space"
            WARNINGS=$((WARNINGS + 1))
        fi
    fi
fi

# Check RAM
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -h | awk 'NR==2 {print $2}')
    AVAIL_RAM=$(free -h | awk 'NR==2 {print $7}')
    echo "RAM Total: $TOTAL_RAM"
    echo "RAM Available: $AVAIL_RAM"

    # Warn if less than 8GB available
    AVAIL_RAM_GB=$(free -g | awk 'NR==2 {print $7}')
    if [ "$AVAIL_RAM_GB" -lt 8 ]; then
        echo "⚠ Less than 8GB RAM available - may need to reduce NUM_WORKERS"
        WARNINGS=$((WARNINGS + 1))
    fi
elif command -v vm_stat &> /dev/null; then
    # macOS
    echo "RAM info: (use Activity Monitor for details)"
else
    echo "RAM: (cannot determine)"
fi

# Summary
echo ""
echo "================================================"
echo "SUMMARY"
echo "================================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✓✓✓ All checks passed! ✓✓✓"
    echo ""
    echo "Ready to run the FSR2 pipeline!"
    echo ""
    echo "Quick start:"
    echo "  ./build_and_run.sh"
    echo ""
    echo "Or run steps manually:"
    echo "  1. python3 run_pipeline.py"
    echo "  2. mkdir -p build && cd build && cmake .. && make -j\$(nproc) && cd .."
    echo "  3. ./build/fsr2_vulkan_min outputs_fsr2 upscaled"
    echo "  4. ffmpeg -i upscaled/%05d.png -c:v libx264 -crf 16 upscaled_output.mp4"

elif [ $ERRORS -eq 0 ]; then
    echo "✓ All required checks passed"
    echo "⚠ $WARNINGS warning(s) - pipeline should still work"
    echo ""
    echo "You can proceed with:"
    echo "  ./build_and_run.sh"

else
    echo "❌ Found $ERRORS error(s) and $WARNINGS warning(s)"
    echo ""
    echo "Please fix the errors before running the pipeline."
    echo ""
    echo "Common fixes:"
    echo ""
    echo "Python packages:"
    echo "  pip install opencv-contrib-python numpy torch"
    echo ""
    echo "Build tools:"
    echo "  sudo apt install build-essential cmake"
    echo ""
    echo "Vulkan:"
    echo "  sudo apt install vulkan-tools libvulkan-dev"
    echo ""
    echo "FFmpeg:"
    echo "  sudo apt install ffmpeg"
    echo ""
    echo "FidelityFX SDK v1.1.2:"
    echo "  git clone --branch v1.1.2 https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git /home/amd-knights/code/FidelityFX-SDK-v1"
    echo "  Then update CMakeLists.txt with the path"
    echo ""
    echo "STB headers:"
    echo "  mkdir -p third_party"
    echo "  wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -O third_party/stb_image.h"
    echo "  wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h -O third_party/stb_image_write.h"
fi

echo "================================================"
echo ""

exit $ERRORS