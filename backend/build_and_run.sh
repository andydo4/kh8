#!/bin/bash
# Complete pipeline to upscale video with FSR2

set -e  # Exit on error

echo "================================================"
echo "FSR2 Video Upscaling Pipeline"
echo "================================================"

# Step 1: Generate FSR2 input data
echo ""
echo "Step 1: Generating FSR2 input data..."
echo "---------------------------------------"
python3 run_pipeline.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate input data"
    exit 1
fi

# Check that outputs exist
if [ ! -d "outputs_fsr2/color" ]; then
    echo "❌ Output directory not created"
    exit 1
fi

FRAME_COUNT=$(ls outputs_fsr2/color/*.png 2>/dev/null | wc -l)
echo "✓ Generated $FRAME_COUNT frames of input data"

# Step 2: Build FSR2 upscaler
echo ""
echo "Step 2: Building FSR2 upscaler..."
echo "---------------------------------------"

mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed"
    exit 1
fi

cmake --build . -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✓ Build successful"
cd ..

# Step 3: Run FSR2 upscaler
echo ""
echo "Step 3: Running FSR2 upscaler..."
echo "---------------------------------------"

./build/fsr2_vulkan_min outputs_fsr2 upscaled

if [ $? -ne 0 ]; then
    echo "❌ FSR2 upscaling failed"
    exit 1
fi

UPSCALED_COUNT=$(ls upscaled/*.png 2>/dev/null | wc -l)
echo "✓ Generated $UPSCALED_COUNT upscaled frames"

# Step 4: Encode final video
echo ""
echo "Step 4: Encoding final video..."
echo "---------------------------------------"

if ! command -v ffmpeg &> /dev/null; then
    echo "⚠ ffmpeg not found. Install with: sudo apt install ffmpeg"
    echo "To manually encode: ffmpeg -y -framerate 30 -i upscaled/%05d.png -c:v libx264 -crf 16 -pix_fmt yuv420p upscaled_output.mp4"
    exit 0
fi

FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 input480.mp4 2>/dev/null | bc -l 2>/dev/null)
if [ -z "$FPS" ]; then
    FPS=30
    echo "⚠ Could not detect FPS, using 30"
fi

echo "Encoding at ${FPS} FPS..."

ffmpeg -y -framerate $FPS -i upscaled/%05d.png \
    -c:v libx264 \
    -preset slow \
    -crf 16 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    upscaled_output.mp4

if [ $? -ne 0 ]; then
    echo "❌ Video encoding failed"
    exit 1
fi

echo ""
echo "================================================"
echo "✓ SUCCESS!"
echo "================================================"
echo "Output video: upscaled_output.mp4"
echo ""
echo "Original: $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 input480.mp4 2>/dev/null)"
echo "Upscaled: $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 upscaled_output.mp4 2>/dev/null)"
echo ""
echo "================================================"