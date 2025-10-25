# This script uses Python and OpenCV with the FSRCNN model to upscale a video by 4x.
# Note: This process is computationally intensive and will be slow as it runs on the CPU.
# The original plain text steps for setup and execution are included below as comments.

# --- ORIGINAL SETUP INSTRUCTIONS (Commented Out) ---
# This script will:
# Use Python and OpenCV.
# Use FSRCNN (a fast spatial model similar to FSR 1.0) because it's readily available for OpenCV.
# Read a video file named input.mp4.
# Upscale it 4x.
# Save the new video as output_4x.mp4.
# This script will be slow, as we discussed, but it will work and prove the concept.

# Step 1: Install OpenCV
# If you haven't already, open your terminal and install the opencv-contrib-python package:
# pip install opencv-contrib-python

# Step 2: Download the AI Model
# You need the pre-trained FSRCNN model. Run this command in your terminal to download it.
# On macOS/Linux:
# curl -L https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb -o FSRCNN_x4.pb

# On Windows (in PowerShell):
# Invoke-WebRequest -Uri "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb" -OutFile "FSRCNN_x4.pb"
# ---------------------------------------------------

import cv2
import os
import time

# --- 1. Configuration ---
INPUT_VIDEO = "Funniest Cat Videos Compilation in 2 Minute - twominutes (480, h264).mp4" 	 # Your source video file
OUTPUT_VIDEO = "output_4x.mp4"	 # The upscaled file that will be created
MODEL_FILE = "FSRCNN_x4.pb" 	 # The model file you must download
MODEL_NAME = "fsrcnn"
MODEL_SCALE = 4 # The upscaling factor

# --- 2. Check for Files ---
if not os.path.exists(INPUT_VIDEO):
	print(f"Error: Input video not found at '{INPUT_VIDEO}'. Please name your video 'input.mp4'.")
	exit()

if not os.path.exists(MODEL_FILE):
	print(f"Error: Model not found at '{MODEL_FILE}'. Please download the 'FSRCNN_x4.pb' file.")
	exit()

# --- 3. Initialize the Model ---
print(f"Loading model '{MODEL_FILE}'...")
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(MODEL_FILE)
sr.setModel(MODEL_NAME, MODEL_SCALE)
print("Model loaded successfully.")

# --- 4. Open Video Streams and Configure Writer ---
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
	print(f"Error: Could not open video stream for '{INPUT_VIDEO}'.")
	exit()

# Get original video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate new dimensions
new_width = original_width * MODEL_SCALE
new_height = original_height * MODEL_SCALE

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' codec for compatibility
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (new_width, new_height))

if not writer.isOpened():
	print(f"Error: Could not open video writer for '{OUTPUT_VIDEO}'.")
	cap.release()
	exit()

print(f"Processing '{INPUT_VIDEO}' ({original_width}x{original_height}) at {fps:.2f} FPS.")
print(f"Outputting to '{OUTPUT_VIDEO}' ({new_width}x{new_height})...")
print("This will be VERY SLOW as it's running on the CPU.")

# --- 5. Process the Video (Frame by Frame) ---
start_time = time.time()
processed_frames = 0

while True:
	ret, frame = cap.read()

	# If the frame was not read correctly, we're at the end of the video
	if not ret:
		break

	# This is the core upscaling operation
	upscaled_frame = sr.upsample(frame)

	# Write the upscaled frame to the output file
	writer.write(upscaled_frame)

	processed_frames += 1

	# Print a progress update every 10 frames
	if processed_frames % 10 == 0:
		print(f"	 ... processed {processed_frames} / {frame_count} frames.")

# --- 6. Clean Up ---
end_time = time.time()
total_time = end_time - start_time
# Calculate average FPS, preventing division by zero
avg_fps = processed_frames / total_time if total_time > 0 and processed_frames > 0 else 0.0

print("\n--- Done! ---")
print(f"Successfully created '{OUTPUT_VIDEO}'.")
print(f"Processed {processed_frames} frames in {total_time:.2f} seconds.")
print(f"Average processing speed: {avg_fps:.2f} FPS.")

cap.release()
writer.release()
cv2.destroyAllWindows()

# --- ORIGINAL EXECUTION INSTRUCTIONS (Commented Out) ---
# How to Run It
# Make sure you have opencv-contrib-python installed.
# Make sure FSRCNN_x4.pb is in the same folder.
# Get a video file (e.g., a short 10-second clip) and name it input.mp4.
# Run the script from your terminal:
# python upscale.py
# ---------------------------------------------------