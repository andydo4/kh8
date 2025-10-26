import React, { useEffect, useRef } from 'react';

interface VideoPlayerProps {
  videoFile: File;
  onBack: () => void; // Function to go back to upload view
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoFile, onBack }) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    // --- HERE: Logic to handle video display ---
    // Option A: If backend processes offline, get the final upscaled video path/URL
    // Option B: If backend streams frames, set up IPC listeners to receive frames
    //           and draw them onto a canvas or update the video source dynamically.
    // Option C (Simple Demo): Display the *original* selected video for now.
    if (videoRef.current) {
        const objectURL = URL.createObjectURL(videoFile);
        videoRef.current.src = objectURL;
        // Optional: Clean up the object URL when the component unmounts
        return () => URL.revokeObjectURL(objectURL);
    }
    // ---

  }, [videoFile]); // Re-run effect if the video file changes

  return (
    <div className="w-full max-w-6xl flex flex-col items-center gap-4">
      <h2 className="text-xl text-gray-300">Upscaling: {videoFile.name}</h2>
      {/* Basic video element - Replace/enhance this based on how you receive frames */}
      <video ref={videoRef} controls width="100%" className="rounded-lg shadow-lg">
        Your browser does not support the video tag.
      </video>
      <button
        onClick={onBack}
        className="mt-4 bg-gray-500 px-6 py-2 text-lg rounded-full font-semibold text-white hover:bg-gray-600 transition-colors duration-150"
      >
        Upload New Video
      </button>
    </div>
  );
};

export default VideoPlayer;