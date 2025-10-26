import React, { useEffect, useRef, useState } from 'react';

interface VideoPlayerProps {
  videoFile: File;
  onBack: () => void; // Function to go back to upload view
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoFile, onBack }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  // State to store the resolution
  const [originalResolution, setOriginalResolution] = useState<{width: number, height: number} | null>(null);

  // Effect to load video source and metadata
  useEffect(() => {
    const videoElement = videoRef.current;
    if (videoElement) {
      const objectURL = URL.createObjectURL(videoFile);
      videoElement.src = objectURL;

      // --- Event listener to get resolution ---
      const handleMetadata = () => {
        if (videoElement) {
          console.log(`Video dimensions: ${videoElement.videoWidth}x${videoElement.videoHeight}`); // Log dimensions
          setOriginalResolution({
            width: videoElement.videoWidth,
            height: videoElement.videoHeight,
          });
        }
      };

      videoElement.addEventListener('loadedmetadata', handleMetadata);
      // --- End event listener ---

      // Cleanup function
      return () => {
        URL.revokeObjectURL(objectURL);
        if (videoElement) {
          videoElement.removeEventListener('loadedmetadata', handleMetadata); // Remove listener
        }
        setOriginalResolution(null); // Reset resolution on file change/unmount
      };
    }
  }, [videoFile]); // Re-run when videoFile changes

  const handleDownload = () => {
    // Note: This currently downloads the *original* uploaded file,
    // not the upscaled version from the backend.
    const objectURL = URL.createObjectURL(videoFile);
    const a = document.createElement('a');
    a.href = objectURL;
    a.download = `original_${videoFile.name}`; // Indicate it's the original
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(objectURL);
  };

  return (
    <div className="w-full max-w-6xl flex flex-col items-center gap-4">
      <h2 className="text-xl text-gray-300">Upscaling: {videoFile.name}</h2>

      {/* --- Display Original Resolution --- */}
      {originalResolution && (
        <p className="text-sm text-gray-400">
          Original Resolution: {originalResolution.width} x {originalResolution.height}
          {/* You can add the target upscaled resolution here too if needed */}
        </p>
      )}
      {/* --- End Resolution Display --- */}

      <video ref={videoRef} controls width="100%" className="rounded-lg shadow-lg bg-black"> {/* Added bg-black */}
        Your browser does not support the video tag.
      </video>

      <div className='flex gap-6'>
        <button
          onClick={handleDownload}
          // Corrected arbitrary value syntax
          className="mt-4 bg-[var(--blue)] px-6 py-2 text-lg rounded-full font-semibold text-white hover:bg-opacity-80 transition-opacity duration-300"
        >
          Download Original {/* Clarified button purpose */}
        </button>
        <button
          onClick={onBack}
          className="mt-4 bg-black/40 px-6 py-2 text-lg rounded-full font-semibold text-gray-700 hover:text-gray-400 hover:bg-black/30 transition-colors duration-150"
        >
          Upload New Video
        </button>
      </div>
    </div>
  );
};

export default VideoPlayer;