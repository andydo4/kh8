// src/components/VideoPlayer.tsx
import React, { useEffect, useRef, useState } from 'react';

interface VideoPlayerProps {
  originalVideoFile: File; // Keep original file for name/info if needed
  upscaledSrc: string | null; // URL of the processed video from server
  onBack: () => void;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ originalVideoFile, upscaledSrc, onBack }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [resolution, setResolution] = useState<{width: number, height: number} | null>(null);

  useEffect(() => {
    const videoElement = videoRef.current;
    if (videoElement && upscaledSrc) { // Use the upscaledSrc
      console.log("Setting video source:", upscaledSrc); // Debug log
      videoElement.src = upscaledSrc; // Set src to the server URL

      const handleMetadata = () => {
        if (videoElement) {
            console.log("Metadata loaded:", videoElement.videoWidth, videoElement.videoHeight); // Debug log
          setResolution({
            width: videoElement.videoWidth,
            height: videoElement.videoHeight,
          });
        }
      };
      // Make sure event listener is added *before* potentially playing
      videoElement.addEventListener('loadedmetadata', handleMetadata);
      // Optional: Try to play automatically once metadata is loaded
      // videoElement.play().catch(e => console.error("Autoplay failed:", e));

      return () => { // Cleanup
        if (videoElement) {
          videoElement.removeEventListener('loadedmetadata', handleMetadata);
          videoElement.src = ''; // Clear source on unmount/change
        }
        setResolution(null);
      };
    } else if (videoElement) {
        // Clear src if upscaledSrc is null/empty
        videoElement.src = '';
    }
  }, [upscaledSrc]); // Depend ONLY on the upscaled source URL

  const handleDownload = () => {
    if (!upscaledSrc) return;
    const a = document.createElement('a');
    a.href = upscaledSrc;
    a.download = `upscaled_${originalVideoFile.name}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="w-full max-w-6xl flex flex-col items-center gap-4 p-4"> {/* Added padding */}
      <h2 className="text-xl text-gray-300">Upscaled: {originalVideoFile.name}</h2>
      {resolution && (
        <p className="text-sm text-gray-400">
          Output Resolution: {resolution.width} x {resolution.height}
        </p>
      )}

      {/* Show loading/message until source is ready and metadata loaded */}
      {!upscaledSrc && <p className="text-yellow-400">Waiting for video stream...</p>}
      {upscaledSrc && !resolution && <p className="text-blue-400">Loading video metadata...</p>}

      <video
          ref={videoRef}
          controls
          width="100%"
          className={`rounded-lg shadow-lg bg-black ${!upscaledSrc ? 'hidden' : 'block'}`} // Hide until src is set
          preload="metadata" // Help browser get metadata quickly
      >
        Your browser does not support the video tag.
      </video>
      <div className='flex flex-wrap justify-center gap-4 md:gap-6 mt-2'> {/* Added wrap and responsive gap */}
        <button
          onClick={handleDownload}
          disabled={!upscaledSrc || !resolution} // Disable until metadata is loaded too
          className="bg-[var(--blue)] px-6 py-2 text-lg rounded-full font-semibold text-white hover:bg-opacity-80 transition-opacity duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Download Upscaled
        </button>
        <button
          onClick={onBack}
          className="bg-gray-500 px-6 py-2 text-lg rounded-full font-semibold text-white hover:bg-gray-600 transition-colors duration-150"
        >
          Upload New Video
        </button>
      </div>
    </div>
  );
};

export default VideoPlayer;