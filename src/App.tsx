import { useState, useEffect } from "react";
import "./App.css";
import Logo from "./components/Logo";
import FileDropzone from "./components/FileDropzone";
import VideoPlayer from "./components/VideoPlayer";
import { io, Socket } from "socket.io-client";

const SERVER_URL = "http://129.212.191.158:5000"; 

function App() {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [view, setView] = useState<'upload' | 'player'>('upload');
  const [showFileInfo, setShowFileInfo] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [originalResolution, setOriginalResolution] = useState<{width: number, height: number} | null>(null);
  
  // Effect to connect WebSocket
  useEffect(() => {
    // Connect to the Socket.IO server
    const newSocket = io(SERVER_URL);
    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('Connected to backend via WebSocket:', newSocket.id);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from backend WebSocket');
    });

    // Listen for completion messages
    newSocket.on('upscale_complete', (data) => {
      console.log('Upscale complete:', data);
      setIsUploading(false); // Stop loading indicator
      // TODO: Handle the result_url (e.g., pass it to VideoPlayer to display)
      // Maybe switch view or update VideoPlayer state here
       setView('player'); // Switch view on completion
    });

     newSocket.on('upscale_error', (data) => {
       console.error('Upscale error:', data);
       setUploadError(`Backend Error: ${data.message || 'Unknown error'}`);
       setIsUploading(false); // Stop loading indicator
     });

    // Cleanup on unmount
    return () => {
      newSocket.disconnect();
    };
  }, []); // Run only once on mount

  const handleVideoSelect = (file: File) => {
    console.log("Valid MP4 file selected:", file.name);
    setSelectedVideo(file);
    setOriginalResolution(null);
    setView('upload');
    setTimeout(() => setShowFileInfo(true), 50);

    const video = document.createElement('video');
    video.preload = 'metadata';
    video.onloadedmetadata = function() {
      window.URL.revokeObjectURL(video.src);
      setOriginalResolution({width: video.videoWidth, height: video.videoHeight });
    }
    video.src = URL.createObjectURL(file);
    // You can now do something with the file, like preparing it for upload
    // or passing it to your Python backend via IPC.
  };

  const handleVideoRemove = () => {
    console.log("Selected file removed.");
    setShowFileInfo(false);
    setSelectedVideo(null);
    setOriginalResolution(null);
    setView('upload');
  }

  const handleUpscaleClick = async () => { // Make async
    if (!selectedVideo || isUploading) return;

    setUploadError(null); // Clear previous errors
    setIsUploading(true); // Start loading indicator
    console.log("Starting upload & upscale process for:", selectedVideo.name);

    const formData = new FormData();
    formData.append('video', selectedVideo); // 'video' must match Flask backend

    try {
      const response = await fetch(`${SERVER_URL}/upload`, {
        method: 'POST',
        body: formData,
        // No 'Content-Type' header needed for FormData, browser sets it
      });

      const result = await response.json();

      if (!response.ok) {
        // Handle HTTP errors (e.g., 400, 500)
        throw new Error(result.error || `HTTP error! status: ${response.status}`);
      }

      console.log('Upload successful, backend processing:', result);
      // Don't set isUploading false here, wait for WebSocket message

    } catch (error: any) {
      console.error("Upload failed:", error);
      setUploadError(`Upload Failed: ${error.message}`);
      setIsUploading(false); // Stop loading on upload failure
    }
  };

  const handleBackToUpload = () => {
    setSelectedVideo(null);
    setOriginalResolution(null);
    setView('upload');
  }

  useEffect(() => {
    if (!selectedVideo || view !== 'upload') {
      setShowFileInfo(false);
    }
  }, [selectedVideo, view]);

// Show input resolution
// Ask for user input on what scalar multiplier they want (2 or 4) (dropdown)
// for now just keep it 2

  return (
    <>
    <section className="flex flex-col gap-4 items-center justify-center"> {/* Added min-h-screen */}
        <div className="flex items-center justify-center gap-2"> 
          <Logo color="var(--blue)" className="w-11 h-11" />
          {/* Corrected h1 using arbitrary value */}
          <h1 className="text-(--blue) font-semibold text-3xl">Bluescale</h1>
        </div>

        {/* --- Conditional Rendering Upload View --- */}
        {view === 'upload' && (
          <>
            <FileDropzone
              onFileSelect={handleVideoSelect}
              onFileRemove={handleVideoRemove}
              className=""
            />
            {/* --- Wrapper Div for Transition --- */}
            <div
              className={`
                transition-all duration-300 ease-out overflow-hidden
                ${showFileInfo && selectedVideo ? 'opacity-100 translate-y-0 max-h-40 mt-4' : 'opacity-0 -translate-y-2 max-h-0 mt-0'}
              `}
            >
              {selectedVideo && ( // Keep inner check for safety
                <div className="flex flex-col items-center gap-4">
                  <p className="text-gray-300">File ready: {selectedVideo.name}</p>
                  {originalResolution && <p className="text-xs text-gray-400">Resolution: {originalResolution.width}x{originalResolution.height}</p>}
                  <button
                    onClick={handleUpscaleClick}
                    disabled={isUploading}
                    // Corrected button background and hover
                    className="bg-(--blue) px-8 py-2 text-xl rounded-full font-semibold text-white hover:bg-blue-600 transition-colors duration-150"
                  >
                    {isUploading ? 'Processing...' : 'Upscale'} {/* Change text */}
                  </button>
                  {uploadError && <p className="text-red-500 text-sm mt-2">{uploadError}</p>}
                </div>
              )}
            </div>
            {/* --- End Wrapper Div --- */}
          </>
        )}

        {/* --- Conditional Rendering Player View --- */}
        {view === 'player' && selectedVideo && (
          <VideoPlayer videoFile={selectedVideo} onBack={handleBackToUpload} />
        )}
      </section>
    </>
  );
}

export default App;