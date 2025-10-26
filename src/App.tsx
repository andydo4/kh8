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
  const [scaleFactor, setScaleFactor] = useState<number>(2); // <-- State for scale factor
  const [uploadProgress, setUploadProgress] = useState<string | null>(null); // <-- State for progress
  const [upscaledVideoUrl, setUpscaledVideoUrl] = useState<string | null>(null); // <-- State for result URL

  useEffect(() => {
    const newSocket = io(SERVER_URL);
    setSocket(newSocket);
    newSocket.on('connect', () => console.log('Connected WS:', newSocket.id));
    newSocket.on('disconnect', () => console.log('Disconnected WS'));

    newSocket.on('upscale_complete', (data) => {
      console.log('Upscale complete:', data);
      setIsUploading(false);
      setUploadProgress(null); // Clear progress
      if (data.result_url) {
        setUpscaledVideoUrl(SERVER_URL + data.result_url); // Store full URL
        setView('player'); // Switch view
      } else {
        setUploadError("Backend Error: Result URL missing.");
      }
    });

    newSocket.on('upscale_error', (data) => {
      console.error('Upscale error:', data);
      setUploadError(`Backend Error: ${data.message || 'Unknown error'}`);
      setIsUploading(false);
      setUploadProgress(null); // Clear progress
    });

    // --- Listen for Progress ---
    newSocket.on('upscale_progress', (data) => {
      // console.log('Progress:', data.progress, 'FPS:', data.fps); // Optional detailed log
      setUploadProgress(`${data.progress}% (${data.fps} FPS)`); // Update progress state
    });
    // --- End Progress Listener ---

    return () => { newSocket.disconnect(); };
  }, []);

  const handleVideoSelect = (file: File) => {
    console.log("Selected:", file.name);
    setSelectedVideo(file);
    setOriginalResolution(null);
    setUploadError(null); // Clear errors
    setUploadProgress(null); // Clear progress
    setView('upload');
    setTimeout(() => setShowFileInfo(true), 50);

    const video = document.createElement('video');
    video.preload = 'metadata';
    video.onloadedmetadata = function() {
      window.URL.revokeObjectURL(video.src);
      setOriginalResolution({width: video.videoWidth, height: video.videoHeight });
    }
    video.src = URL.createObjectURL(file);
  };

  const handleVideoRemove = () => {
    console.log("Removed.");
    setShowFileInfo(false);
    setSelectedVideo(null);
    setOriginalResolution(null);
    setUploadError(null);
    setUploadProgress(null);
    setView('upload');
  };

  const handleUpscaleClick = async () => {
    if (!selectedVideo || isUploading || !socket) return;
    setUploadError(null);
    setUploadProgress("Uploading..."); // Initial progress message
    setIsUploading(true);
    setUpscaledVideoUrl(null); // Clear previous result
    console.log(`Starting upload & upscale (${scaleFactor}x) for:`, selectedVideo.name);

    const formData = new FormData();
    formData.append('video', selectedVideo);
    formData.append('scale', scaleFactor.toString()); // <-- Send scale factor
    // Optional: Send Socket ID if using header method
    // const headers: HeadersInit = {};
    // if (socket.id) { headers['X-SocketIO-SID'] = socket.id; }

    try {
      const response = await fetch(`${SERVER_URL}/upload`, {
        method: 'POST',
        body: formData,
        // headers: headers, // Add headers if using that method
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `HTTP error! status: ${response.status}`);
      console.log('Upload OK, backend processing:', result);
      setUploadProgress("Processing..."); // Update progress
      // Now wait for WebSocket messages for progress/completion
    } catch (error: any) {
      console.error("Upload failed:", error);
      setUploadError(`Upload Failed: ${error.message}`);
      setIsUploading(false);
      setUploadProgress(null);
    }
  };

  const handleBackToUpload = () => {
    setSelectedVideo(null);
    setOriginalResolution(null);
    setUpscaledVideoUrl(null); // Clear result URL
    setUploadError(null);
    setUploadProgress(null);
    setView('upload');
  };

  useEffect(() => {
    if (!selectedVideo || view !== 'upload') {
      setShowFileInfo(false);
    }
  }, [selectedVideo, view]);

  return (
    <>
      <section className="flex flex-col gap-4 items-center justify-center min-h-screen p-4"> {/* Added padding */}
        <div className="flex items-center justify-center gap-2 mb-8">
          <Logo color="var(--blue)" className="w-11 h-11" />
          <h1 className="text-[var(--blue)] font-semibold text-3xl">Bluescale</h1>
        </div>

        {view === 'upload' && (
          <>
            <FileDropzone
              onFileSelect={handleVideoSelect}
              onFileRemove={handleVideoRemove}
              className={isUploading ? 'opacity-50 pointer-events-none' : ''} // Disable dropzone during upload
            />
            <div
              className={`
                transition-all duration-300 ease-out overflow-hidden w-full max-w-lg  // Added width constraint
                ${showFileInfo && selectedVideo ? 'opacity-100 translate-y-0 max-h-60 mt-4' : 'opacity-0 -translate-y-2 max-h-0 mt-0'} // Increased max-h
              `}
            >
              {selectedVideo && (
                <div className="flex flex-col items-center gap-3 p-4 bg-gray-800 rounded-lg"> {/* Added background */}
                  <p className="text-gray-300 text-sm">File ready: {selectedVideo.name}</p>
                  {originalResolution && <p className="text-xs text-gray-400">Resolution: {originalResolution.width}x{originalResolution.height}</p>}

                  {/* --- Scale Factor Selector --- */}
                  {!isUploading && ( // Hide selector during upload
                    <div className="flex items-center gap-2">
                        <label htmlFor="scaleFactor" className="text-sm text-gray-400">Scale:</label>
                        <select
                            id="scaleFactor"
                            value={scaleFactor}
                            onChange={(e) => setScaleFactor(Number(e.target.value))}
                            className="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-[var(--blue)] focus:border-[var(--blue)] p-1.5"
                        >
                            <option value={2}>2x</option>
                            <option value={4}>4x</option>
                        </select>
                    </div>
                  )}
                  {/* --- End Scale Factor --- */}

                  <button
                    onClick={handleUpscaleClick}
                    disabled={isUploading}
                    className="bg-[var(--blue)] px-8 py-2 text-lg rounded-full font-semibold text-white hover:bg-opacity-80 transition-opacity duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isUploading ? (uploadProgress || 'Processing...') : `Upscale (${scaleFactor}x)`}
                  </button>
                  {/* Display Progress if uploading */}
                  {isUploading && uploadProgress && <p className="text-blue-400 text-sm animate-pulse">{uploadProgress}</p>}
                  {uploadError && <p className="text-red-500 text-sm mt-1">{uploadError}</p>}
                </div>
              )}
            </div>
          </>
        )}

        {view === 'player' && selectedVideo && upscaledVideoUrl && ( // Check for URL
          <VideoPlayer
             originalVideoFile={selectedVideo} // Pass original file too
             upscaledSrc={upscaledVideoUrl} // Pass the state URL
             onBack={handleBackToUpload}
           />
        )}
        {/* Handle case where view is player but URL isn't ready yet */}
        {view === 'player' && !upscaledVideoUrl && isUploading && (
             <p className="text-yellow-400 text-lg animate-pulse">Waiting for processing to complete...</p>
        )}
      </section>
    </>
  );
}
export default App;