import { useState, useEffect } from "react";
import "./App.css";
import Logo from "./components/Logo";
import FileDropzone from "./components/FileDropzone";
import VideoPlayer from "./components/VideoPlayer";

function App() {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [view, setView] = useState<'upload' | 'player'>('upload');
  const [showFileInfo, setShowFileInfo] = useState<Boolean>(false);
  const [originalResolution, setOriginalResolution] = useState<{width: number, height: number} | null>(null);
  
  const handleVideoSelect = (file: File) => {
    console.log("Valid MP4 file selected:", file.name);
    setSelectedVideo(file);
    setView('upload');
    setTimeout(() => setShowFileInfo(true), 0);

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
    setView('upload');
  }

  const handleUpscaleClick = () => {
    if (selectedVideo) {
      console.log("Starting upscale process for:", selectedVideo.name);
      // Python backend stuff
      setView('player');
    }
  }

  const handleBackToUpload = () => {
    setSelectedVideo(null);
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
                    // Corrected button background and hover
                    className="bg-(--blue) px-8 py-2 text-xl rounded-full font-semibold text-white hover:bg-blue-600 transition-colors duration-150"
                  >
                    Upscale
                  </button>
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