import { useState } from "react";
import "./App.css";
import Logo from "./components/Logo";
import FileDropzone from "./components/FileDropzone";
import VideoPlayer from "./components/VideoPlayer";

function App() {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [view, setView] = useState<'upload' | 'player'>('upload');

  const handleVideoSelect = (file: File) => {
    console.log("Valid MP4 file selected:", file.name);
    setSelectedVideo(file);
    setView('upload');
    // You can now do something with the file, like preparing it for upload
    // or passing it to your Python backend via IPC.
  };

  const handleVideoRemove = () => {
    console.log("Selected file removed.");
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


// Show input resolution
// Ask for user input on what scalar multiplier they want (2 or 4) (dropdown)
// for now just keep it 2

  return (
    <>
      <section className="flex flex-col gap-4 items-center justify-center">
        <div className="flex items-center justify-center gap-2">
          <Logo color="var(--blue)" className="w-11 h-11" />
          <h1 className="text-(--blue) font-semibold">Bluescale</h1>
        </div>
        {/* --- Conditional Rendering --- */}
        {view === 'upload' && (
          <>
            <FileDropzone
              onFileSelect={handleVideoSelect}
              onFileRemove={handleVideoRemove}
              className=""
            />
            {selectedVideo && (
              <div className="mt-4 flex flex-col items-center gap-4">
                <p className="text-gray-300">File ready: {selectedVideo.name}</p>
                <button
                  onClick={handleUpscaleClick} // Use the new handler
                  className="bg-(--blue) px-8 py-2 text-xl rounded-full font-semibold text-white hover:bg-opacity-80 transition-opacity duration-300"
                >
                  Upscale
                </button>
              </div>
            )}
          </>
        )}

        {view === 'player' && selectedVideo && (
          // Pass the selected file and potentially a function to go back
          <VideoPlayer videoFile={selectedVideo} onBack={handleBackToUpload} />
        )}
        {/* --- End Conditional Rendering --- */}

      </section>
    </>
  );
}

export default App;