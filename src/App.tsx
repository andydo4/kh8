import { useState } from "react";
import "./App.css";
import Logo from "./components/Logo";
import FileDropzone from "./components/FileDropzone";

function App() {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);

  const handleVideoSelect = (file: File) => {
    console.log("Valid MP4 file selected:", file.name);
    setSelectedVideo(file);
    // You can now do something with the file, like preparing it for upload
    // or passing it to your Python backend via IPC.
  };

  const handleVideoRemove = () => {
    console.log("Selected file removed.");
    setSelectedVideo(null);
  }

  return (
    <>
      <section className="flex flex-col gap-4 items-center justify-center">
        <div className="flex items-center justify-center gap-2">
          <Logo color="var(--blue)" className="w-11 h-11" />
          <h1 className="text-(--blue) font-semibold">Bluescale</h1>
        </div>
        <FileDropzone onFileSelect={handleVideoSelect} onFileRemove={handleVideoRemove} className="" />
        {selectedVideo && (
          <div className="mt-4 flex flex-col gap-4">
            <p>File ready: {selectedVideo.name}</p>
            <button className="bg-(--blue) px-8 py-2 text-xl rounded-full font-semibold hover:bg-(--blue)/80 transition-colors duration-150">Upscale</button>
          </div>
        )}

      </section>
    </>
  );
}

export default App;
