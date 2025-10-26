import React, { useState, useRef } from "react";
import type { DragEvent, ChangeEvent } from "react";
import "../App.css";
import UploadIcon from "./UploadIcon";

// Define the props the component will accept
interface FileDropzoneProps {
  onFileSelect: (file: File) => void;
  onFileRemove: () => void;
  className?: string; // Optional className for the ROOT element
}

const FileDropzone: React.FC<FileDropzoneProps> = ({
  onFileSelect,
  onFileRemove,
  className,
}) => {
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File | null | undefined) => {
    setError(null);
    setSelectedFileName(null);
    if (!file) return;
    if (file.type !== "video/mp4") {
      setError("Invalid file type. Please upload an MP4 video.");
      return;
    }
    setSelectedFileName(file.name);
    onFileSelect(file);
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    const files = event.dataTransfer.files;
    if (files && files.length > 0) handleFile(files[0]);
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) handleFile(files[0]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleRemoveClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    setError(null);
    setSelectedFileName(null);
    onFileRemove();
    if (fileInputRef.current) {
        fileInputRef.current.value = '';
    }
  };

  // --- Tailwind Classes ---
  const baseClasses =
    "border-2 border-black/20 rounded-lg bg-black/40 w-[600px] h-[350px] flex justify-center items-center cursor-pointer transition-colors duration-300";
  const draggingClasses =
    "bg-(--blue) rounded-lg w-[600px] h-[350px] flex justify-center items-center cursor-pointer transition-colors duration-300";
  const errorClasses = "border-red-500 bg-red-50";
  const fileSelectedClasses = 'cursor-default';

  return (
    // Combine classes using template literals and ternary operators
    <div
      className={`
        ${baseClasses}
        ${isDragging ? draggingClasses : ""}
        ${error ? errorClasses : ""}
        ${selectedFileName ? fileSelectedClasses : ""}
        ${className || ""}
      `.replace(/\s+/g, ' ').trim()}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      role="button"
      tabIndex={0}
      aria-label="File drop zone"
    >
      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="video/mp4"
        className="hidden" // Tailwind class to hide the input
        aria-hidden="true"
      />
{/* --- Updated Display Content --- */}
      {selectedFileName ? (
        <div className="flex items-center justify-center gap-2"> {/* Use flex to align */}
          <p className="text-gray-700 truncate max-w-[80%]"> {/* Truncate long names */}
            Selected: {selectedFileName}
          </p>
          {/* --- The Remove Button --- */}
          <button
            onClick={handleRemoveClick}
            className="text-red-500 hover:text-red-700 font-bold text-xl p-1 leading-none"
            aria-label="Remove selected file"
            title="Remove file" // Tooltip
          >
            &times; {/* Simple 'X' character */}
            {/* Or use an SVG Trash Can Icon here */}
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-2">
            <UploadIcon color="#1e2939" className="w-30 h-30"/>
            <p className="text-gray-800 text-xl">
            {isDragging
                ? 'Drop the MP4 file here...'
                : 'Drag & drop an MP4 file here, or click to select'}
            </p>
        </div>

      )}

      {error && (
        <p className="text-red-600 mt-2 text-sm">{error}</p>
      )}
    </div>
  );
};

export default FileDropzone;