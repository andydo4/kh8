import React from 'react';

// Define the props the component will accept
interface UploadIconProps extends React.SVGProps<SVGSVGElement> {
  color?: string; // Make color optional, provide a default
}

// Create the functional component
const UploadIcon: React.FC<UploadIconProps> = ({ color = '#3a86ff', ...props }) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      {...props} // Spread remaining props (like className, style, onClick, etc.)
    >
      <path
        fill={color} // Use the color prop here
        d="M5 20h14v-2H5zm0-10h4v6h6v-6h4l-7-7z"
      />
    </svg>
  );
};

export default UploadIcon;