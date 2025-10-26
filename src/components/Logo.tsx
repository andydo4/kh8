import React from "react";

// Define the props the component will accept
interface IconProps extends React.SVGProps<SVGSVGElement> {
  color?: string; // Make color optional, provide a default
}

// Create the functional component
const Logo: React.FC<IconProps> = ({ color = "#000", ...props }) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      {...props} // Spread remaining props (like className, style, onClick, etc.)
    >
      <path
        fill="none"
        stroke={color} // Use the color prop here
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M14 21v-5.5l4.5 2.5M10 21v-5.5L5.5 18m-2-3.5L8 12L3.5 9.5m17 0L16 12l4.5 2.5M10 3v5.5L5.5 6M14 3v5.5L18.5 6M12 11l1 1l-1 1l-1-1z"
      />
    </svg>
  );
};

export default Logo;
