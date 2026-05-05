import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#0B1121', // Main app background
          900: '#111827', // Card backgrounds
          800: '#1F2937', // Hover states
        },
        pulse: {
          cyan: '#06b6d4',
          cyanGlow: 'rgba(6, 182, 212, 0.5)',
          green: '#22c55e',
          yellow: '#eab308',
          red: '#ef4444',
        }
      }
    },
  },
  plugins: [],
};
export default config;