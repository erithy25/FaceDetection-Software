/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0A0A0B",
        surface: "#141416",
        border: "#2A2A2E",
        "text-primary": "#F5F5F7",
        "text-secondary": "#8E8E93",
        "accent-green": "#34D399",
        "accent-yellow": "#FBBF24",
        "accent-red": "#F87171",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: {
        card: "12px",
        button: "8px",
      },
    },
  },
  plugins: [],
};
