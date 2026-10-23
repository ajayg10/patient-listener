/** @type {import('tailwindcss').Config} */
// Palette follows a dark, cool blue-grey "devtools" look: warm off-white type,
// pale teal accent, muted mint (good) and muted coral (bad). The stock palette
// names are remapped so every existing class picks up the new look.
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // cool blue-grey surfaces, warm off-white at the light end
        slate: {
          50: "#f5f1ea",
          100: "#ebe7e0",
          200: "#d4d8d8",
          300: "#b3bcbf",
          400: "#8b979c",
          500: "#66737a",
          600: "#4a565c",
          700: "#323c42",
          800: "#232c31",
          900: "#182025",
          950: "#11171b",
        },
        // pale teal accent (was indigo)
        indigo: {
          50: "#eef7f9",
          100: "#d9eef2",
          200: "#c0e2e9",
          300: "#a8d3dd",
          400: "#88bfcc",
          500: "#69a8b8",
          600: "#4f8e9e",
          700: "#3d717f",
          800: "#2d5662",
          900: "#203e47",
          950: "#13282e",
        },
        // muted mint for "Patient Listener / good"
        emerald: {
          50: "#edf8f3",
          100: "#d4efe3",
          200: "#b0e0cb",
          300: "#8ccfb2",
          400: "#6dbb9b",
          500: "#52a082",
          600: "#3f816a",
          700: "#326757",
          800: "#264f43",
          900: "#1b3a31",
          950: "#10251f",
        },
        // muted coral for "interrupted / bad"
        rose: {
          50: "#fcf1ef",
          100: "#f6ddd8",
          200: "#eebfb6",
          300: "#e3a093",
          400: "#d68374",
          500: "#c26657",
          600: "#a35145",
          700: "#824137",
          800: "#62322b",
          900: "#45231f",
          950: "#2b1513",
        },
        amber: {
          50: "#fbf6ea",
          100: "#f4e8c9",
          200: "#ead39c",
          300: "#ddbd76",
          400: "#cfa75a",
          500: "#b78d45",
          600: "#957137",
          700: "#75582c",
          800: "#574222",
          900: "#3d2e18",
          950: "#251c0e",
        },
        background: "#141c21",
        surface: "#182025",
        border: "#2a3439",
      },
      fontFamily: {
        sans: ["Geist", "Inter", "-apple-system", "Segoe UI", "sans-serif"],
        display: ["Geist", "Inter", "sans-serif"],
        mono: ["JetBrains Mono", "Geist Mono", "Consolas", "monospace"],
      },
      // flatter, squarer corners like the reference
      borderRadius: {
        sm: "2px",
        DEFAULT: "3px",
        md: "4px",
        lg: "5px",
        xl: "6px",
        "2xl": "8px",
      },
    },
  },
  plugins: [],
};
