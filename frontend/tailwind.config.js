/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        dark:   '#12121e',
        card:   '#1c1c2e',
        panel:  '#252540',
        accent: '#7c5cbf',
        safe:   '#22c55e',
        mild:   '#f97316',
        danger: '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
