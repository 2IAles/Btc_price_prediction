/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0a',
        panel: '#111111',
        'panel-border': '#1a1a1a',
        'accent-green': '#00ff88',
        'accent-red': '#ff4444',
        'accent-btc': '#f7931a',
        'text-primary': '#ffffff',
        'text-secondary': '#888888',
        'text-muted': '#555555',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
};
