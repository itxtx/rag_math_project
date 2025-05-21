// postcss.config.cjs
const config = {
  plugins: {
    "@tailwindcss/postcss": {}, // This is for Tailwind v4
    autoprefixer: {}, // Essential for adding vendor prefixes
  },
};

module.exports = config;
