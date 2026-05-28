/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Allow the frontend to be served either by `next start` or exported as
  // static assets (for desktop packaging via Tauri/Electron).
  // Set OUTPUT=export at build time to switch.
  output: process.env.OUTPUT === "export" ? "export" : undefined,
  // No image optimization domains — local-first, no remote images.
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
