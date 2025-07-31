const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_Maps_API_KEY: process.env.NEXT_PUBLIC_Maps_API_KEY,
    NEXT_PUBLIC_FLASK_API_URL: process.env.NEXT_PUBLIC_FLASK_API_URL,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
};

module.exports = nextConfig;
