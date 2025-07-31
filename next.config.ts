import { config } from 'dotenv';
import path from 'path';

// Manually load .env.local from the correct path
config({ path: path.resolve(__dirname, '.env.local') });

console.log('Loaded MAPS KEY:', process.env.NEXT_PUBLIC_Maps_API_KEY);

const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_Maps_API_KEY: process.env.NEXT_PUBLIC_Maps_API_KEY,
    NEXT_PUBLIC_FLASK_API_URL: process.env.NEXT_PUBLIC_FLASK_API_URL,
  },
};

export default nextConfig;