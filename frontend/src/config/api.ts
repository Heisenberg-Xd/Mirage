/**
 * API base URL configuration.
 *
 * Priority:
 *   1. VITE_API_URL environment variable (set in Vercel dashboard for production,
 *      or in frontend/.env.local for local development)
 *   2. Falls back to localhost for local development convenience
 *
 * To configure for production:
 *   In Vercel → Project Settings → Environment Variables:
 *   VITE_API_URL = https://<your-railway-service>.railway.app
 *
 * For local development create frontend/.env.local:
 *   VITE_API_URL=http://localhost:8000
 */
const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export default API_URL;
