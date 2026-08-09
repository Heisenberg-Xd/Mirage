import API_URL from '../config/api';

interface FetchOptions extends RequestInit {
  onRetry?: (attempt: number) => void;
  timeoutMs?: number;
}

export const api = {
  async get<T>(path: string, options: FetchOptions = {}): Promise<T> {
    return this.request<T>(path, { ...options, method: 'GET' });
  },

  async post<T>(path: string, body: any, options: FetchOptions = {}): Promise<T> {
    return this.request<T>(path, {
      ...options,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        ...options.headers,
      },
      body: JSON.stringify(body),
    });
  },

  async request<T>(path: string, options: FetchOptions): Promise<T> {
    const { onRetry, timeoutMs = 60000, ...customConfig } = options;
    const url = `${API_URL}${path}`;
    
    let attempt = 0;
    const maxAttempts = 3;

    while (attempt < maxAttempts) {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

      try {
        const response = await fetch(url, {
          ...customConfig,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          // If it's a server error (502/503 from Render waking up), we can retry
          if (response.status >= 500 && attempt < maxAttempts - 1) {
            throw new Error(`Server Error: ${response.status}`);
          }
          const errorText = await response.text().catch(() => 'Unknown error');
          throw new Error(`Server responded with ${response.status}: ${errorText}`);
        }

        return (await response.json()) as T;
      } catch (error: any) {
        clearTimeout(timeoutId);

        // Don't retry client errors (4xx) unless explicitly handled
        if (error.message.includes('Server responded with 4')) {
          throw error;
        }

        attempt++;

        if (attempt >= maxAttempts) {
          if (error.name === 'AbortError') {
            throw new Error('Request timed out after 60 seconds.');
          }
          throw new Error(`Failed to fetch after ${maxAttempts} attempts: ${error.message}`);
        }

        // Call the onRetry callback so UI can update (e.g. "Waking up the verification engine...")
        if (onRetry) {
          onRetry(attempt);
        }

        // Exponential backoff for retries: 1s, 2s, ...
        await new Promise((resolve) => setTimeout(resolve, attempt * 1000));
      }
    }

    throw new Error('Unexpected exit from request loop');
  },
};
