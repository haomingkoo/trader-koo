const getBaseUrl = (): string => {
  if (import.meta.env.VITE_API_BASE) return import.meta.env.VITE_API_BASE;
  return window.location.origin;
};

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const base = getBaseUrl();
  const res = await fetch(`${base}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    // Always throw a proper Error with a string message
    let msg = `API ${res.status}`;
    try {
      const body = await res.text();
      // Try to parse as JSON and extract detail
      try {
        const json = JSON.parse(body);
        msg =
          typeof json.detail === "string"
            ? json.detail
            : `API ${res.status}: ${body.slice(0, 200)}`;
      } catch {
        msg = `API ${res.status}: ${body.slice(0, 200)}`;
      }
    } catch {
      msg = `API ${res.status}: Unknown error`;
    }
    throw new Error(msg);
  }
  return res.json();
}
