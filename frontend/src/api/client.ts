import type { HealthResponse } from "../types/health";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE_URL}${path}`, init);
  if (!resp.ok) {
    throw new Error(`${init?.method ?? "GET"} ${path} failed: ${resp.status}`);
  }
  return (await resp.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}
