import type {
  AuditEntry,
  OfficerAction,
  VerifyResponse,
} from "../types/verification";
import type { HealthResponse } from "../types/health";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${BASE_URL}${path}`, init);
  } catch {
    // A network-level failure is the officer's most likely problem in the
    // field, so name it plainly rather than surfacing "Failed to fetch".
    throw new ApiError(
      `Cannot reach the verification service at ${BASE_URL}. Check that the backend is running.`,
      0,
    );
  }

  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* response body was not JSON; keep the status line */
    }
    throw new ApiError(detail, resp.status);
  }

  return (await resp.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export interface VerifyOptions {
  documentImage: File | Blob;
  liveFaceImage?: File | Blob | null;
  /**
   * Restrict OCR to the machine-readable zone: roughly ten times faster and
   * still validates every check digit, but does not read the printed fields.
   */
  fast?: boolean;
}

export function verifyDocument({
  documentImage,
  liveFaceImage,
  fast = false,
}: VerifyOptions): Promise<VerifyResponse> {
  const form = new FormData();
  form.append("document_image", documentImage, "document.png");
  if (liveFaceImage) {
    form.append("live_face_image", liveFaceImage, "capture.png");
  }

  return request<VerifyResponse>(`/api/verify?fast=${fast}`, {
    method: "POST",
    body: form,
  });
}

export function getAuditLog(limit = 50): Promise<AuditEntry[]> {
  return request<AuditEntry[]>(`/api/audit-log?limit=${limit}`);
}

export function submitDecision(
  verificationId: string,
  action: OfficerAction,
  officerId: string,
  note?: string,
): Promise<AuditEntry> {
  return request<AuditEntry>(`/api/audit-log/${verificationId}/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, officer_id: officerId, note: note || null }),
  });
}
