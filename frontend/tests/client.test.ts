import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, submitDecision, verifyDocument } from "../src/api/client";

/**
 * The API client's job on a bad day is to explain what went wrong. An officer at
 * a checkpoint seeing "Failed to fetch" learns nothing; seeing "cannot reach the
 * verification service" knows to check the backend.
 */

afterEach(() => vi.unstubAllGlobals());

function stubFetch(impl: unknown) {
  vi.stubGlobal("fetch", impl);
}

describe("api client", () => {
  it("names the service when the network is unreachable", async () => {
    stubFetch(vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    await expect(
      verifyDocument({ documentImage: new Blob(["x"]) }),
    ).rejects.toThrow(/cannot reach the verification service/i);
  });

  it("surfaces the server's own explanation for a rejected upload", async () => {
    stubFetch(
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: async () => ({ detail: "The document image must be an image." }),
      }),
    );

    await expect(
      verifyDocument({ documentImage: new Blob(["x"]) }),
    ).rejects.toThrow(/must be an image/i);
  });

  it("falls back to the status line when the error body is not JSON", async () => {
    stubFetch(
      vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        statusText: "Bad Gateway",
        json: async () => {
          throw new Error("not json");
        },
      }),
    );

    await expect(
      verifyDocument({ documentImage: new Blob(["x"]) }),
    ).rejects.toThrow(/502/);
  });

  it("carries the status code on the error for callers to branch on", async () => {
    stubFetch(
      vi.fn().mockResolvedValue({
        ok: false,
        status: 413,
        statusText: "Payload Too Large",
        json: async () => ({ detail: "too big" }),
      }),
    );

    await expect(
      verifyDocument({ documentImage: new Blob(["x"]) }),
    ).rejects.toMatchObject({ status: 413 });
  });

  it("sends the live capture only when one was supplied", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    stubFetch(fetchMock);

    await verifyDocument({ documentImage: new Blob(["doc"]) });
    let body = fetchMock.mock.calls[0][1].body as FormData;
    expect(body.get("document_image")).toBeTruthy();
    expect(body.get("live_face_image")).toBeNull();

    await verifyDocument({
      documentImage: new Blob(["doc"]),
      liveFaceImage: new Blob(["face"]),
    });
    body = fetchMock.mock.calls[1][1].body as FormData;
    expect(body.get("live_face_image")).toBeTruthy();
  });

  it("passes the fast-mode flag through to the query string", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    stubFetch(fetchMock);

    await verifyDocument({ documentImage: new Blob(["doc"]), fast: true });
    expect(fetchMock.mock.calls[0][0]).toContain("fast=true");
  });

  it("posts an officer decision as JSON", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    stubFetch(fetchMock);

    await submitDecision("abc-123", "referred", "BSF-2291", "Sending to secondary.");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toContain("/api/audit-log/abc-123/decision");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toMatchObject({
      action: "referred",
      officer_id: "BSF-2291",
      note: "Sending to secondary.",
    });
  });

  it("exposes ApiError so callers can distinguish it", async () => {
    stubFetch(vi.fn().mockRejectedValue(new TypeError("nope")));

    await expect(
      verifyDocument({ documentImage: new Blob(["x"]) }),
    ).rejects.toBeInstanceOf(ApiError);
  });
});
