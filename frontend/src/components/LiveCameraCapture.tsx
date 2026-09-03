import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Live face capture from the device camera.
 *
 * The stream is stopped on unmount and immediately after a frame is taken. A
 * verification kiosk that leaves the camera running after it is done is both a
 * privacy problem and an obvious one to a traveller watching the indicator
 * light.
 */

interface Props {
  onCapture: (blob: Blob | null) => void;
  captured: Blob | null;
  disabled?: boolean;
}

export default function LiveCameraCapture({
  onCapture,
  captured,
  disabled = false,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setActive(false);
  }, []);

  useEffect(() => stop, [stop]);

  const start = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setActive(true);
    } catch (err) {
      const name = (err as Error).name;
      setError(
        name === "NotAllowedError"
          ? "Camera access was denied. Allow it in the browser, or upload a photo instead."
          : name === "NotFoundError"
            ? "No camera was found on this device. Upload a photo instead."
            : `Could not start the camera (${name}).`,
      );
    }
  };

  const capture = () => {
    const video = videoRef.current;
    if (!video) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (!blob) return;
      setPreview((old) => {
        if (old) URL.revokeObjectURL(old);
        return URL.createObjectURL(blob);
      });
      onCapture(blob);
      stop();
    }, "image/png");
  };

  const retake = () => {
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return null;
    });
    onCapture(null);
    void start();
  };

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm font-medium text-slate-200">
          Live face capture
          <span className="ml-2 text-xs font-normal text-slate-500">optional</span>
        </span>
        {captured && (
          <button
            type="button"
            onClick={retake}
            disabled={disabled}
            className="text-xs text-slate-500 hover:text-slate-300 disabled:opacity-50"
          >
            Recapture
          </button>
        )}
      </div>

      <div className="overflow-hidden rounded-xl border border-dashed border-slate-700 bg-slate-900/40">
        {preview ? (
          <img
            src={preview}
            alt="Live capture preview"
            className="max-h-56 w-full object-contain"
          />
        ) : (
          <>
            <video
              ref={videoRef}
              playsInline
              muted
              className={`max-h-56 w-full bg-black object-contain ${active ? "" : "hidden"}`}
            />
            {!active && (
              <div className="flex flex-col items-center gap-2 px-6 py-10 text-center">
                <span className="text-sm text-slate-300">
                  Capture the traveller&rsquo;s face to enable face matching
                </span>
                <button
                  type="button"
                  onClick={start}
                  disabled={disabled}
                  className="mt-1 rounded-lg bg-slate-800 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-700 disabled:opacity-50"
                >
                  Start camera
                </button>
              </div>
            )}
          </>
        )}

        {active && (
          <div className="flex items-center justify-between border-t border-slate-800 px-3 py-2">
            <span className="text-xs text-slate-500">Camera live</span>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={stop}
                className="rounded px-2 py-1 text-xs text-slate-400 hover:text-slate-200"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={capture}
                className="rounded bg-sky-600 px-3 py-1 text-xs font-medium text-white hover:bg-sky-500"
              >
                Capture
              </button>
            </div>
          </div>
        )}
      </div>

      {error && <p className="mt-2 text-xs text-amber-400">{error}</p>}
    </div>
  );
}
