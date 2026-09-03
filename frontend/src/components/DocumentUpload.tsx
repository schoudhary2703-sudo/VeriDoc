import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Document upload with drag-and-drop and a live preview.
 *
 * Rejects oversized or non-image files here rather than letting the server do
 * it, so an officer at a checkpoint gets the reason immediately instead of
 * waiting out an upload that was never going to succeed.
 */

const MAX_BYTES = 25 * 1024 * 1024;

interface Props {
  label: string;
  hint?: string;
  file: File | null;
  onChange: (file: File | null) => void;
  disabled?: boolean;
}

export default function DocumentUpload({
  label,
  hint,
  file,
  onChange,
  disabled = false,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  // The preview is derived from the `file` prop rather than set only when the
  // user picks something. Keeping it in local state alone meant a file supplied
  // by the parent -- loading a demo document, for instance -- was accepted but
  // never shown, so the panel claimed nothing was loaded while the verification
  // button was live. The object URL is revoked whenever it is replaced.
  useEffect(() => {
    if (!file) {
      setPreview((old) => {
        if (old) URL.revokeObjectURL(old);
        return null;
      });
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return url;
    });
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const accept = useCallback(
    (candidate: File | undefined) => {
      if (!candidate) return;

      if (!candidate.type.startsWith("image/")) {
        setError(`${candidate.name} is not an image file.`);
        return;
      }
      if (candidate.size > MAX_BYTES) {
        setError(
          `${candidate.name} is ${(candidate.size / 1_048_576).toFixed(1)} MB, over the 25 MB limit.`,
        );
        return;
      }

      setError(null);
      onChange(candidate);
    },
    [onChange],
  );

  const clear = () => {
    setError(null);
    onChange(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm font-medium text-slate-200">{label}</span>
        {file && (
          <button
            type="button"
            onClick={clear}
            disabled={disabled}
            className="text-xs text-slate-500 hover:text-slate-300 disabled:opacity-50"
          >
            Remove
          </button>
        )}
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) accept(e.dataTransfer.files?.[0]);
        }}
        className={`relative overflow-hidden rounded-xl border border-dashed transition ${
          dragging
            ? "border-sky-500 bg-sky-500/5"
            : "border-slate-700 bg-slate-900/40"
        } ${disabled ? "opacity-60" : ""}`}
      >
        {preview ? (
          <div className="relative">
            <img
              src={preview}
              alt={`${label} preview`}
              className="max-h-56 w-full object-contain"
            />
            <div className="border-t border-slate-800 px-3 py-2 text-xs text-slate-500">
              {file?.name} · {file ? (file.size / 1024).toFixed(0) : 0} KB
            </div>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={disabled}
            className="flex w-full flex-col items-center gap-2 px-6 py-10 text-center"
          >
            <span className="text-sm text-slate-300">
              Drop an image here, or click to browse
            </span>
            {hint && <span className="text-xs text-slate-500">{hint}</span>}
          </button>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          disabled={disabled}
          onChange={(e) => accept(e.target.files?.[0])}
        />
      </div>

      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
    </div>
  );
}
