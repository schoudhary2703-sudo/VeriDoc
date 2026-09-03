/**
 * Mirrors the `VerifyResponse` contract in backend/app/core/schemas.py.
 *
 * Evidence carries a four-state status rather than a boolean. The officer needs
 * to tell "this check failed" from "this check is borderline" from "this check
 * could not run on this document" — collapsing those would either overstate a
 * weak signal or silently hide a missing one.
 */

export type EvidenceStatus = "pass" | "weak" | "fail" | "not_applicable";

export type RiskBand = "clear" | "review" | "high_risk";

export type PipelineStage =
  | "preprocessing"
  | "ocr_mrz"
  | "forensics"
  | "face"
  | "db_crosscheck";

/**
 * A coarse region of interest, in pixels of the submitted image.
 *
 * Deliberately coarse: tampered areas on ID documents are 0.27–4.17% of the
 * image and state-of-the-art detectors score near-zero on pixel-level
 * localization, so this is a box for an officer to look at — never a claim of
 * exact-pixel segmentation.
 */
export interface Region {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
}

export interface EvidenceItem {
  stage: PipelineStage | string;
  check: string;
  status: EvidenceStatus;
  detail: string;
  confidence: number;
  regions: Region[];
}

export interface Verdict {
  band: RiskBand;
  score: number;
  recommendation: string;
  evidence: EvidenceItem[];
}

export interface ExtractedFields {
  name: string | null;
  surname: string | null;
  given_names: string | null;
  dob: string | null;
  document_number: string | null;
  expiry_date: string | null;
  nationality: string | null;
  issuing_state: string | null;
  sex: "M" | "F" | "X" | null;
  personal_number: string | null;
}

export interface CheckDigitResult {
  field: string;
  raw_value: string;
  expected: string;
  actual: string;
  passed: boolean;
}

export interface MRZCheckResult {
  present: boolean;
  mrz_format: "TD1" | "TD2" | "TD3" | null;
  raw_lines: string[];
  valid: boolean;
  checksum_match: boolean;
  checks: CheckDigitResult[];
  errors: string[];
}

export interface ForensicsFinding {
  check: string;
  tamper_type: string | null;
  flagged: boolean;
  confidence: number;
  detail: string;
  regions: Region[];
  applicable: boolean;
}

export interface ForensicsResult {
  tampered: boolean;
  score: number;
  findings: ForensicsFinding[];
  processing_time_ms: number;
}

export interface FaceMatchResult {
  performed: boolean;
  match_score: number | null;
  matched: boolean | null;
  threshold: number | null;
  faces_in_document: number;
  faces_in_capture: number;
  liveness_passed: boolean | null;
  detail: string;
}

export interface DBCrosscheckResult {
  performed: boolean;
  found: boolean;
  blacklisted: boolean;
  status: string | null;
  /** Always states that the record set is simulated. Render it. */
  source: string;
  detail: string;
}

export interface VerifyResponse {
  verification_id: string;
  verdict: Verdict;
  extracted_fields: ExtractedFields;
  mrz_check: MRZCheckResult;
  forensics: ForensicsResult;
  face_match: FaceMatchResult;
  db_crosscheck: DBCrosscheckResult;
  processing_time_ms: number;
  stage_timings_ms: Record<string, number>;
}

export type OfficerAction = "cleared" | "referred" | "escalated";

export interface AuditEntry {
  verification_id: string;
  recorded_at: string;
  band: RiskBand;
  score: number;
  document_number: string | null;
  failed_checks: string[];
  weak_checks: string[];
  processing_time_ms: number;
  officer_action: OfficerAction | null;
  officer_id: string | null;
  officer_note: string | null;
  decided_at: string | null;
}

/** Presentation metadata for each band. */
export const BAND_LABELS: Record<RiskBand, string> = {
  clear: "Clear",
  review: "Review",
  high_risk: "High risk",
};

export const STAGE_LABELS: Record<string, string> = {
  preprocessing: "Capture",
  ocr_mrz: "OCR & MRZ",
  forensics: "Document forensics",
  face: "Face",
  db_crosscheck: "Records",
};

/** Human-readable names for individual checks. */
export const CHECK_LABELS: Record<string, string> = {
  mrz_checksum: "MRZ checksum and OCR agreement",
  document_validity: "Document validity",
  error_level_analysis: "Compression consistency",
  copy_move_detection: "Duplicated region detection",
  noise_consistency: "Sensor noise consistency",
  intra_document_face_consistency: "Portrait and ghost image consistency",
  cnn_classifier: "Learned tamper classifier",
  face_match: "Face match — live capture vs document photo",
  watchlist_and_record_check: "Record and watchlist cross-check",
};
