# Data Strategy — VeriDoc (SIH26188)

**The constraint, stated plainly:** no student team gets access to real MHA travel-document records, real border-checkpoint CCTV, or real watchlist data. Everything below is built around that — public benchmark datasets plus deliberately-generated synthetic data, with real data access named as a clearly-marked integration point for production, not implied or faked.

This file expands Section 5 of `BUILD_PLAN.md` with concrete datasets, current as of the 2025–2026 research landscape, one per pipeline module.

---

## 1 · OCR & MRZ — no training data needed

PaddleOCR and the MRZ checksum parser are **not trained by you** — PaddleOCR ships pretrained, and MRZ validation (ICAO 9303) is a fixed checksum spec, not a model. What you need is **test images**, not training data:

| Dataset | What it is | Size | Access | Use |
|---|---|---|---|---|
| **MIDV-500** | Video clips of 50 identity document types (mock/specimen documents from many countries) | 500 clips | Public, direct download | OCR/MRZ field extraction test set |
| **MIDV-2020** | Successor to MIDV-500 — higher resolution, more realistic capture conditions | 1,000 document images + video/photo variants | Public, direct download ([arxiv.org/abs/2107.00396](https://arxiv.org/abs/2107.00396)) | Primary OCR/MRZ validation set |

**Action:** pull both, run your OCR/MRZ pipeline against them, report field-extraction accuracy. No fine-tuning required unless accuracy is poor on a specific document layout — if so, fine-tune PaddleOCR's recognition head on a small labelled subset of these, not from scratch.

---

## 2 · Image Forensics / Tamper Detection — the module that actually needs training

This is where real training data matters. The good news: 2024–2025 produced several purpose-built, license-clean synthetic fraud datasets that didn't exist when MIDV-500 was the only option.

| Dataset | What it is | Size | License / access | Use |
|---|---|---|---|---|
| **IDNet (2025)** | Fully synthetic ID/passport documents across 10 countries, with two built-in fraud simulations: *inpaint-and-rewrite* (content replacement) and *crop-and-replace* (element substitution) | Large-scale (models + 10 per-country packages) | **CC-BY-4.0**, downloadable directly from Hugging Face ([huggingface.co/datasets/cactuslab/IDNet-2025](https://huggingface.co/datasets/cactuslab/IDNet-2025)) | **Primary training set** — closest public match to the PS's named tamper types (altered photos, modified DOB, tampered stamps) |
| **SIDTD** | Genuine + fake ID video clips, purpose-built for document verification systems | Public benchmark | Open, GitHub-hosted ([github.com/Oriolrt/SIDTD_Dataset](https://github.com/Oriolrt/SIDTD_Dataset)) | Cross-dataset validation — checks your classifier generalizes past IDNet's specific synthesis style |
| **DocTamper** | Large synthetic text-region tampering set (character/word-level edits) | ~170,000 images | Research use | Augments the DOB-edit / field-tampering case specifically |
| **DocXPand** | A synthetic identity-document *generator* (code, not a fixed dataset) | N/A — generates on demand | Open source ([github.com/quicksign/docxpand](https://github.com/quicksign/docxpand)) | **Use this to generate your own Indian-format specimen documents** — none of the datasets above cover Indian passports/visas, so this fills that gap without touching real data |
| **MIDV-Holo** | Video dataset for hologram/security-feature fraud detection | Public benchmark | Open, ICDAR 2023 | Optional stretch goal — only pursue if core pipeline is ahead of schedule |
| **FantasyID (IJCB 2025, Idiap)** | Authentic + digitally manipulated "fantasy" ID cards across 13 template designs (Arabic, Chinese, Hindi, French, Persian, Portuguese, Russian, Turkish, Ukrainian, Singaporean, Dutch, Hong Kong, English) | 2.6 GB (`FantasyID.tgz`) | **CC-BY-4.0 / CC0 - direct download, no data agreement.** [zenodo.org/records/17063366](https://zenodo.org/records/17063366), DOI `10.34777/c966-nn94` | Second training/validation source alongside IDNet - license-clean and script-diverse |

**Correction (verified 2026-09-01):** FantasyID is *not* gated behind an Idiap
data-access agreement. It is published on Zenodo under CC-BY-4.0 / CC0 as a single
2.6 GB archive and can be pulled directly. Nothing in this project now requires an
application with lead time - every dataset named above is fetch-on-demand.

**Downloaded and verified 2026-09-02.** FantasyID is in `data/raw/FantasyID`
(2.4 GB, MD5 checked against the Zenodo record). What it actually contains, now
that we have looked: 362 fantasy cards *printed on a card printer and
re-captured* with an iPhone 15 Pro, a Huawei Mate 30 and a Kyocera scanner, with
an official train/test split whose two halves use different card templates, and
per-image JSON giving every field a bounding box plus an original/modified
provenance flag. 196 images use the Indian template. Attacks are face swaps
(Inswapper, FaceDancer) and diffusion text inpainting (DiffSTE, TextDiffuser-2).

**One licensing nuance to state accurately in the submission:** the cards are
fictional, but the *faces* printed on the bonafide cards are real people, drawn
from AMFD, the Face Research Lab London set, and HQ-WMCA. The FantasyID release
is CC-BY-4.0/CC0, while HQ-WMCA is ordinarily an Idiap research-licence dataset.
So "no real personal documents" holds exactly as written; "no real faces" does
not, and should not be claimed.

**A finding worth knowing before you commit to an approach:** a 2026 benchmark study (DocForge-Bench) found that even state-of-the-art forgery detectors get high image-level discrimination (AUC ≥ 0.76) but near-zero *pixel-level* localization accuracy, because tampered regions are tiny — typically 0.27–4.17% of the image, versus 10–30% in natural-image forgery benchmarks that most forensics models were designed around. **Practical implication for your build:** don't over-promise pixel-perfect tamper heatmaps. Aim for image/field-level classification ("this photo region shows splice artifacts") with a coarse bounding box, not exact-pixel segmentation — it's honestly achievable in your timeline and still gives the evidence panel something concrete to show.

**Recommended training sequence:**
1. Generate your own synthetic Indian-specimen documents with DocXPand-style generation (clearly watermarked "SPECIMEN," never real citizen data)
2. Apply IDNet's tamper operations (inpaint-rewrite, crop-replace) plus DocTamper-style field edits to your specimens
3. Fine-tune EfficientNet-B0 on this combined set (your specimens + IDNet)
4. Validate on SIDTD as a held-out, differently-synthesized set — if accuracy collapses on SIDTD, the model overfit to IDNet's specific generation signature rather than learning general tamper cues

---

## 3 · Face Match — skip training, use a pretrained model

Do **not** train a face-recognition model from scratch — that needs millions of identities, and it's a solved problem you'd be reinventing. Use **InsightFace's pretrained ArcFace embeddings** (the `buffalo_l` model) directly.

| What you need | Source | Use |
|---|---|---|
| Pretrained embedding model | InsightFace `buffalo_l` (ships pretrained) | Face matching — no training needed |
| Threshold calibration set | LFW-style public benchmark + your own team's consented photos | Tune the match-confidence threshold to your camera/lighting setup |

**Action:** integrate the pretrained model, then spend your time tuning the similarity threshold — not training. Test specifically against low-quality document photos (the risk register in the project blueprint already flags this as a likely failure mode).

---

## 4 · Liveness / Anti-Spoofing — train a lightweight classifier

| Dataset | Size | Attack types covered | License |
|---|---|---|---|
| **CelebA-Spoof** | 625,537 images, 10,177 subjects | 10 spoof types (print, replay, etc.) — largest public 2D dataset | Research use only |
| **OULU-NPU** | 4,950 videos, 55 subjects, 6 devices | Print, replay — canonical mobile-era benchmark | Research use only |
| **Replay-Attack** | 1,300 videos, 50 subjects | Print, mobile replay, HD replay | Research use only |

**Action:** train your lightweight passive-liveness classifier on CelebA-Spoof (biggest, most diverse), cross-validate on OULU-NPU or Replay-Attack to check it isn't overfit to one capture setup.

**Important flag for the submission:** every public liveness/anti-spoofing dataset is licensed **research-only**. That's fine for a prototype and a hackathon demo — say so explicitly rather than implying a production-ready, commercially-licensed model. It's the same honesty move as the real-document-access boundary, and judges read it the same way: as a team that understands deployment constraints, not one that's hand-waving past them.

---

## 5 · What to say in the submission, verbatim-ready

> "The prototype is trained and validated entirely on public benchmark datasets (IDNet, SIDTD, MIDV-500/2020, CelebA-Spoof) and synthetic specimen documents generated for this project — no real citizen data was used at any stage. Production deployment would integrate with the issuing authority's live document-record and watchlist APIs under MHA's data-governance terms, and would require re-licensing the liveness/anti-spoofing components currently trained on research-only datasets for commercial use."

This is a stronger answer under judge questioning than most teams give, because it's specific: named datasets, a named integration point, and a named licensing gap — not a vague "we'd use real data in production."

---

## 6 · Quick-reference table

| Module | Primary dataset | Backup / validation | Training needed? |
|---|---|---|---|
| OCR & MRZ | MIDV-2020 | MIDV-500 | No — pretrained + rules-based |
| Forgery detection | IDNet + your own DocXPand-generated specimens | SIDTD, DocTamper | Yes — fine-tune EfficientNet-B0 |
| Face match | — | LFW (threshold calibration only) | No — pretrained ArcFace/InsightFace |
| Liveness | CelebA-Spoof | OULU-NPU, Replay-Attack | Yes — lightweight classifier |

---

## Sources

- [IDNet: A Novel Dataset for Identity Document Analysis and Fraud Detection](https://huggingface.co/datasets/cactuslab/IDNet-2025) — Hugging Face, CC-BY-4.0
- [SIDTD Dataset](https://github.com/Oriolrt/SIDTD_Dataset) — GitHub
- [Synthetic dataset of ID and Travel Documents (SIDTD paper)](https://www.nature.com/articles/s41597-024-04160-9) — Scientific Data
- [DocXPand — synthetic identity documents generator](https://github.com/quicksign/docxpand) — GitHub
- [MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis](https://arxiv.org/abs/2107.00396) — arXiv
- [MIDV-Holo: A Dataset for ID Document Hologram Detection in a Video Stream](https://link.springer.com/chapter/10.1007/978-3-031-41682-8_30) — ICDAR 2023
- [FantasyID: A dataset for detecting digital manipulations of ID-documents](https://www.idiap.ch/paper/fantasyid/) — Idiap, IJCB 2025
- [DocForge-Bench: A Comprehensive Benchmark for Document Forgery Detection and Analysis](https://arxiv.org/html/2603.01433v1) — arXiv, 2026
- [CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570069.pdf) — ECCV 2020
- [Liveness Detection Datasets: Guide to Face Anti-Spoofing Data](https://axonlab.ai/liveness-detection-datasets/) — AxonLab, dataset licensing comparison
