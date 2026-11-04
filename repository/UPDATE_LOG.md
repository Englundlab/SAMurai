# SAMurai update log

## 0.10.7 — 2026-11-04

- Light correction and YOLO are separate actions for projects and single images. Skip optional correction for images already corrected upstream. Correction saves independently; YOLO uses the saved preprocessing choice and preserves completed detections when resumed.
- Application headings, dialogs and launcher descriptions use SAMurai. New projects use the SAMurai format identifier; existing projects remain readable.
- No model weights, dependencies or runtime changes.
- Validation: 23 project/review-queue tests; GUI checks for direct and corrected inference inputs, project reopening, SAM consistency and export; real CPU YOLO/SAM/export with saved mask pixel equality. The Windows portable application passed the GUI workflow using its bundled Python. macOS was not tested for this update.
