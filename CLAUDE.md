# Project
vehicle-plate-recognition: ALPR pipeline (Ultralytics YOLO plate detector + PaddleOCR).
Detector trained on a CCPD2019 subset plus US/EU data; evaluated on US/EU plates.
Source of truth for goals, datasets, metrics and milestones: `PLAN.md` (repo root).
If PLAN.md is missing or stale, stop and say so; do not guess scope.

# Working agreement
- Plan before code: state the approach in a few lines, wait for approval on anything non-trivial.
- During planning/interviews: ask questions freely, one at a time, with a recommended answer. Brevity rules below don't apply.
- Ask before: downloading >1 GB, adding a dependency, changing the repo layout, deleting files.
- Never read or list `data/` or `runs/` contents. Inspect via small scripts that print summaries.
- Work one milestone from PLAN.md at a time. When done: verify, tick the box in PLAN.md, stop.

# Engineering rules
- Smallest correct diff. Stdlib first. No unrequested boilerplate, abstractions, or features.
- After code: max 3 lines of explanation.
- Type hints on all public functions. Validate inputs; handle corrupt images, missing files, bad labels explicitly.
- Every module has a runnable self-check or a pytest test. Tests use `tests/fixtures/` only (<10 small images).
- Fix root causes; don't silence errors. Seed everything that affects results.
- Dependencies via `uv` only (never raw pip). Exact commands live under Commands once PLAN.md fixes the layout.

# Data & Git safety
- Never commit: datasets, `*.tar.xz`, `*.zip` archives, `*.pt`, `runs/`, `data/`, `.venv/`.
- Record every dataset's source URL, license, and checksum in PLAN.md (Datasets section) before using it.
- Verify MD5 after every large download.

# Colab
- The Colab runtime is remote: local files are not visible to it.
- Persistent data, checkpoints and runs go to Google Drive (mounted). Copy data to the VM's local disk as one archive before training; never train from many small files on a Drive mount.
- The notebook only clones the repo and calls repo CLIs. Logic lives in `src/`, not in cells.
- Training must be resumable from the latest checkpoint on Drive.

# Commands
(Filled in after PLAN.md decides the layout: install, test, lint, train, eval.)