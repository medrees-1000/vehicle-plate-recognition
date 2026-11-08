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
- Tests: one test file per module in src/alpr/. Test only data plumbing and shared helpers, not notebooks and not trivial code. Fixtures under 10 small images.
- Fix root causes; don't silence errors. Seed everything that affects results.
- Dependencies via `uv` only (never raw pip). Exact commands live under Commands once PLAN.md fixes the layout.

# Data & Git safety
- Never commit: datasets, `*.tar.xz`, `*.zip` archives, `*.pt`, `runs/`, `data/`, `.venv/`.
- Record every dataset's source URL, license, and checksum in PLAN.md (Datasets section) before using it.
- Verify MD5 after every large download.

# Colab
- The Colab runtime is remote: local files are not visible to it.
- Persistent data, checkpoints and runs go to Google Drive (mounted). Copy data to the VM's local disk as one archive before training; never train from many small files on a Drive mount.
- Notebooks (`notebooks/`, numbered) hold EDA, exploration, pretrained-model use, training (Ultralytics API), evaluation, results and error analysis. `src/alpr/` holds only data plumbing (download, MD5, stream-extract, sampling, packing, filename-to-YOLO conversion) and small tested helpers shared by several notebooks (e.g. exact-match, CER).
- Training must be resumable: checkpoints saved to Drive every epoch, `resume=True` after a disconnect. Parameters and seed in a top cell. Run Restart & Run All before committing a notebook.
- Raw `pip install uv` is approved only to bootstrap `uv` on Colab; everything else goes through `uv`.

# Commands
- Install: `uv sync` (add `--extra ocr-paddle` for PaddleOCR)
- Test: `uv run pytest`
- Lint: set in the milestone that introduces it. Train and eval: run the numbered notebooks in `notebooks/`.