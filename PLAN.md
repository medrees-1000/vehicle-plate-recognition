# PLAN

Single planning file. Decisions below were agreed in the planning interview. Anything marked "verify" is not yet checked.

## Goal
- ALPR for US/EU plates: YOLO plate detector + OCR. Image-only.
- Only detection is trained. OCR is pretrained and evaluated, not trained.
- Main goal (not gated): a detector trained on CCPD (stage 1), then fine-tuned on a US/EU-weighted mix (stage 2), beats a pretrained plate-detector baseline on a held-out US/EU test set. Results are reported honestly either way.
- Domain gap: CCPD is Chinese plates only (blue, province character). It teaches "what a plate looks like" and detection under tilt/weather/blur. It gives no usable Latin OCR labels. Stage 2 closes the gap; an ablation (US/EU only) measures what CCPD adds.
- Bonus (secondary): a few hundred held-out CCPD images, reading accuracy reported separately with PaddleOCR's Chinese model, no extra training.
- Pretrained weights everywhere: COCO YOLO nano (small only if nano misses by a clear margin), pretrained PaddleOCR, a pretrained plate-detector baseline.
- Deployment in mind (Hugging Face Space, Gradio, CPU-only, ONNX): drives the nano choice. Not planned in detail.

## Datasets & licenses
Tiers: A = permissive/CC-BY. B = non-commercial/research-only. C = unclear/none (not used).
If any tier B data is used, the README says "weights are non-commercial".
Every row needs source URL, license, tier, checksum before use. Rows marked "verify" are open.

| Dataset | Role | License / tier | Source / checksum |
|---|---|---|---|
| CCPD2019 subset | Stage 1 train + 300 held-out Chinese test | Zenodo record: CC BY 4.0. Original repo (github.com/detectRecog/CCPD) terms not yet read; handled strictly as research-only, tier B until checked. Committed notebooks: stats and plots only, no sample images | Zenodo 15647076, CCPD2019.tar.xz, 13.2 GB, MD5 0dfbca0e6fcb7cb8ea720b0eae94c735; archive has 9 subfolders, 355,013 jpgs: base 199,996, challenge 50,003, tilt 30,216, fn 20,967, blur 20,611, db 10,132, rotate 10,053, weather 9,999, np 3,036 (no plates, excluded) |
| UC3M-LP | "Spain (EU)" test, evaluation only. Never train, never share converted copies | Zenodo files: CC BY 4.0 (tier A). Paper record says CC BY-NC-ND, GitHub says ODbL; conflict, so handled strictly | Zenodo 17152029, UC3M-LP.zip, 4.5 GB, MD5 496e3b5e66b78d231f71d85c0e5e11a3 |
| Roboflow US/EU + 1-2 other CC-BY multi-region sets | Stage 2 train/val (detection boxes, no text) | CC BY 4.0 per uploader (verify each page) | public.roboflow.com/object-detection/license-plates-us-eu; others chosen in M1b |
| One whole Roboflow US source, held out | US test; ~150 plates hand-labelled by the author | verify | chosen in M1b |
| OpenALPR benchmark | Not used: repo is AGPL-3.0 but image license is unstated (tier C) | C | github.com/openalpr/benchmarks |
| Pretrained baseline detector | Baseline to beat | verify license + training data | shortlist 2-3 in M2 |

Rules:
- Test sets are whole held-out sources, split by source/vehicle.
- Near-duplicate check before freezing: dHash (Pillow), Hamming distance <= 6, test vs all training data (CCPD and stage 2).
- Baseline: prefer documented training data, do not reject undocumented ones. If the baseline saw test images, that can only inflate its score (the comparison is then conservative); say so in the README.
- Downloads > 1 GB need user approval first. Raw archives are kept once in `MyDrive/alpr/archives` with MD5 checked.
- CCPD2019 subfolders overlap: 34,917 basenames occur in more than one subfolder (76,412 of 355,013 files, 21.5%), only outside ccpd_base and ccpd_np. Share of each subfolder's files in a duplicated name: blur 72%, db 67%, fn 67%, rotate 61%, challenge 48%, tilt 35%, weather 1%. Names encode the annotation, so these are very likely the same image (not verified byte-for-byte). Rule: deduplicate by basename before sampling (ccpd_np excluded first); each unique image keeps one subset label by fixed priority, rarest first: weather, rotate, db, blur, fn, tilt, challenge (base last). The subset label is only a sampling stratum; analysis and error slices use filename metadata (tilt, brightness, blur, bbox), not the subset label.

## Evaluation
- Detection: mAP@0.5 and mAP@0.5:0.95 on the US/EU test set.
- Models compared: (1) pretrained baseline, (2) COCO YOLO fine-tuned on US/EU only (ablation), (3) main model (CCPD stage 1, then US/EU stage 2).
- Reading: exact match (case-insensitive, ignoring spaces/hyphens) is the headline; also character error rate and CPU latency per plate.
- Crops: ground-truth boxes (isolates OCR) and detector boxes (end-to-end). End-to-end uses one fixed OCR engine across detectors.
- Reported separately: Spain (EU), US, Chinese held-out.
- Test sets are never used for tuning or model selection. Validation slices are held out from each training source.
- Target size: at least 300 text-labelled US/EU plates in total.

## OCR
- Compare three engines on identical crops: PaddleOCR (primary), fast-plate-ocr, EasyOCR. Tesseract only if trivial.
- PaddleOCR: pretrained English/Latin recognizer for US/EU, Chinese model for the Chinese test.
- fast-plate-ocr: code MIT. Weights license unstated, regions of current models unstated (legacy global model: "+65 countries", unverified). Evaluate it; deploy only if the weight license is clarified (open an issue with the author). Mark "weights license unstated" here and in the README.
- Deployed engine: best exact match on US/EU with a CPU latency ceiling of 500 ms/plate.

## Colab/Drive
- Free T4, disconnects expected. Runtime is remote: nothing local is visible.
- Drive layout: `MyDrive/alpr/{archives,ckpt,runs}`. Raw archives and packed subsets are single `.tar` files.
- CCPD: stream-extract only the sampled subset, never the whole archive.
- Training always runs from the VM's local disk: copy the packed subset to `/content` first, never train from the Drive mount. MD5 check after copy.
- Notebooks mount Drive, clone the public repo, install without torch (preinstalled), then explore, train (Ultralytics API), evaluate and report. `src/alpr/` is only data plumbing plus small tested shared helpers. Data prep runs as repo commands (`python -m alpr.ccpd ...`).
- Checkpoints saved to Drive every epoch; after a disconnect the notebook resumes with `resume=True` from the latest `last.pt`. Stage 2 takes stage 1's `best.pt` from Drive.
- Parameters and seed (`seed=0`), `imgsz=640`, AMP on, all in a top cell. Notebooks are run with Restart & Run All before committing; executed outputs are kept.

## Repo structure
- `pyproject.toml` + `uv` + lockfile. `src/alpr/` installable package (data plumbing + shared helpers). `notebooks/` numbered, executed outputs kept. Tests in `tests/` with `tests/fixtures/` (<10 small images).
- Torch is NOT a dependency on Colab (preinstalled). PaddlePaddle lives in an optional extra (`ocr-paddle`).
- `requirements.txt` is deleted once everything is migrated (user approved).
- Public GitHub repo, code only. Weights and demo go to Hugging Face.
- License: AGPL-3.0 (Ultralytics).
- This is the only planning file. No DATA.md, no `docs/`.
- Exact commands go in CLAUDE.md "Commands" after M0.

## Pipeline coverage
Status is updated as milestones finish. Verify column = how the claim is reproduced. Verification means running the numbered notebooks in order plus the data-prep commands, all listed in the README.

| Stage | Status | Reason | How to verify |
|---|---|---|---|
| EDA | not yet (M1a) | Know the subset before training: balance, sizes, label sanity | run `notebooks/01_eda.ipynb`: subset balance, image sizes, bbox sizes, brightness/blur/tilt, label sanity, duplicates (plots in the executed notebook) |
| Cleaning | partly done (M1a: unparseable names, cross-subfolder duplicates); rest M1b | CCPD filenames carry labels; corrupt images/bad boxes must not reach training; the same image appears in several subfolders (see Datasets rules) | sample command prints unparseable and collapsed-duplicate counts; cleaning script prints counts dropped per reason; pytest on fixtures; duplicate analysis in notebooks/01_eda.ipynb |
| Preprocessing | not yet (M1b) | CCPD filename -> YOLO boxes; Ultralytics letterbox to 640 | conversion script + test; counts printed |
| Feature engineering | skipped for the detector | CNN learns its own features; hand features would not be available at inference. CCPD filename metadata (tilt, brightness, blur, plate position) is used for stratified sampling and error analysis only, never as model input | sampling script prints strata counts (M1a) |
| Split | not yet (M1b) | CCPD train/val by seed; 300 held-out; test sets are whole held-out sources | split command prints sizes; dHash duplicate report (notebook, until a shared helper is needed) is empty |
| Model selection | not yet (M2-M4) | YOLO nano first, small only on a clear miss; OCR by exact match under latency ceiling | results tables in the evaluation notebook, copied to README |
| Hyperparameters | not yet (M3-M4) | Ultralytics defaults; limited tuning (free T4), chosen on validation only | parameters and seed in the top cell of the training notebook, chosen on validation only |
| Evaluation | not yet (M2, M4) | Metrics in the Evaluation section | evaluation notebook (number set in M2), using tested metric helpers from `src/alpr` |
| Error analysis | not yet (M4-M5) | Slice by CCPD metadata, box size, source; failure gallery; OCR error types | error-analysis notebook: slice tables, failure gallery, OCR error types |

## Milestones checklist
One at a time; verify, tick, stop.
- [x] M0 Setup: pyproject, `src/alpr/`, pytest + fixtures, remove requirements.txt, LICENSE (AGPL-3.0), fill CLAUDE.md Commands
- [ ] M1a CCPD + EDA: download CCPD2019 (MD5 check, archive kept on Drive), stream-extract the stratified sample (2,200 train per subset label x 8 = 17,600 after basename dedup, + 300 held-out, seed 0; ccpd_np excluded), pack it, then EDA in notebooks/01_eda.ipynb: plots embedded in the notebook, PNGs also written to Drive (subset balance, image sizes, bbox sizes, brightness/blur/tilt, label sanity, duplicates)
- [ ] M1b Data: UC3M-LP download (MD5) and YOLO conversion (local only), choose Roboflow stage 2 sources + US held-out source, hand-label ~150 US plates, CCPD cleaning + filename-to-YOLO conversion + split, dHash near-duplicate check, freeze test sets
- [ ] M2 Evaluation harness + baselines: metrics code; shortlist 2-3 baseline detectors, score the best; score PaddleOCR / fast-plate-ocr / EasyOCR on ground-truth crops
- [ ] M3 Stage 1: train on CCPD, resumable on Colab
- [ ] M4 Stage 2: fine-tune on US/EU-weighted mix; US/EU-only ablation; Chinese held-out report; error analysis
- [ ] M5 Pipeline: detect -> crop -> OCR CLI; final engine choice
- [ ] M6 Report + deploy: README tables with reproduce commands, ONNX export, minimal Gradio Space (CPU)
- [ ] M7 (optional, last) Video/webcam inference
- [ ] M8 (optional) Post-submission cleanup: move training/eval logic from notebooks into src/alpr CLIs, keep notebooks as thin reports, verify results match the tagged submission

Context: progress presentation on 15 Nov 2026; project due December. No scope cuts, no day-by-day schedule. The professor has the dataset and will test claims, so everything done or skipped needs a reason (see Pipeline coverage).

## Definition of done
- Gated: every claim reproduces from README commands, and results are reported honestly for all models and engines (including if the baseline wins, with an explanation). Goal, not gated: the main model beats the pretrained baseline on mAP@0.5:0.95 on the US/EU test set. No reading-accuracy floor yet; revisit after M2.
- Reported, not gated: ablation vs main, Spain (EU), US and Chinese reading accuracy for all three engines, CPU latency.
- Tests pass; no datasets, weights or `runs/` in git; boxes through M6 ticked; README notes "weights are non-commercial" if tier B data was used and "weights license unstated" for fast-plate-ocr.
- Hugging Face Space live, stating which engine and weights it uses.
