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
Licenses rule: for every dataset record the source URL, version, checksum and the license as shown on its page, before use. Images and archives are never committed or redistributed (`data/` stays git-ignored). Which weights to publish is decided at deploy time, and the README lists the data sources and their licenses. Rows marked "verify" are open.

| Dataset | Role | License as shown | Source / checksum |
|---|---|---|---|
| CCPD2019 subset | Stage 1 train + 300 held-out Chinese test | Zenodo record: CC BY 4.0. Original repo (github.com/detectRecog/CCPD) terms not yet read (verify). Committed notebooks: stats and plots only, no sample images | Zenodo 15647076, CCPD2019.tar.xz, 13.2 GB, MD5 0dfbca0e6fcb7cb8ea720b0eae94c735; archive has 9 subfolders, 355,013 jpgs: base 199,996, challenge 50,003, tilt 30,216, fn 20,967, blur 20,611, db 10,132, rotate 10,053, weather 9,999, np 3,036 (no plates, excluded) |
| UC3M-LP | "Spain (EU)" test, evaluation only. Never train, never share converted copies | Zenodo files: CC BY 4.0. Paper record says CC BY-NC-ND, GitHub says ODbL; conflict, confirm on the record page (M1b-6). Test only, no shared converted copies. 1,975 images. Authors blur some random plate characters and alter vehicle colour for GDPR, so OCR scores on it are not absolute; M1b-6 checks whether the text labels contain the blurred characters | Zenodo 17152029, UC3M-LP.zip, 4.5 GB, MD5 496e3b5e66b78d231f71d85c0e5e11a3 |
| Roboflow License Plates, 350 images, v5 downloaded (augmented: re-download non-augmented before use) | Role open (M1b-3/4). Boxes only | Open Images subset. Annotations CC BY 4.0, images CC BY 2.0 (recorded as shown, not verified) | public.roboflow.com/object-detection/license-plates-us-eu; M1b-3 confirms which page the zip came from (samrat-sahoo appears to be a fork); zip `License Plates.v5-augmented-license-plates.yolov8.zip` MD5 13e58f5a41f8e2e79375e9944c66dd4d (notebooks/03_roboflow_audit.ipynb, Step 0; VM copy matched the Drive copy) |
| Objects In The Wild, 462 images, v6 | Role open. Plate-only boxes, no text. Some images blurry | CC BY 4.0 as shown (recorded as shown, not verified); images from Google images and Central Florida parks; the README has a takedown clause | https://universe.roboflow.com/objects-in-the-wild/license-plate-recognition-lhqow ; zip `License Plate Recognition.v6i.yolov8.zip` MD5 5970cdf35e4dbe188f2c82de92536ecb (notebooks/03_roboflow_audit.ipynb, Step 0; VM copy matched the Drive copy) |
| license-plate-usa, 670 images, v1 | Role open. Plate-only boxes, no text | CC BY 4.0 as shown (recorded as shown, not verified); no image source stated | https://universe.roboflow.com/custom-dataset/license-plate-usa ; zip `License-plate-usa.v1i.yolov8.zip` MD5 47ab70118bd47344bacba2c1efe4101a (notebooks/03_roboflow_audit.ipynb, Step 0; VM copy matched the Drive copy) |
| Roboflow EU candidates (e.g. patrick-neicu/european-license-plates, sumit-sharma-kcdlk/eu-license-plate-detection) | Candidates for stage 2 training | pending license check; ELP is academic-only and its country is unclear, so it needs checking. Paid sets (UniData, Appen) not used. Kaggle UniData unverified, not used | chosen at M1b-3 |
| OpenALPR benchmark | Candidate, check in M1b-3 (US/EU images, verify annotations and license as shown). Repo is AGPL-3.0 but image license is unstated | verify | github.com/openalpr/benchmarks |
| Pretrained baseline detector | Baseline to beat | verify license + training data | shortlist 2-3 in M2 |

Rules:
- Test sets are whole held-out sources: a source is never both train and test. CCPD is split by plate, not by image: all photos of one plate stay on the same side (train, validation or test).
- Near-duplicate check before freezing: dHash (Pillow), Hamming distance <= 6, between every training source and every test set, and among the three Roboflow sets (the Objects In The Wild author says some photos came from another dataset).
- Augmented copies (Roboflow exports with augmentation) must never go into a test pool or a split, because they create near-duplicates. Use non-augmented exports only.
- Labels for plate text are "AI-proposed, human-verified": Claude Code reads copied crops and proposes text, the author checks every one against its crop. PaddleOCR, EasyOCR and fast-plate-ocr are never used to make labels, because they are the models being tested. The crops are read by Claude Code (an AI service).
- Baseline: prefer documented training data, do not reject undocumented ones. If the baseline saw test images, that can only inflate its score (the comparison is then conservative); say so in the README.
- Downloads > 1 GB need user approval first. All raw archives and zips (including the Roboflow zips and UC3M-LP) are kept once in `MyDrive/alpr/archives`, a private folder that is not shared, with MD5 checked.
- CCPD2019 subfolders overlap: 34,917 basenames occur in more than one subfolder (76,412 of 355,013 files, 21.5%), only outside ccpd_base and ccpd_np. Share of each subfolder's files in a duplicated name: blur 72%, db 67%, fn 67%, rotate 61%, challenge 48%, tilt 35%, weather 1%. Names encode the annotation, so these are very likely the same image (not verified byte-for-byte). Rule: deduplicate by basename before sampling (ccpd_np excluded first); each unique image keeps one subset label by fixed priority, rarest first: weather, rotate, db, blur, fn, tilt, challenge (base last). The subset label is only a sampling stratum; analysis and error slices use measured or derived quantities (box geometry from bbox/vertices, filename brightness; see the EDA findings for fields not to use), not the subset label.
- CCPD overlap findings (notebooks/01_eda.ipynb, 5a): 313,518 unique names (310,482 excluding ccpd_np). A name appears in 1 to 6 subfolders (1: 278,601; 2: 28,989; 3: 5,384; 4: 454; 5: 74; 6: 16). ccpd_base overlaps with nothing; largest pairwise overlaps: blur-challenge 12,694, challenge-fn 7,919, fn-tilt 4,808. Available per subset label after dedup: weather 9,999, rotate 10,050, db 9,901, blur 16,629, fn 16,718, tilt 21,353, challenge 25,836, base 199,996. The hard subfolders (blur, challenge, db, fn, rotate, tilt, weather) are overlapping tags, not disjoint classes.
- CCPD sample (built in M1a, seed 0): 17,900 images = 17,600 train + 300 held-out, packed as one `ccpd_sample.tar` (1.26 GB; `train/`, `heldout/`, `manifest.csv`), MD5 5019a5333b4a92aded34cd3fb52eff4c, checked after the build and again after the copy to `MyDrive/alpr/archives/` (manifest `ccpd_manifest.csv` beside it).
- CCPD EDA findings (notebooks/01_eda.ipynb, M1a; counts only):
  - Images and boxes: all 17,900 images open; one image size, 720x1160. Plate box median 225x111 px (3.1% of the image), centred at cx 0.50, cy 0.45; aspect ratio has two modes (about 1.3 and 2.7).
  - Filename fields are not comparable across subfolders. In the 7 non-base labels the area field is about the measured plate fraction x 10,000 (median ratio 9,973-9,991 per label). In ccpd_base 98% of rows have area > 1e6 (median ratio about 7.6e12) and the tilt_h median is 90 vs 2-22 elsewhere. Rule: do not use the area or tilt fields across labels; derive geometry (plate area fraction, aspect, skew) from bbox/vertices.
  - Brightness and blur: filename brightness matches the measured plate brightness (Spearman 0.98 overall, 0.89-0.99 per label). Filename blur only partly (0.65 overall; per label from -0.14 in fn to 0.84 in db): do not use blur for slicing, notably in fn.
  - Label sanity: all 17,900 images usable; a box vs vertices gap above 10% of the longer box side occurs only in 47 ccpd_base images.
  - Plates: 17,311 distinct plates in 17,900 images, up to 5 images per plate. 16 of the 300 held-out images share a plate with a training image (vehicle-level leakage); open decision for M1b.
  - Domain: 16,038 of 17,900 plates (89.6%) start with the Anhui character and 15,391 (86.0%) have A as the second character, so the sample is regionally very narrow, which supports the US/EU stage 2.
  - Duplicates inside the sample: 8 exact duplicate files in 4 groups (none across train and held-out); dHash Hamming <= 6: 162 pairs, 131 images (0.7%); 5 of the 300 held-out images have a near-duplicate in train.
  - Re-split (M1b-1, `python -m alpr.ccpd split`, seed 0, manifest `MyDrive/alpr/archives/ccpd_split.csv`: member, subset, split, plate, bundle, box_gap): images are bundled if they share a plate or have dHash Hamming <= 6 (chains join); whole bundles are assigned. Counts: train 16,600 / validation 1,000 / held-out 300 = 17,900; per label 2,075 / 125 / 37-38 (held-out 38 for base, blur, challenge, db). 47 box_gap images (all base), none in held-out. Bundles larger than 20 images go wholly to train; only one is that large (47 images). The next largest has 10 images, then 5. No validation or held-out image shares a plate or a dHash link with a train image.
  - YOLO conversion (M1b-2, `python -m alpr.ccpd yolo`, checked in notebooks/02_ccpd_yolo_check.ipynb): `ccpd_yolo.tar` in `MyDrive/alpr/archives/`, MD5 bd44986acd6691a4f90fbea1e6017682 (VM copy matches the Drive file). Layout `images/{train,val,heldout}/`, `labels/{...}/` (one line `0 cx cy w h` from the filename box, normalised), `data.yaml` (train images/train, val images/val, test images/heldout, `names: {0: plate}`, no `path:` key). Of 17,900 images in the split file, 17,853 kept and 47 dropped (all box_gap, all ccpd_base: 44 train, 3 val; none in held-out); no other drop reason occurred. Kept per split: train 16,556 / val 997 / heldout 300. Per label: ccpd_base 2,031 / 122 / 38; the other seven labels unchanged (train 2,075, val 125, held-out 37-38). Extracted image and label counts match the expected counts in every split; 0 malformed label files. Visual check: boxes sit on the plates in train and held-out samples; smallest boxes 0.22-0.25% of the image, largest 29-32%. The dropped list is `ccpd_yolo.dropped.csv` beside the tar.
  - dHash finding: the 47-image bundle has 47 different plates, 0 same-plate pairs and 105 dHash-only pairs across 6 subfolders. Whole-image dHash at <= 6 therefore links look-alike scenes of different cars, not only copies. M1b-7 must look at the flagged pairs before treating them as duplicates.
  - Sample vs population: within each label the sample matches the population (medians agree); across labels it does not: the sampled fraction runs from 22.6% (db) to 1.1% (base), so the 300 held-out is also stratified by label. Chinese reading accuracy must be reported per label.

## Evaluation
- Detection: mAP@0.5 and mAP@0.5:0.95 on the US/EU test set.
- Detectors compared: (1) baseline = an existing public plate detector, untouched, named in M2, (2) CCPD only, (3) US/EU only (ablation), (4) CCPD then US/EU (main).
- OCR engines are tested, never trained. Pick the OCR on the human-verified US crops, pick the detector by mAP, then run the best of each end to end. No grid of all pairings.
- Reading: exact match (case-insensitive, ignoring spaces/hyphens) is the headline; also character error rate and CPU latency per plate.
- Crops: ground-truth boxes (isolates OCR) and detector boxes (end-to-end). End-to-end uses one fixed OCR engine across detectors.
- Reported separately: Spain (EU), US, Chinese held-out (per CCPD subset label, since the 300 are stratified by label).
- Test sets are never used for tuning or model selection. Validation slices are held out from each training source.
- Target size: at least 300 text-labelled US/EU plates in total (about 150-300 US plus UC3M-LP). The real count is reported.

## Known limits (for the report)
- Stage 2 training data is small; its real count is reported.
- The EU test covers one country (Spain).
- The US test mixes image quality, and OCR excludes unreadable plates, so its score leans toward easier images.
- US plate text is AI-proposed, human-verified. The report gives the number proposed, corrected and skipped as unreadable.
- CCPD results are reported per label (about 37 held-out images each, noisy).
- CCPD boxes are upright rectangles around 4 marked corners, so large tilted plates have loose boxes with empty corners; this may lower mAP@0.5:0.95 (strict IoU) on them.
- All three Roboflow exports are stretched to squares (416 px for samrat_v5 and florida_v6, 640 px for usa_v1), so plate aspect ratio is distorted.
- Plates are small: median plate width about 50 to 65 px (samrat_v5, florida_v6, at 416) and about 80 to 100 px (usa_v1, at 640), versus about 225 px in CCPD. OCR crops will be hard.
- usa_v1 valid and test have only 34 images each and are video frames (an "fps" overlay). About half its labels are polygons, not boxes; they would need converting to bounding rectangles.
- samrat_v5 has a second class (vehicle); only the plate class would be kept.
- The Step 7 "unaltered_looking" column in notebooks/03_roboflow_audit.ipynb is unreliable for usa_v1 and samrat_v5, because the black-border score also fires on natural black bars.
- The usa_v1 ORB threshold of 40 has not been checked on the lowest-scoring pairs.

## Presentation notes (15 Nov)
- Data availability: no clearly licensed US plate set of useful size beyond the 350-image one, so stage 2 data is small.
- Licensing: every source's license as shown on its page is recorded; no images or archives are redistributed; the choice of published weights is made at deploy time.

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
| EDA | done (M1a) | Know the subset before training: balance, sizes, label sanity | run `notebooks/01_eda.ipynb` (Restart & Run All): subset balance, image sizes, bbox sizes, brightness/blur/tilt, label sanity, duplicates (plots in the executed notebook, PNGs also on Drive); findings in the Datasets rules |
| Cleaning | partly done (M1a: unparseable names, cross-subfolder duplicates; sample check: 17,900 of 17,900 images usable, 8 exact-duplicate files, 47 base images with a box/vertices gap > 10%); rest M1b | CCPD filenames carry labels; corrupt images/bad boxes must not reach training; the same image appears in several subfolders (see Datasets rules) | sample command prints unparseable and collapsed-duplicate counts; cleaning script prints counts dropped per reason; pytest on fixtures; duplicate and label-sanity analysis in notebooks/01_eda.ipynb |
| Preprocessing | CCPD done (M1b-2); rest M1b | CCPD filename -> YOLO boxes; Ultralytics letterbox to 640 | conversion script + test; counts printed; box check in notebooks/02_ccpd_yolo_check.ipynb |
| Feature engineering | skipped for the detector | CNN learns its own features; hand features would not be available at inference. The filename area and tilt fields are not comparable across subfolders (EDA), so analysis uses geometry derived from bbox/vertices (plate area fraction, aspect, skew) and filename brightness; blur is unreliable. Used for sampling and error analysis only, never as model input | sampling script prints strata counts (M1a); notebooks/01_eda.ipynb (5e, 5h) |
| Split | CCPD done (M1b-1); rest M1b | CCPD train/val by seed; 300 held-out; test sets are whole held-out sources. Decided: regroup the 17,900 images by plate (plus dHash links) and re-split whole groups. Per label, 2,200 is train plus validation (about 2,075 train and 125 validation); the label-stratified 300 held-out is on top (37-38 per label); validation is about 1,000 in total. The 47 box/vertices-gap base images never go to held-out. Test: no plate on two sides | split command prints sizes; dHash duplicate report (notebook, until a shared helper is needed) is empty |
| Model selection | not yet (M2-M4) | YOLO nano first, small only on a clear miss; OCR by exact match under latency ceiling | results tables in the evaluation notebook, copied to README |
| Hyperparameters | not yet (M3-M4) | Ultralytics defaults; limited tuning (free T4), chosen on validation only | parameters and seed in the top cell of the training notebook, chosen on validation only |
| Evaluation | not yet (M2, M4) | Metrics in the Evaluation section | evaluation notebook (number set in M2), using tested metric helpers from `src/alpr` |
| Error analysis | not yet (M4-M5) | Slice by derived box geometry (plate area fraction, aspect, skew), filename brightness and source; no filename area, tilt or blur; Chinese held-out reported per subset label; failure gallery; OCR error types | error-analysis notebook: slice tables, failure gallery, OCR error types |

## Milestones checklist
One at a time; verify, tick, stop.
- [x] M0 Setup: pyproject, `src/alpr/`, pytest + fixtures, remove requirements.txt, LICENSE (AGPL-3.0), fill CLAUDE.md Commands
- [x] M1a CCPD + EDA: download CCPD2019 (MD5 check, archive kept on Drive), stream-extract the stratified sample (2,200 train per subset label x 8 = 17,600 after basename dedup, + 300 held-out, seed 0; ccpd_np excluded), pack it, then EDA in notebooks/01_eda.ipynb: plots embedded in the notebook, PNGs also written to Drive (subset balance, image sizes, bbox sizes, brightness/blur/tilt, label sanity, duplicates)
- [ ] M1b Data (tick each sub-step when done):
  - [x] M1b-1 CCPD re-split by plate (+dHash links); 300 held-out stratified by label; test that no plate is on two sides
  - [x] M1b-2 CCPD filename -> YOLO boxes; cleaning counts per reason
  - [x] M1b-3 Check the three Roboflow zips: README.roboflow version and image count vs 462 / 670 / 350; record URL, version, license as shown, MD5; you look at the images (US share, quality, readable plates)
    - weakest ORB band (scores 40 to 66) of usa_v1 pairs not visually reviewed
  - [ ] M1b-3 leftovers (not done): confirm which page each came from; no augmentation (re-download v5 non-augmented); vet the EU candidates and OpenALPR
  - [ ] M1b-4 Decide roles (see options below), then assemble the stage 2 train/val set; report the real count
    - roles of the three sets and the samrat_v5 train/test question are open; decided after the audit
  - [ ] M1b-5 US OCR labels on the test source(s): spot-check boxes by drawing a sample; copy only the crops to be read to a folder outside `data/` (`~/alpr_crops/`, not in the repo), batches of about 20 with contact sheets; Claude Code writes proposals to a CSV (readable / unreadable / unsure); you check each against its crop and correct; record counts proposed / corrected / skipped
  - [ ] M1b-6 UC3M-LP: ask first (4.5 GB), MD5, confirm license, convert, check whether labels contain blurred characters
  - [ ] M1b-7 dHash among the three Roboflow sets, test sets vs every training source, UC3M-LP vs training; freeze test sets
  - M1b-4 role options for the three Roboflow sets (a source is never both train and test; decide after looking at the images):
    1. 350 trains with the EU sets; 462 + 670 form the US test pool (about 1,100 images).
    2. 462 + 670 train; the 350 is the US test (small; Open Images, mixed regions, not all US).
    3. One of 462 / 670 trains, the other tests.
    To decide: how many plates per set are US and readable (the OCR test needs about 150 readable), how many images are blurry, how many dHash overlaps exist between sets (overlapping sets must land on the same side or be deduplicated), and whether a test set has boxes for the detector test.
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
- Tests pass; no datasets, weights or `runs/` in git; boxes through M6 ticked; README lists data sources and licenses and notes "weights license unstated" for fast-plate-ocr.
- Hugging Face Space live, stating which engine and weights it uses.

## Decision log
2026-11-09 (author's decisions):
- Licenses rule replaces the tier system: no tiers, no tier C exception. Record URL, version, checksum and license as shown on the page; no images or archives committed or redistributed; published weights are chosen at deploy time; README lists sources and licenses.
- Roles of the three Roboflow sets (350, 462, 670) are open until M1b-3/4. A source is never both train and test. dHash checks, no augmented copies, split by source stay.
- Plate-text labels are AI-proposed, human-verified; PaddleOCR, EasyOCR and fast-plate-ocr never make them. Crops are read by Claude Code.
- All zips go in `MyDrive/alpr/archives` (private, unshared). Crops for reading live in `~/alpr_crops/` on the PC, outside the repo.
- Downloaded versions: Objects In The Wild v6, license-plate-usa v1, Roboflow License Plates v5 (augmented; re-download non-augmented before use). M1b-3 checks counts 462 / 670 / 350 and which page each zip came from.
- Four detectors (baseline untouched, CCPD only, US/EU only, CCPD then US/EU); CCPD re-split by plate; OpenALPR is a candidate to check in M1b-3.
- Data-availability finding and licensing reasoning recorded under Presentation notes.

2026-11-11:
- M1b-2 largest-box check (notebooks/02_ccpd_yolo_check.ipynb, step 6b). Over all 17,853 kept images, max |YOLO box - filename box| = 0.0008 px: the converter is exact. The filename box equals the rectangle around the 4 vertices (median gap 0.0% in all three size buckets; the max stays below 10% only because the box_gap filter already dropped the larger gaps). The largest boxes look loose because CCPD boxes are upright rectangles around the 4 marked corners, and large tilted plates leave empty corners; in one image the corner marks look slightly above the plate (seen by eye, not measured). No rebuild of `ccpd_yolo.tar`; M1b-2 stays ticked. Consequence: mAP@0.5:0.95 may be lower on large tilted plates (see Known limits).
- M1b-3 Roboflow audit (notebooks/03_roboflow_audit.ipynb; audit only, no roles decided). MD5s and licences are in the Datasets table.
  - Sets and counts: samrat_v5 350 images (245/70/35 train/valid/test); florida_v6 462 (324/92/46); usa_v1 670 (602/34/34). All three counts match the dataset pages. The exported train folders are augmented and larger: 979 / 972 / 1,806 images; valid and test have the page counts.
  - Methods: dHash (Hamming <= 6) and ORB local features (500 features, Lowe ratio 0.75, RANSAC partial affine; the inlier count is the score).
  - Controls: on 30 synthetic altered copies per set (rotated, cropped, noised), dHash recall was 0/30 in every set; ORB recall was 0.93 (samrat_v5 at 42), 0.93 (florida_v6 at 25), 0.97 (usa_v1 at 40). So dHash alone misses altered copies.
  - ORB thresholds are set per dataset by hand: samrat_v5 42, florida_v6 25, usa_v1 40. Reason: a rule of "trimmed 99.5th percentile of random different-source pairs, floor 12" gave 12, too low for samrat_v5 (highest ORB score among its 20,000 random different-source pairs: 39; in the full search 2 pairs score 30 or more and none 50 or more; the highest full-search score is not recorded) and for usa_v1; the first rule (99.95th percentile) was pulled too high (83 for usa_v1) by real duplicates among the random pairs.
  - Results: samrat_v5 has 0 cross-split pairs by ORB and by dHash. florida_v6 has copies across every split (train x valid 419 pairs, train x test 207, valid x test 28; only 3 of these are different-source names), so name grouping captures nearly all copies, and its valid and test are not usable as a test. usa_v1: train x valid 63 pairs (60 with different file names), train x test 61 (57 different names), valid x test 4 (all different names); about 20 of 34 valid and 20 of 34 test images have a twin in train. Name grouping fails for usa_v1, so near-duplicate clusters (ORB) are needed.
  - Pairs viewed for usa_v1 (scores about 67 to 299) are the same plate or car, or the same road scene from a video camera with a different vehicle in view.
