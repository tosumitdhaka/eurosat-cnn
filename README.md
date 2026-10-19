# EuroSAT land-use classification (CNN) — eurosat-crop-cnn

Classifying 64×64 Sentinel-2 RGB tiles (EuroSAT, 10 classes, 27,000 images) with three CNN set-ups, an error analysis with a validation-selected fix, Grad-CAM, ONNX export and a FastAPI/Docker service.

## Results (test set, 4,050 images, each final model scored once)

| Model | Trainable params | Input | Test accuracy | Test macro-F1 | Train time |
|---|---|---|---|---|---|
| Baseline CNN (from scratch) | 1.18 M | 64 | 0.9817 | 0.9814 | 89 s |
| ResNet-18, frozen backbone (head only) | 5 k | 128 | 0.9286 | 0.9255 | 52 s |
| ResNet-18, fine-tuned | 11.2 M | 128 | 0.9877 | 0.9873 | 93 s |
| ResNet-18, fine-tuned + error-analysis fix | 11.2 M | 224 | **0.9894** | **0.9891** | 218 s |

Validation numbers, per-class reports and full histories are in `reports/metrics/*.json`; logs in `logs/`; MLflow runs in `mlruns/`.

**Error analysis.** On validation, the fine-tuned model's most confused pair was **AnnualCrop ↔ PermanentCrop** (15 errors). Candidate fixes were compared on validation only: 2× loss weight on the pair (11 errors), 224 px input (12), both (9, best). The combination was chosen, then scored on test: pair errors **16 → 11**, accuracy 0.9877 → 0.9894. This is a single-seed result on 4,050 images (a difference of 7 images overall), so treat the improvement as suggestive rather than statistically established; the fix also roughly doubles training time.

Figures (`reports/figures/`): `confusion_matrix_before.png`, `confusion_matrix_after.png`, `gradcam.png`, `training_curves.png`, `model_comparison.png`.

## Setup / protocol
* Split: stratified 70/15/15 by class, seed 42 → 18,900 / 4,050 / 4,050 (`data/splits.csv`; own split, not the dataset's csvs).
* All choices (best epoch, fix) use validation only. `src/evaluate.py` scores test once per model and refuses repeats (`reports/metrics/test_ledger.json`).
* Hardware: RTX 5070 Laptop GPU (8 GB), Intel Core Ultra 9 275HX (24 threads), 31.4 GB RAM, Windows 11, torch 2.11.0+cu128, bf16 autocast.
* The main table uses RGB jpgs only; the 13-band/NDVI stretch is reported separately below.
* Models were trained once each (seed 42); no confidence intervals.

## Stretch: NDVI and multispectral input (13-band GeoTIFFs)
Same ResNet-18 fine-tuning recipe (128 px, 15 epochs), inputs built from the 13-band tifs (`src/bands.py`, `src/train_bands.py`). NDVI = (B08 − B04)/(B08 + B04). Two seeds (42, 43) per variant; every run scored on test once (`src/evaluate_bands.py`); "rgb_tif" is the control (RGB taken from the tifs, since tif and jpg radiometry differ).

| Input channels | Val acc (mean of 2 seeds) | Test acc (s42 / s43, mean) | Test macro-F1 (mean) | Annual↔Permanent test errors | Train time |
|---|---|---|---|---|---|
| RGB from tif (control) | 0.9884 | 0.9879 / 0.9904 (0.9892) | 0.9887 | 11 / 10 | ~82 s |
| RGB + NDVI | 0.9896 | 0.9896 / 0.9923 (0.9910) | 0.9906 | 9 / 8 | ~87 s |
| All 13 bands | **0.9903** | 0.9926 / 0.9926 (**0.9926**) | 0.9923 | **4 / 4** | ~97 s |
| 13 bands + NDVI | 0.9897 | 0.9919 / 0.9923 (0.9921) | 0.9918 | 6 / 8 | ~100 s |

Reading: NDVI alone adds about +0.2 pp over the RGB control; using all 13 bands adds about +0.3 pp and cuts the crop-pair confusion by more than half. Adding NDVI on top of 13 bands gives nothing (a ratio of two bands the network already sees). Seed-to-seed spread (~0.2 pp) is comparable to the NDVI-only gain, so that gain is suggestive; the 13-band gain and the crop-pair reduction are more consistent but still come from 2 seeds. Note these tif-based models are not the same data as the RGB jpg models above (the best jpg model, 0.9894, is not a like-for-like comparison). The deployed API still serves the RGB model.

## Reproduce
```bash
python -m venv .venv && .venv/Scripts/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-dev.txt
# put EuroSAT RGB images in data/EuroSAT/<Class>/*.jpg
python -m src.data                      # split
python -m src.train --stage baseline
python -m src.train --stage frozen
python -m src.train --stage finetune
python -m src.train --stage finetune --name fix_size224_pairw2 --size 224 --pair AnnualCrop PermanentCrop --pair-weight 2.0
python -m src.evaluate baseline frozen finetune fix_size224_pairw2   # test, once
python -m src.figures && python -m src.export
pytest -q tests
```
The notebook `notebooks/eurosat_cnn.ipynb` is executed top to bottom and reads the saved artefacts.

## Serve
```bash
docker build -t eurosat-api . && docker run -p 8000:8000 eurosat-api
curl -F "file=@tile.jpg" localhost:8000/predict
```
Model: `models/eurosat_resnet18.onnx` (max logit difference vs PyTorch 7e-7; ~9 ms/image on CPU).

Interview-style explanation of every step: `docs/INTERVIEW_NOTES.md`.
