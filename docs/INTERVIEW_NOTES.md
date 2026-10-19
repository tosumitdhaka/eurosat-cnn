# Interview notes — plain-language walkthrough

1. **Problem.** Given a small satellite photo (64×64 px), say whether it shows forest, river, highway, crops, etc. (10 classes, 27,000 labelled tiles).
2. **Split.** I split 70/15/15 per class with a fixed seed. Train is for learning, validation is for decisions (which epoch, which fix), test is a final exam touched once per model. A ledger file refuses repeat test runs so I can't quietly tune on the exam.
3. **Baseline CNN.** A small 4-block convolutional network trained from scratch. It gives a reference: 98.2 % test accuracy, which is already high because EuroSAT is an easy dataset.
4. **Transfer learning, frozen.** I took ResNet-18 pretrained on ImageNet and only trained a new final layer. It got 92.9 % — worse than the baseline, because ImageNet features of everyday photos don't transfer perfectly to top-down 64-px imagery without adaptation.
5. **Fine-tuning.** Same network, but all layers trained with a small learning rate, a one-cycle schedule, label smoothing and flip/rotation augmentation (aerial images have no "up"). 98.8 %. Pretraining helps once the features are allowed to adapt.
6. **Training details worth knowing.** AdamW, bf16 mixed precision for speed on the RTX 5070, images upsampled to 128 px so ResNet's downsampling doesn't shrink the 64-px tile to almost nothing, best epoch chosen by validation accuracy.
7. **Error analysis.** The confusion matrix shows where the model is wrong. Annual vs permanent crops were most confused: both are farmland, differing mainly in subtle row/parcel texture. Grad-CAM heat-maps show what the network looks at.
8. **The fix.** Candidates (upsample to 224 px so fine texture has more pixels; double the loss weight of the two crop classes; both) were compared on validation. The combination had fewest pair errors, so it was chosen, then evaluated on test: crop-pair errors 16 → 11 and accuracy 98.77 → 98.94 %. Caveat: one seed, ~7 images difference overall — suggestive, not proven; cost is ~2× training time.
9. **Honesty.** Every number comes from a saved run (metrics JSON, logs, MLflow). I skipped the 13-band NDVI stretch.
10. **Deployment.** Exported to ONNX (verified against PyTorch to 1e-6), served with FastAPI `/predict`, packaged in a Docker image, one pytest checks the endpoint.
