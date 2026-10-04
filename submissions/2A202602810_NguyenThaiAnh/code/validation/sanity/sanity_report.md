# Sanity checks — real train images

- Status: PASS; ResNet-50 ImageNet pretrained, seed 42.
- Batch contract, optimizer groups, Focal/LS equivalence, Mixup/CutMix: PASS.
- Frozen: only head changes; all backbone weights and BN buffers stay identical.
- Evaluate: no gradients/state changes; filename order and prediction CSV round-trip: PASS.
- Initial CE: 2.208589; ln(9): 2.197225 (reference only).
- Overfit: 40 updates on 18 train images; eval-mode CE 0.014244, top-1 100.0%.
- Skipped updates: 0; final training CE 0.005834.
- No val/test images evaluated; these numbers do not measure generalization.
- See summary.json, train_batch.csv, overfit_history.csv and curves/sanity/overfit_train_batch.png.
- Next: one full pretrained train/val epoch to verify run outputs before B01–B05.
