# Official Code Index and Reuse Notes

| Method | Repository | Status | Most useful reuse |
|---|---|---|---|
| BCP | [DeepMed-Lab-ECNU/BCP](https://github.com/DeepMed-Lab-ECNU/BCP) | Public | U-Net/V-Net conventions, two-stream sampler, EMA update, ACDC/LA evaluation |
| UniMatch | [LiheYoung/UniMatch](https://github.com/LiheYoung/UniMatch) | Public | Strong FixMatch/weak-to-strong reference, dual strong streams, medical scenario logs |
| AugSeg | [ZhenZHAO/AugSeg](https://github.com/ZhenZHAO/AugSeg) | Public | Continuous-magnitude augmentation implementation and adaptive injection baseline |
| ABD | [chy-upc/ABD](https://github.com/chy-upc/ABD) | Public | Confidence rank maps and fair medical SSL comparison setup |
| OMF | [vigilliu/OMF](https://github.com/vigilliu/OMF) | Public | Overlay/mantle implementation; useful collision baseline, not a component to copy into ViSA-MT |
| MiDSS | [MQinghe/MiDSS](https://github.com/MQinghe/MiDSS) | Public | Multi-domain prostate setup and progressive amplitude mixing comparison |
| Task-driven augmentation | [krishnabits001/task_driven_data_augmentation](https://github.com/krishnabits001/task_driven_data_augmentation) | Public | Prostate intensity/shape transformation generator baseline |
| MisMatch | [moucheng2017/MisMatchSSL](https://github.com/moucheng2017/MisMatchSSL) | Public | Morphological feature perturbation and calibration evaluation |
| SemiGDA | [taozh2017/SemiGDA](https://github.com/taozh2017/SemiGDA) | Public | Current CVPR 2026 medical SSL baseline |
| Conformal risk control | [aangelopoulos/conformal-risk](https://github.com/aangelopoulos/conformal-risk) | Public | Reference implementation if a later risk-calibrated extension is pursued |
| SF-DA | [ZYS-four/SF-DA](https://github.com/ZYS-four/SF-DA) | Placeholder only | Repository explicitly states implementation will not be released; paper claim and repository conflict |
| Supervised Mollification | [markusheinonen/supervised-mollification](https://github.com/markusheinonen/supervised-mollification) | Public | Closest reference for coupling known degradation severity to target entropy |
| FixMatch | [google-research/fixmatch](https://github.com/google-research/fixmatch) | Public | Canonical hard-pseudo-label weak-to-strong comparison |
| GeoLS | [previously reported repository](https://github.com/anonymous35783578/GeoLS) | Unavailable (404 on 2026-08-10) | Paper-level image-aware spatial label-smoothing comparison; no code reused |

## BCP Code Audit

The official `ACDC_BCP_train.py` uses:

- a supervised Copy-Paste pretraining stage;
- a student and EMA model initialized from that pretraining;
- teacher pseudo-labels for unlabeled images;
- a fixed rectangular mask covering roughly two-thirds of each image dimension;
- two bidirectional mixed inputs (unlabeled-in-labeled and labeled-in-unlabeled);
- region-weighted Dice and cross-entropy losses;
- connected-component post-processing for ACDC pseudo-labels;
- EMA decay `0.99` in the released script.

Therefore, removing only the mixing lines while retaining pretraining, mask-specific loss weighting, or connected-component cleanup does not automatically yield a canonical Mean Teacher. The baseline code needs an explicit diff/audit before experiments.

## Integration Recommendation

Implement CoDA-MT first as a small target-construction wrapper around a clean Mean Teacher training step:

1. `augment_with_metadata(x)` returns the strong view, coordinate transform, and known degradation parameters.
2. `evidence_loss_map(x_weak, x_strong, metadata)` returns spatial `gamma` values.
3. `align_teacher_target(q, metadata)` applies exact geometric equivariance.
4. `soften_target(q_aligned, gamma)` constructs the evidence-coupled dense pseudo-target.

Retain ViSA-MT's candidate selection as a secondary comparison, not as a dependency of the minimal method.

Avoid forking several complete repositories. Reuse only dataset/evaluation utilities after checking licenses and split protocols.
