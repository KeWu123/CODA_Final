# PROMISE12 Dataset and Evaluation

- Challenge: Prostate MR Image Segmentation 2012.
- Data: multi-center, multi-vendor, multi-protocol transverse T2-weighted prostate MRI; original challenge used 50 training, 30 test, and 20 live-challenge cases.
- Acquisition variation includes field strength, endorectal coil usage, in-plane resolution, through-plane resolution, and manufacturer differences.
- Relevance: supports an acquisition-robustness motivation, but its small labeled training set makes evaluation variance and leakage serious concerns.
- Recommended protocol: patient-level five-fold cross-validation on the 50 labeled training cases, with label hiding only inside the training fold and multiple labeled-subset draws.
- Metrics: Dice plus boundary/surface and volume metrics; include HD95/ASD and confidence intervals.
- Primary paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC4137968/
- Dataset archive: https://zenodo.org/records/8014041
