# SGRS-Net

- Paper: *Synergy-Guided Regional Supervision of Pseudo Labels for Semi-Supervised Medical Image Segmentation*.
- Venue/year: MICCAI 2025.
- Core idea: compare behavior before and after mix augmentation, partition pseudo-label regions using synergy, and evaluate regional losses separately.
- Relevance: a close collision for any method framed as checking augmentation validity or partitioning supervision after augmentation.
- Consequence: further weakens ViSA-MT as a headline because transformation response and regional supervision are occupied.
- Difference from CoDA: SGRS uses mix augmentation and region partition/loss evaluation; CoDA defines a continuous target-distribution transform from the realized degradation channel and avoids cross-patient mixing.
- Paper: https://papers.miccai.org/miccai-2025/0890-Paper1721.html

