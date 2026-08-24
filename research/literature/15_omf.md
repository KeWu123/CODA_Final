# Overlay Mantle-Free (OMF)

- Venue/year: MICCAI 2024.
- Core idea: overlay augmentation sharpens edges during teacher pretraining; a frozen teacher and differentiated teacher/student inputs then teach the student to infer hidden shape.
- Relevance: demonstrates both the appeal and reviewer risk of augmentation-only medical SSL. MICCAI reviews explicitly questioned novelty and single-dataset generality.
- Gap relative to ViSA-MT: OMF proposes a particular overlay and two-stage distillation; ViSA-MT proposes a view-selection principle and avoids cross-patient overlay.
- Paper/reviews: https://papers.miccai.org/miccai-2024/585-Paper0481.html
- Code: https://github.com/vigilliu/OMF
