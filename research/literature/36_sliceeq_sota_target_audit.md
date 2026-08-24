# SliceEq PROMISE12 SOTA target audit (2026-08-11)

## Question and scope

What public result must SliceEq exceed on PROMISE12 under semi-supervised
training with 20% labeled training volumes? Local project results are explicitly
excluded. The primary comparison class is 50 PROMISE12 volumes split 7:1:2
(35/5/10), seven labeled training volumes, 2-D segmentation, and Dice measured
on the held-out ten volumes. Full supervision, challenge-server submissions,
cross-domain adaptation, interactive segmentation, and different label budgets
are not treated as the same leaderboard.

## Evidence hierarchy

| Scope | Method | Venue | Setting | Dice | Other reported metrics | Interpretation |
|---|---|---|---|---:|---|---|
| Closest accepted top-conference peer | PSC | MICCAI 2024 | 35/5/10, 7 labeled, 2-D U-Net | 83.64 | HD95 4.58, ASD 2.04 | Direct coarse-protocol comparator |
| Closest accepted top-conference peer | beta-FFT | CVPR 2025 | 7/35 labeled, PROMISE12 20% | 83.75 +/- 0.65 | ASD 1.20 +/- 0.07 | Strong pre-2026 top-conference comparator |
| Current accepted top-conference frontier | PMPC | AAAI 2026 | 7:1:2, 20% labeled, U-Net; three runs | **85.80 +/- 1.07** | Jaccard 77.74 +/- 0.93, 95HD 3.23 +/- 0.18, ASD 0.68 +/- 0.08 | Public top-conference SOTA found for this benchmark setting |
| Broad published numerical frontier | Dual-teacher B2CP | BSPC 2026 | PROMISE12, 20% labeled | **87.50** | Abstract-level value | Broader ceiling, but architecture/protocol differs: dual teachers, CE-Net/SCFR and pretrained encoder; not an apples-to-apples U-Net result |

## Decision

- For a statement restricted to the conventional 2-D U-Net/top-conference
  PROMISE12 20% protocol, SliceEq must exceed **85.80 Dice**.
- For an unqualified claim of overall published PROMISE12 semi-supervised SOTA,
  it must exceed **87.50 Dice** and reproduce the competing protocol carefully.
- A one-seed result of 85.9 is only a screening success, not robust evidence of
  SOTA, because PMPC reports 85.80 +/- 1.07 across three runs. For the currently
  requested fixed-seed screen, use **>=86.0** as the minimum continuation gate
  and **>=87.5** as the ambitious publication-facing numerical target.
- Dice alone is insufficient. A credible claim must also report Jaccard and
  physically correct HD95/ASD, use the same held-out cases, and distinguish the
  fixed U-Net baseline from foundation-model or pretrained-backbone comparisons.

## Primary sources

1. He et al., Pair Shuffle Consistency, MICCAI 2024:
   https://papers.miccai.org/miccai-2024/paper/0942_paper.pdf
2. Hu et al., beta-FFT, CVPR 2025:
   https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_beta-FFT_Nonlinear_Interpolation_and_Differentiated_Training_Strategies_for_Semi-Supervised_Medical_CVPR_2025_paper.pdf
3. Wang et al., PMPC, AAAI 2026:
   https://ojs.aaai.org/index.php/AAAI/article/download/37998/41960
4. Fang et al., dual-teacher B2CP, Biomedical Signal Processing and Control 2026:
   https://www.sciencedirect.com/science/article/abs/pii/S1746809426005392

## Comparability caveat

A 7:1:2 ratio does not prove that two papers used the identical patient IDs in
each split. Public papers often omit ordered split lists. Therefore 85.80 and
87.50 are literature frontiers, not guaranteed values on the project's exact
ten-case test list. A paper should rerun available competitors on one published
split or state clearly which values are quoted versus reproduced.
