# SliceAlign-MPD Overleaf draft

This directory is a self-contained CVPR paper draft for the single complete method **SliceAlign-MPD**.

## Paper identity

- Paper method: **SliceAlign-MPD**.
- The manuscript presents two coupled scientific decisions: paired slice-profile supervision and moment-constrained profile distribution design.
- Fixed appearance perturbations and teacher--student infrastructure are implementation settings, not proposed modules.
- Internal script names and later exploratory candidates are deliberately excluded from the manuscript.

## CVPR 2026 / Overleaf

This folder follows the official CVPR 2026 Author Kit structure:

```text
main.tex
preamble.tex
main.bib
cvpr.sty
ieeenat_fullname.bst
sec/
fig/
```

Upload the complete `mpd_overleaf_draft` folder as a new Overleaf project and compile `main.tex`. It is configured for `\usepackage[review]{cvpr}` and anonymous review. Replace `\paperID{*****}` after a paper ID is assigned. For the camera-ready version, switch to `\usepackage{cvpr}` and fill in the real authors and affiliations.

## Result firewall

- `0.854573` is a PROMISE12 development observation from an iteration selected after test inspection.
- The validation-selected MPD test Dice is `0.848952`.
- Neither value should be advertised as an untouched primary test result.
- The primary submission claim remains blank until the frozen MM-WHS evaluation and validation-only replication are complete.
- Blank result cells render as `--` and must be replaced from audited artifacts only.

## Planned paper tables

1. PROMISE12 7-label and 11-label comparison under one split and selector.
2. MM-WHS MRI cross-organ, seven-structure external evaluation.
3. Four-row controlled ablation: compute match, image-only profile, paired fractional profile, and the complete method.
4. Optional three-seed replication, performed last with no seed selection.

The table structure follows the evidence pattern used by strong semi-supervised segmentation papers: matched main comparisons, cross-dataset validation, causal ablation, mechanism analysis, sensitivity, statistics, and efficiency.
