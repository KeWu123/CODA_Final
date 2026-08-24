# A Simple Baseline for Semi-Supervised Semantic Segmentation With Strong Data Augmentation

- Authors: Jianlong Yuan, Yifan Liu, Chunhua Shen, Zhibin Wang, Hao Li
- Venue: ICCV 2021
- URL: https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_A_Simple_Baseline_for_Semi-Supervised_Semantic_Segmentation_With_Strong_Data_Augmentation_ICCV_2021_paper.html

## Key finding

The paper identifies strong-augmentation distribution shift as harmful to ordinary
BatchNorm statistics and introduces distribution-specific BatchNorm plus a
self-correction loss. It demonstrates that apparently minor training details can
dominate semi-supervised segmentation results.

## Relevance to OBA

OBA changes the student BN mixture from `12L+12U` to `12L+12U+ +12U-`. Its late
collapse therefore cannot automatically be attributed to orbit geometry; a
compute-matched two-IID-view control with the same effective batch is mandatory.

