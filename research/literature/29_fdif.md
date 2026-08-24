# FDIF: Formula-Driven Supervised Learning with Implicit Functions

- Authors: Yukinori Yamamoto et al.
- Year: 2026
- Status: arXiv preprint, submitted to ECCV 2026
- Primary source: https://arxiv.org/abs/2603.23199

FDIF constructs fully synthetic 3-D labeled volumes from signed-distance implicit
functions. Its surface-driven intensity mapper assigns appearance as a function of
distance to the generated boundary, alongside geometric surface perturbation. It uses
procedural functions and no real patient data for supervised pretraining.

Relevance: FDIF occupies the broad claim of using signed distance to control boundary
appearance. BMER can no longer claim novelty for distance-conditioned intensity
synthesis itself. A remaining distinction is empirical conditional evidence estimated
from real unlabeled scans, transported to fixed real labeled anatomy while preserving
recipient residuals and exact identity outside a narrow band. If BMER collapses to a
distance-only mapper, the novelty claim fails.

