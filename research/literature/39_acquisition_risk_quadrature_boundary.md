# Literature boundary for H7.5 acquisition-risk quadrature

## Positive basis

Balestriero, Misra, and LeCun, *A Data-Augmentation Is Worth A Thousand Samples* (NeurIPS 2022), explicitly analyzes expectation and variance under augmentation sampling and shows that common augmentation risks may require many samples to estimate accurately. This supports measuring the Monte Carlo error of SliceEq's profile risk before another full run.

Chen, Dobriban, and Lee, *A Group-Theoretic Framework for Data Augmentation* (NeurIPS 2020), frames augmentation as averaging over transformation orbits and derives variance-reduction effects. Slice-profile re-acquisition is not a label-invariant group action, so this is motivation for integration, not a direct theorem for SliceEq.

## Collision boundary

The general claim that augmentation expectations can be integrated or variance-reduced is not novel. H7.5 must therefore remain a bounded optimization of SliceEq's main paired-acquisition idea, not a standalone claim that quadrature itself is new.

Generic alternatives are more crowded and are not recommended here: AugSeg (CVPR 2023) and SAA (ICCV 2023) already cover adaptive augmentation in SSL, while POS (ICCV 2025) directly addresses supervised/unsupervised gradient conflict and magnitude. H7.5 must not add sample hardness selection, confidence-adaptive severity, or gradient surgery.

## Defensible distinction

The specific object being integrated is a low-dimensional, paired MRI forward operator acting jointly on neighboring-slice images and fractional tissue occupancy. The scientific question is whether batch-stratified coverage of the unchanged physical acquisition distribution stabilizes paired non-label-invariant supervision at identical compute. Publication novelty remains SliceEq's paired forward model; quadrature is an estimator and stability component whose value must be demonstrated against the current IID sampler.
