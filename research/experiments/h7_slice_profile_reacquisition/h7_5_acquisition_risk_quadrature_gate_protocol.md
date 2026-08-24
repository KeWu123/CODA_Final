# H7.5 acquisition-risk quadrature gate

## Status

Drafted after H7.4 failed its locked zero-training fidelity gate. This is a read-only variance-and-integration test. It does not authorize a training run.

## Hypothesis

SliceEqOcc intends to optimize expected paired acquisition risk

`R(theta) = E_{sigma, phase}[L(f_theta(A_{sigma,phase} X), A_{sigma,phase} Y)]`,

but the current implementation independently samples one profile per sample and step. H7.3 observed acquisition-active residuals in only 128/191 labeled and 120/192 unlabeled sampled stacks, while 0.8427% of exact residual pixels supplied 65.65% of the full CE+Dice gradient. The resulting minibatch gradient may therefore be a noisy estimate of the same unchanged joint data-acquisition risk. Batch-stratified low-order quadrature over the two-dimensional physical profile domain may reduce estimator variance without changing the method's target, loss, network, EMA, schedule, batch size, or inference.

## Locked variants

1. Current estimator: 12 independent profile draws within each labeled/unlabeled branch.
2. Candidate estimator: batch-stratified 2x2 Gauss-Legendre nodes over the existing uniform profile distribution.
3. Dense deterministic per-sample reference grid used only to estimate integration error.

For the candidate, the nodes are fixed before execution:

- `sigma = 0.65 +/- 0.20 / sqrt(3)`;
- `phase = 0 +/- 0.25 / sqrt(3)`;
- Cartesian product of the two sigma and two phase nodes;
- equal weight `1/4` per node.

Each 12-sample branch receives the four nodes exactly three times per step. Node-to-sample assignment is independently permuted for labeled and unlabeled branches with fixed seeded generators; it must not depend on image, mask, confidence, loss, or anatomy. Every sample still produces exactly one re-acquired view. The student input remains 36 images: 12 original labeled, 12 re-acquired labeled, and 12 re-acquired unlabeled.

The exact-GT labeled and detached hard-LCC unlabeled occupancy targets remain exactly those of SliceEqOcc. No posterior, confidence, topology, residual, or class weighting is permitted.

## Read-only metrics

Using the frozen iteration-23k checkpoint and preregistered fixed minibatches, report separately for labeled and unlabeled proxy branches over repeated IID draws and repeated seeded node-to-sample permutations:

- minibatch absolute and relative error of estimated expected CE+Dice risk against the dense per-sample reference;
- last-decoder-layer parameter-gradient cosine and relative norm error against the dense-reference batch gradient;
- across-repeat variance of current IID sampling and stratified node assignment;
- expected fractional-support fraction and residual mass, verifying that stratification estimates rather than changes the profile risk;
- forward-pass, batch-size, and memory equality.

## Pass/kill rule

Authorize a full H7.5 training entry only if all conditions hold:

1. batch-stratified quadrature reduces median minibatch risk-estimation error by at least 30% versus current IID sampling;
2. it wins on at least 60% of preregistered minibatches;
3. mean last-layer gradient cosine to the dense reference is at least 0.95 and its relative norm error is at most 0.70x current IID;
4. expected residual mass differs from the dense reference by at most 5%;
5. batch size remains 36, every sample has one re-acquired view, and measured forward count and peak memory differ from current SliceEqOcc by at most 5%.

If the gate fails, stop SliceEq optimization rather than adding more target/loss/sampling modules. Preserve H7.2 as the exploratory method and spend the next runs on mandatory matched controls, checkpoint-selection repair, and cross-dataset validation.

## Conditional training authorization

A passing candidate may create a new parallel training entry. It must replace only `sample_slice_profiles` with independent seeded permutations of the fixed three-repeats-per-node tables and keep all SliceEqOcc defaults otherwise unchanged. The first full-run comparison is current IID SliceEqOcc versus batch-stratified quadrature from the identical pretrained checkpoint and seed.
