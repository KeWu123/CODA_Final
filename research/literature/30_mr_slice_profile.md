# MR Slice Profile Estimation by Learning to Match Internal Patch Distributions

- Authors: Shuo Han et al.
- Year: 2021
- Primary source: https://arxiv.org/abs/2104.00100

The paper treats the MR slice-selection profile as a physical degradation model from
high to low through-plane resolution and estimates it by matching internal in-plane and
through-plane patch distributions. It is a reconstruction/super-resolution method,
not a segmentation augmentation method.

Relevance: it supports the physical basis of the backup stochastic virtual
re-acquisition direction. It also narrows the novelty claim: the slice profile itself
is established prior art, so a future contribution must concern label-preserving,
task-validated acquisition randomization for slice-wise semi-supervised segmentation,
not merely applying a through-plane blur.

