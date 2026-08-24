# H7.4 posterior-commutation gate analysis

## Decision

`stop_posterior_commutation`. The H7.3 reproduction guard matches all six locked values exactly and the checkpoint SHA-256 matches, so the decision is valid. Neither soft candidate satisfies all four preregistered criteria. Thresholds remain unchanged.

| candidate | exact-support Brier ratio | residual Pearson | outside mass | full-image Brier ratio | decision |
|---|---:|---:|---:|---:|---|
| raw posterior | 0.659075 | 0.828573 | 0.231295 | 0.763318 | fail outside mass |
| topology-gated posterior | 0.892700 | 0.780727 | 0.152661 | 0.936941 | fail Brier and outside mass |

## Mechanistic reading

Raw posterior commutation contains useful fractional-magnitude information: exact-support Brier improves by 34.09% and residual Pearson is 0.8286. It is not acquisition-specific, however. Its nonzero residual support covers 9,542,518 of 12,517,376 pixels (76.23%), support precision is 1.11%, and 23.13% of residual mass lies outside exact acquisition-change support.

Topology gating controls the spatial spread to 380,975 pixels (3.04%) and retains 91.48% recall, but the operation removes most of the magnitude benefit: exact-support Brier improves only 10.73%, outside mass remains 15.266%, and foreground occupancy is biased low by 4.12%.

This is a representation conflict rather than a near-threshold success. Raw probabilities are locally informative but globally dense; LCC topology is spatially selective but clips the posterior magnitude that H7.4 was meant to recover. The 177 non-clamped stacks also fail, so endpoint replication is not the explanation.

## Consequence

Close posterior commutation and all additional SliceEq target/loss variants. Preserve SliceEqOcc target construction for the next investigation. The only bounded optimization path is operator-side estimation: determine whether independent profile draws yield a high-variance minibatch estimate of the intended acquisition-risk expectation, then test batch-stratified low-order quadrature against the current IID sampler at identical batch and forward cost before authorizing training.

## Evidence limit

The 23k checkpoint contains only the student state. Posterior results use the frozen student in evaluation mode as a proxy, not the historical EMA teacher. This limitation does not invalidate the locked comparison, but prevents a claim about direct EMA posterior fidelity.
