# H2 Protocol: MRI Acquisition Perturbations

## Hypothesis

The validity-informativeness selector benefits more from a perturbation bank that approximates MRI acquisition variability than from generic image transforms, especially on held-out centers/vendors.

## Design

- Keep the Mean Teacher, selector, optimizer, and label split identical to H1.
- Compare generic versus MRI-oriented banks under matched candidate count and measured severity.
- Stratify PROMISE12 folds by center/vendor if reliable metadata are available.
- Add leave-one-center-out evaluation only if center labels can be verified from primary dataset metadata.

## Critical Comparison

This hypothesis must be distinguished from MiDSS/FedDG/frequency-mixing work. The proposed bank should model acquisition effects without transferring pixels or Fourier amplitudes between patients. The contribution is selection of trustworthy counterfactual acquisition views, not domain interpolation.
