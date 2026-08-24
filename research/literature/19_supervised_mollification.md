# Supervised Mollification

- Paper: *Robust Classification by Coupling Data Mollification with Label Smoothing*.
- Venue/year: AISTATS 2025.
- Core idea: couple image noising/blurring severity to label smoothing so prediction confidence reflects input degradation.
- Evidence: the authors report improved corruption robustness and uncertainty quantification on CIFAR, TinyImageNet, and ImageNet.
- Relevance: this is the strongest direct support for the claim that data and target certainty should be augmented jointly.
- Collision risk: high at the principle level. CoDA-MT cannot claim that coupling degradation with label entropy is new in general.
- Required distinction: semi-supervised dense prediction; pseudo-label rather than ground-truth uncertainty; spatially varying evidence loss rather than one global class label; medical acquisition corruptions and boundary behavior.
- Paper: https://proceedings.mlr.press/v258/heinonen25a.html
- Code: https://github.com/markusheinonen/supervised-mollification

