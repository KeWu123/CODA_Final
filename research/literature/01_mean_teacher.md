# Mean Teacher

- Paper: *Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results*.
- Venue/year: NeurIPS 2017.
- Core idea: teacher weights are an exponential moving average of student weights; consistency targets are generated under independent perturbations.
- Relevance: this is the correct algorithmic family for the user's no-Copy-Paste baseline.
- Gap relative to ViSA-MT: it samples perturbations but does not decide whether a particular strong view is valid or useful.
- Code: https://github.com/CuriousAI/mean-teacher
