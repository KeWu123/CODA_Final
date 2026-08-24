# GeoLS

- Paper: *GeoLS: Geodesic Label Smoothing for Image Segmentation*.
- Venue/year: MIDL 2024.
- Core idea: create spatially informed soft labels using image geodesic distances, explicitly modeling image gradients and relationships near class boundaries.
- Relevance: direct evidence that dense medical segmentation needs spatial rather than classification-style label smoothing.
- Collision risk: CoDA cannot claim that image-aware or spatial label smoothing for segmentation is new.
- Required distinction: GeoLS is supervised and constructs static soft ground truth from image context. CoDA transforms an unlabeled pseudo-target using the realized evidence loss of the sampled strong augmentation.
- Paper: https://proceedings.mlr.press/v227/vasudeva24a.html
- Code availability audit (2026-08-10): the previously indexed URL
  `https://github.com/anonymous35783578/GeoLS` currently returns 404. Treat the
  paper as the primary reference and do not claim that its code was reused.
