# MixMatch: A Holistic Approach to Semi-Supervised Learning

- Authors: David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot,
  Avital Oliver, Colin Raffel
- Venue: NeurIPS 2019
- URL: https://proceedings.neurips.cc/paper/2019/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html

## Key finding

MixMatch averages model predictions across multiple augmentations of an unlabeled
example, sharpens the resulting guessed label, and combines this with MixUp and an
unlabeled consistency objective.

## Relevance to OBA

Simple averaging of the OBA endpoint predictions before the loss is not itself a novel
headline. If used, a probability-barycenter objective is only a stability diagnostic;
the defensible contribution would still have to come from antithetic orbit sampling
and evidence that it beats IID views.

