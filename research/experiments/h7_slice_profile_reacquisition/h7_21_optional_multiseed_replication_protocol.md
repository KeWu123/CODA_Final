# H7.21 Optional Independent-Seed Replication Protocol

## Status

Prospectively registered on 2026-08-24. This experiment is optional and must run only after the frozen MM-WHS evaluation and the compute-matched ablation. No run is authorized by this document alone.

## Method

The only method is the complete frozen `SliceAlign-MPD` implementation:

```text
code/train_sliceeq_occ_oaac_strong_mpd.py
```

No PARS, checkpoint averaging, test-time augmentation, profile redesign, appearance-range change, or additional pseudo-label module is allowed.

## Seeds

Use exactly `1337`, `2024`, and `3407`. Report all three runs. Do not replace a failed or low-performing seed unless the run is invalidated by a documented infrastructure failure before test evaluation.

The existing seed-1337 artifact may be reused only if its code hash, data split, pretraining checkpoint, MPD design artifact, optimizer, validation cadence, and checkpoint selector match this protocol. The test-informed iteration-29000 observation is not reusable as the replication endpoint; the seed-1337 endpoint must be selected by validation only.

## Frozen training contract

- PROMISE12 split: 35 train / 5 validation / 10 test.
- Labeled training cases: 7, using the existing canonical identities.
- Pretraining/self-training: 10,000 / 30,000 iterations.
- Batch: 24 loader samples, 36 student views after warm-up.
- Supervised warm-up: first 1,000 self-training iterations.
- Network, losses, EMA mode/decay, optimizer, learning rate, augmentation ranges, profile grid, MPD constraints, and inference are unchanged.
- One MPD distribution is designed by the same train-only procedure for each run. If the design is deterministic and data-only, its hash must match across seeds.

## Selection and test firewall

For every seed, select exactly one checkpoint by the frozen five-case validation rule. Evaluate the ten-case test split once. No periodic test checkpoint search, best-seed selection, or hyperparameter change is permitted.

## Reporting

Report each seed and mean plus standard deviation for Dice, Jaccard, HD95, and ASD. Also report patient-level paired differences against the frozen matched comparator if that comparator is available for all three seeds. This experiment estimates optimization variability; it does not repair the historical PROMISE12 development leakage.

## Stop rule

Run the three registered seeds at most once each. Regardless of outcome, close this replication without changing seeds or method parameters.
