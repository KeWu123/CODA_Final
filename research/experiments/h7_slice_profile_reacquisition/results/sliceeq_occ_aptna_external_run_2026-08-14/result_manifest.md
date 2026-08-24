# SliceEqOcc APTNA external result manifest

## Provenance

- Imported: 2026-08-14
- Source training log: `Z:\\Downloads\\log.txt`
- Source test report: `Z:\\Downloads\\performance.txt`
- Source training-log SHA-256: `83D2417CAF6A9E095047242C03C4548145454F05BA8FA2F53C6007793EE1E40D`
- Source test-report SHA-256: `C3069B532B3A7F6E8617CA1CB7E304D3EB1B6F6EA5912527A00A71170CA15D65`
- Archived training log: `training_log.txt` (byte-identical to source)
- Archived test report: `test_performance.txt` (text content preserved; archive has one normalized final newline)

## Run identity

- Experiment: `SliceEqOccAPTNA_PROMISE12`
- Seed: `1337`
- Shared pretrain SHA-256: `49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9`
- Loader batch/labeled batch: `24/12`
- Main student forward: 36 views
- Auxiliary native-U forward: 12 views with the locked BN/RNG isolation
- Training: complete through iteration 30,000
- Validation: unchanged five-case SliceEqOcc rule, every 200 iterations

## Artifact-backed result

- Best validation: `0.804265` at iteration `28,800`
- Final validation: `0.772908` at iteration `30,000`
- Tested checkpoint: validation-selected `unet_best_model.pth`
- Test Dice/Jaccard/HD95/ASD: `0.829420 / 0.711343 / 6.888081 / 2.203267`

The user additionally reports approximately `0.835` test Dice for the penultimate periodic
checkpoint. Under the 3,000-iteration periodic-save contract this most likely denotes iteration
27,000, whose logged validation Dice is `0.781877`. No checkpoint-specific performance artifact
for that observation was supplied, so it is recorded as a diagnostic, test-inspected result and
not as the unchanged validation-selected outcome.

## Decision

APTNA is negative. Do not tune its native coefficient, cutoff, ramp, or checkpoint using the
PROMISE12 test set. Together with H7.8 DA, this closes the native-anchor optimization family.

