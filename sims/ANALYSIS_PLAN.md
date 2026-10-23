# Analysis plan (fixed before any real data)

Thresholds below are set in advance so the results cannot be tuned to look good. They are
project conventions unless a source is named. Change them only before opening the data, and write
down the change.

## S0 acceptance (run_s0.py)

| Check | Rule | Why |
| --- | --- | --- |
| Ruler ticks found | at least 8 ticks in every slide capture; no capture skipped | A short ruler gives a poor pitch fit |
| um per pixel repeatability | CV across the 10 captures at most 1.5%, session means within 1.5% | Every width inherits this error |
| Photon transfer linear | r2 above 0.98 and gain above 0 | Otherwise the camera applies gamma or auto exposure; gain and read noise stay ASSUMED |
| PSF fit | fit residual under 10% of tick depth | A bad fit means the PSF is not Gaussian or the tick width is wrong |
| Gradient and contrast | measured, not assumed | Both change S1 and S2 |

If any check fails, fix the capture, do not change the rule.

## Sweeps (run_sims.py)

- Width tolerance: |bias| at most 10% of the true width (S1). The smallest limb meeting it at the measured sigma and pixel size is the resolvable limb.
- Giant-call tolerance: combined error rate at most 5% for 45 versus 55 um (S4).
- Density call: report P(7/mm read as reduced) and the ROC area for "under 7/mm" with a bootstrap interval (S2, S4).
- Every table reports the number of replicates; every proportion has a Wilson 95% interval; every mean has a 95% interval.

## S5 gate (all must hold before S1 to S4 are reported as validated)

1. The two raters agree: inter-rater ICC(2,1) at least 0.85 (Dinsdale 2017 reports 0.85 for width).
2. The simulated mean tool-minus-truth bias is consistent with the real mean tool-versus-raters bias: within 1.96 standard errors of it (the real SE is SD/sqrt(n), combined with the simulation's SE). The limits of agreement are NOT used: they describe individual differences and would pass almost any simulation.
3. The simulated SD of the tool error is between 0.5 and 2 times the real SD.
4. Real versus simulated slide profile: normalized RMSE at most 0.20 after alignment.
5. At least 10 capillaries, measured by both raters blind to the tool and to each other.

If any item fails, the report says "model not validated" on every S1 to S4 figure and the write-up says so.

## D1: synthetic training images versus real images (before any model is trained on synthetic data)

Frames from `make_training_set.py` (colour, four morphology classes, YOLO labels) are compared with a held-out half of the real frames (the dataset's Morphology_detection images, and your own microscope captures) using `domain_gap.py`. Target: classifier two-sample AUC at most 0.75 on held-out real frames, with the worst feature gaps listed. A low AUC does not prove the shapes are right, so the training claim is always made on REAL validation images: train on synthetic plus real, report mAP on real held-out frames, and report the real-only baseline next to it. The dataset's own published baseline is mAP50 about 0.78 and mAP50-95 about 0.44 on its validation split (from its repo's training log); quote ours against that, not against a synthetic test set.

## S7

The simulated ICC (absolute agreement) must fall inside the bootstrap 95% interval of the real repeat-capture ICC, or the discrepancy is reported as a limit of the model. With a single finger repeated 10 times only the within-capture SD, CV and repeatability coefficient can be compared.

## Model-misspecification (S8) and sensitivity

S8 reports the width-bias gap when the true PSF is a disc or exponential instead of Gaussian. Carry the largest gap as extra uncertainty on S1. The Monte Carlo interval from `sens` is the uncertainty on S1 and S4 headline numbers; quote the interval, not the point value.

## Wording rules

- Say "model output" until S5 passes.
- Say which scope values were measured and which were assumed (printed in every REPORT.md).
- Never say "diagnose" or "screen". Apical widths are compared with the EULAR thresholds as measurements.
- Poiseuille results are error amplification in an idealized tube.
