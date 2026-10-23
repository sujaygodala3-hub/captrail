# Captrail capillaroscopy simulation dataset v2.0.0

Measurements and simulation results from a low-cost USB nailfold capillaroscope (Jiusion 40-1000x USB digital microscope,
640 x 480, on the Captrail cradle), produced by the
Captrail `sims` pipeline. Numbers only: no images, subjects pseudonymized. **Research prototype, not diagnostic.**

## What is in it

| file | rows | one row per |
|---|---|---|
| `tables/sessions.csv` | 5 | capture session |
| `tables/frames.csv` | 264 | image file |
| `tables/loops.csv` | 1486 | detected capillary loop |
| `tables/sites.csv` | 73 | finger site and view |
| `tables/paired_views.csv` | 131 | capture seen in both raw and enhanced view |
| `tables/splits.csv` | 39 | finger site and its split |
| `calibration/ruler_photos.csv` | 3 | ruler photo |
| `calibration/noise_spectrum.csv` | 5 | noise band of the blank card |
| `sims/scenarios.csv` | 115 | simulated scope (parameters) |
| `sims/metrics.csv` | 115 | simulated scope (headline errors) |
| `sims/s1_width_bias.csv` | 40320 | S1 width point |
| `sims/s2_density.csv` | 17250 | S2 density trial |
| `sims/s2_texture.csv` | 4140 | S2 texture trial |
| `sims/s4_misclassification.csv` | 920 | S4 blur level |
| `sims/s6_propagation.csv` | 1380 | S6 width and assumption |
| `sims/s7_repeatability.csv` | 115 | S7 scenario |
| `qa/reproducibility.csv` | 0 | pair of identical inputs |
| `llm/records.jsonl` | 264 | frame, as structured fields plus text |
| `llm/sft_train.jsonl` / `val` / `test` | 1093 / 218 / 224 | chat example |
| `calibration/calibration.json` | | ruler and blank-card summary |

Columns, types and units: `SCHEMA.md`. Hashes of every file and the exact source run: `manifest.json`.

## Sessions

| session | subjects | views | frames | usable measured frames | resolved loops | duplicate inputs ignored |
|---|---|---|---|---|---|---|
| S01 | SUBJ-01;SUBJ-02 | enhanced;raw | 60 | 50/60 | 255 | 0 |
| S02 | SUBJ-03 | enhanced;raw | 48 | 48/48 | 263 | 0 |
| S03 | SUBJ-04 | enhanced;raw | 48 | 48/48 | 264 | 0 |
| S04 | SUBJ-05 | enhanced;raw | 48 | 34/48 | 151 | 0 |
| S05 | SUBJ-06 | enhanced;grey;raw | 60 | 16/30 | 92 | 0 |

## Splits

Whole sites (subject + hand + digit) go to one split, so no finger appears in two splits. Within each session, sites are
dealt in the order of sha256(site_id): about 15% test, 15% val, the rest train, at least one site each, so every session
appears in every split. Scenario lookups (`simulation_lookup`) are all train.

| split | sites | frames | loops |
|---|---|---|---|
| train | 27 | 180 | 988 |
| val | 6 | 42 | 243 |
| test | 6 | 42 | 255 |

## LLM files

| task | examples | what the model learns |
|---|---|---|
| `frame_qc` | 264 | decide if a capture is usable and say how to fix it |
| `loop_measurement` | 1025 | turn pixel widths into estimated micrometres with the right caveats |
| `simulation_lookup` | 115 | expected error for a given scale, blur, contrast and noise |
| `raw_vs_enhanced` | 131 | which view to measure and why |

Fine-tune on `sft_train.jsonl`, pick checkpoints on `sft_val.jsonl`, report once on `sft_test.jsonl`. `records.jsonl` is for
retrieval (embed `text`, filter on `split`). Every answer comes from the pipeline's numbers and rules, so a model trained here learns
Captrail's pipeline, not clinical judgement.

## Limits (read before training or quoting)

- **Not validated.** No human rater has measured these loops yet (S5). Labels are the pipeline's output.
- **Scale is estimated.** `*_um_est` uses 1.6 um/px (bounds 1.3 to 2.0) from healthy anatomy (loop spacing and widths). The calibration card measures 6.102 um/px (0.1 mm ticks; the build guide's own value is 6.094), but at a lower zoom than the nail captures: at that scale the loops would be ~190 um wide and ~1.5 per mm, which is anatomically impossible. Photograph the card at the exact nail zoom to replace the estimate.
- **Small and healthy.** 6 subjects, all healthy volunteers; no disease labels. Do not train a diagnostic classifier on this.
- **Quick-mode sims.** Few replicates per scenario, all computed on one platform. A cross-platform check on dataset v1 found S1 apical bias and S7 repeatability can move between machines (up to 6 points and 0.29 ICC); S4 decisions and density calls did not. Treat S1 and S7 per-scenario values as approximate.
- **Auto labels.** Apex positions and widths in `tables/loops.csv` are detector output, useful as pseudo-labels for a detector after a human check.

## Reproduce

```
python batch_run.py <folder of capture zips> --out results/batch
python export_dataset.py results/batch --out datasets/captrail-v2
python export_dataset.py --check datasets/captrail-v2
```

Source run: batch code `40e81b42b447`, python 3.10.12, numpy 2.2.6, scipy 1.15.3, Linux. The same inputs give byte-identical outputs on one machine (`qa/reproducibility.csv`). 575 sim steps were computed on a second machine (Linux) from byte-identical scope files and merged by cache key, so all sims in this release come from one platform.

Images stay on the team's computers. To build an image + keypoint set locally (never committed):

```
python export_dataset.py results/batch --out datasets/captrail-v2 --vision-images <folder of zips> --vision-out datasets/_local_vision
```
