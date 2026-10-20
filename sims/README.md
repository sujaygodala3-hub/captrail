# sims: Captrail simulation core

Synthetic capillary, optical forward model, automated measurement, error curves, and the
tools that tie them to the real Celestron. **All numbers are model outputs until S5 passes.**
Anything computed from a scope value that was not measured is stamped ASSUMED on its figures.

```
pip install -r requirements.txt
python -m pytest -q                                   # 70 tests, ~1 min
python run_s0.py                                      # real calibration -> results/scope_params.json
python run_sims.py all --scope results/scope_params.json
python run_sims.py all --quick                        # smoke run on assumed defaults (not a result)
python run_experiments.py                             # legacy E1 to E6 first pass
```

## Workflow once the Celestron captures exist

| Step | Command | Needs (see `data/README.md`) | Answers |
| --- | --- | --- | --- |
| S0 | `run_s0.py` | slide, blank-card and gradient captures | True um per pixel, PSF sigma, gain, read noise, lighting gradient |
| S1 | `run_sims.py s1` | S0 | Limb, apical and total width bias vs blur and pixel size; smallest limb under 10% bias |
| S2 | `run_sims.py s2` | S0 | Apex-count density error vs blur, contrast, texture; ROC for "under 7/mm" |
| S3 | `run_sims.py s3` | oil / no-oil captures | Contrast and width bias vs depth, with and without oil (phenomenological) |
| S4 | `run_sims.py s4` | S0 | P(45 um read as 50+), P(55 um read as under 50), P(7/mm read as under 7); largest safe sigma |
| S5 | `run_sims.py s5` | slide profile, two raters' blind widths | Gate: does the simulated bias fall inside the real limits of agreement |
| S6 | `run_sims.py s6` | S0 | Width error amplified into resistance and shear (4 held-fixed assumptions) |
| S7 | `run_sims.py s7` | 10 re-seated captures | Simulated vs real repeatability |

If S5 fails, fix the model before reporting S1 to S4. Outputs go to `results/<tag>/`, where the tag
is `measured` only when every scope value came from S0.

## Modules

| Module | Does |
| --- | --- |
| `capsim/scope.py` | `ScopeParams`: the numbers S0 measures; each is flagged measured or ASSUMED |
| `capsim/calibration.py` | S0: um/px from slide ticks, PSF sigma from ticks or an edge, photon transfer (gain, read noise), lighting gradient |
| `capsim/geometry.py` | Hairpin loops with exact ground truth; separate limb, apical and total widths (`make_loop`); loop rows with exact density (`loop_row`); disc-union renderer |
| `capsim/optics.py` | `I = S[h * (L * I_ideal)] + n` with Gaussian PSF, lighting gradient, skin texture, position jitter, Poisson and read noise, quantization; phenomenological `depth_model` |
| `capsim/measure.py` | Flat-fielding; half-depth, fixed-threshold and Otsu limb width; apical and total width; apex-based density; image-quality gate |
| `capsim/sweeps.py` | S1 to S7 experiments |
| `capsim/propagate.py` | S6: Poiseuille error amplification, baseline shear, Monte Carlo summary |
| `capsim/stats.py` | Bland-Altman, ICC(2,1) and (3,1) with bootstrap interval, repeatability coefficient, ROC area |
| `capsim/analytic.py` | Closed-form blurred-bar half-depth width (S1 pass test) |
| `capsim/psf_fit.py`, `capsim/intake.py` | Single-tick sigma fit; image loading and QC |
| `run_s0.py`, `run_sims.py`, `plots_s.py` | CLIs and figures |

## What is verified, what is assumed

- Tested: slide um/px, sigma, gain, read noise and gradient are recovered from a simulated session with known truth (`tests/test_run_s0_end_to_end.py`); half-depth width matches the closed form within 2% (`test_half_depth_matches_closed_form_bar`); stats functions on known cases.
- **Assumed until you measure them:** um/px, sigma, gain, read noise, gradient (S0); loop contrast; skin texture amplitude and correlation length (no literature value used); S3 depth, attenuation and blur growth.
- **Not implemented on purpose:** the Pries and Secomb 2005 in vivo viscosity law (`propagate.pries_secomb_viscosity` raises). The paper was not readable when this was written. Implement it from the paper and test against a value it prints.
- Literature numbers still to confirm in the originals: Fagrell velocities, the Aguiar healthy widths used in S7 (`sweeps.HEALTHY`), EULAR citation details.
- S6 is an amplification argument for an idealized straight tube, not a blood-flow prediction. Nothing here is diagnostic.

## Legacy first pass: E1 to E6 (results/*.csv, fig0 to fig6)

| Fig | Question | Headline (full run) |
| --- | --- | --- |
| 1 | Width error vs blur sigma | 12 um vessel: +0.3% at sigma 2 um, +3% at 3, +12% at 4, +39% at 6. 8 um vessel: +18% already at 3 um. FWHM beats naive Otsu. |
| 2 | Shot noise | Small effect (SD about 1% or less even at 200 photons/px) because the ROI averages ~80 rows |
| 3 | Lighting gradient | Uncorrected, width bias grows to +10% at a 30% gradient and +19% at 50%. Flat-fielding holds it under 1.5%. |
| 4 | Pixel size | Fine up to ~2 um/px; 8 um vessel degrades by 4-5% at 3-4 um/px |
| 5 | Loop counting | Exact (7 of 7) down to 10% contrast at every blur tested; at 5% contrast and sigma 6-8 um noise creates false loops (7 true, ~8 and ~24 counted) |
| 6 | Propagation | Width error → Poiseuille: R ~ r^-4, wall shear ~ r^-3. A 12 um vessel at sigma 4 um (+12% width) gives about -36% modeled resistance |

Figure 0 shows example synthetic images at three blur/noise settings.

## What this does and does not show

- Results are for the **simplified** forward model. Blur is Gaussian, tissue scattering and depth are not modeled, and noise is white. Do not quote numbers as properties of the Celestron.
- The width estimators assume two vertical limbs in a known ROI. Real images need orientation and ROI selection first.
- E6 is an amplification argument for an idealized straight tube, not a blood-flow prediction.
- Nothing here is diagnostic.

## Notes on the legacy first pass

- Results are for the simplified forward model. Do not quote numbers as properties of the Celestron.
- Its width estimators assume two vertical limbs in a known ROI.
- The renderer was changed after the first pass (union of discs instead of nearest-point), so rerunning `run_experiments.py` may shift numbers slightly.

## Beyond S1 to S7
- `python run_sims.py s8` (non-Gaussian PSF misspecification) and `sens` (tornado and Monte Carlo over the S0 measurement uncertainty).
- `python run_sims.py all` writes `REPORT.md` with the gate table (`capsim/gates.py`, thresholds fixed in `ANALYSIS_PLAN.md` before data arrives). "MODEL NOT VALIDATED" until the S5 gate passes.
- `python analyze_captures.py <folder> --apex auto` measures real captures with the same estimators.
- Estimators return NaN (not a wrong number) when limbs are unresolved; S5 compares the simulated bias with the uncertainty of the real mean bias, not the limits of agreement; S7 does not give the tool the true shift; pixel-size sweeps scale photons with pixel area.

## Training data for the model (dataset-matched, colour, labelled)
The Capillary Dataset (Nguyen and Jeong, KDD 2025; GitHub `urgonguyen/Capillarydataset` is code only, the 13.5 GB of images are on Hugging Face `HanaNguyen/Capillary-Dataset`) labels four morphologies, class ids `bushy 0, crossing 1, hairpin 2, tortuous 3`, as YOLO boxes. Its published baseline on its own validation split is mAP50 about 0.78 and mAP50-95 about 0.44.

```bash
# 1. real data (about 205 MB): Morphology_detection only
huggingface-cli download HanaNguyen/Capillary-Dataset --repo-type dataset --include "Morphology_detection/*" --local-dir data/capillary_dataset
# 2. profile it: class mix, boxes per image, box size, colour, sharpness, implied um/px
python profile_dataset.py data/capillary_dataset/Morphology_detection/images data/capillary_dataset/Morphology_detection/labels
# 3. fit the simulator's look to the real frames (reports held-out two-sample AUC before and after)
python fit_look.py data/capillary_dataset/Morphology_detection/images --base dataset --out configs/look_fitted.json
# 4. generate labelled synthetic frames (use --look celestron or --scope results/scope_params.json for your own scope)
python make_training_set.py --n 3000 --look configs/look_fitted.json --profile results/dataset_profile.json --out data/synth
# 5. check the gap on held-out REAL frames, then train
python run_sims.py s9 --quick        # needs data/real_images/ for the D1 gate
yolo detect train model=yolov8n.pt data=data/synth/data.yaml epochs=50          # pre-train on synthetic
yolo detect train model=runs/detect/train/weights/best.pt data=<real data.yaml>   # then fine-tune on real
```
Rules: validate only on REAL held-out images and report the real-only baseline beside the synthetic-pretrained number. Weights and Biases: `pip install wandb && wandb login && yolo settings wandb=True`. The synthetic shapes are parametric proxies; `domain_gap.py` (two-sample AUC and per-feature gaps) is how we find out how far off they are. Until step 3 has run on the full dataset the look is an educated default (checked only against 14 tiles cropped from the paper's figure, which is not a valid reference for sharpness or noise).
Quick check of the sim-to-real direction with their weights: `yolo detect val model=improved_yolov8_morphology/runs/detect/train_n/weights/best.pt data=data/synth/data.yaml split=test` (a model trained on real frames should find loops in good synthetic frames; a low score means the synthetic look is off).
