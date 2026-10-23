# Schema

Every CSV column: type, unit and meaning. `*_um_est` columns use the ESTIMATED scale in `tables/sessions.csv`.

## `tables/sessions.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `session_id` | str |  | Capture session, S01, S02, ... |
| `subjects` | str |  | Pseudonymized subjects in the session, ';'-separated |
| `views` | str |  | Views present: raw (unprocessed camera frame), enhanced (app-processed), grey |
| `n_inputs` | int |  | Distinct input zips (identical copies counted once) |
| `n_duplicate_inputs` | int |  | Byte-identical copies of an input that were ignored |
| `input_sha256` | str |  | sha256 of each distinct input zip, ';'-separated |
| `n_frames` | int |  | Image files in the session |
| `n_measured_frames` | int |  | Measured frames: one per capture per input zip (raw preferred) |
| `n_usable_frames` | int |  | Measured frames passing the usability rule |
| `n_loops_resolved` | int |  | Loops with both limbs measured, usable frames |
| `um_per_px_est` | float | um/px | Scale used for every *_um_est column (anatomical estimate) |
| `um_per_px_low` | float | um/px | Lower bound of the scale estimate |
| `um_per_px_high` | float | um/px | Upper bound of the scale estimate |
| `scale_status` | str |  | How the scale was obtained; 'anatomical_estimate' until a ruler is shot at nail zoom |

## `tables/frames.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `frame_id` | str |  | Stable id: session/subject-hand+digit-capture-view |
| `session_id` | str |  | Session |
| `subject_id` | str |  | Pseudonymized subject |
| `site_id` | str |  | Subject + hand + digit; the unit for train/val/test splits |
| `hand` | str |  | L, R or empty when not recorded |
| `digit` | str |  | thumb, pointer, middle, ring, pinky |
| `capture` | int |  | Capture number at this site (1 = first) |
| `view` | str |  | From the file label: raw, enhanced or grey |
| `view_by_pixels` | str |  | What the pixel classifier says: raw, enhanced, grey |
| `view_label_matches_pixels` | bool |  | False when the file label and the pixels disagree |
| `measured` | bool |  | This frame is the one measured for its capture within its input zip (raw preferred; an enhanced-only zip measures its enhanced frames) |
| `usable` | bool |  | measured AND >= 3 loops with both limbs AND glare < 30% |
| `unusable_reason` | str |  | Why not usable: not_measured_view, too_few_resolved_loops, glare (';'-separated) |
| `split` | str |  | train, val or test (by site) |
| `n_apices` | int | count | Loop apexes detected |
| `n_loops_measured` | int | count | Loops measured (away from edges and glare) |
| `n_loops_resolved` | int | count | Loops with both limbs resolved |
| `glare_pct` | float | % | Frame area saturated by LED glare |
| `clipped_pct` | float | % | Pixels at >= 250 in any channel |
| `sharpness` | float | DN^2 | Variance of the Laplacian outside glare |
| `noise_dn` | float | DN | Robust high-pass noise of the green channel |
| `mean_r` | float | DN | Mean red outside glare |
| `mean_g` | float | DN | Mean green outside glare |
| `mean_b` | float | DN | Mean blue outside glare |
| `illum_strength` | float | fraction | Peak-to-peak lighting gradient across the frame |
| `blur_sigma_px` | float | px | Gaussian blur from the shared-sigma limb fit (needs >= 4 limbs) |
| `lumen_width_fit_px` | float | px | Median lumen width from that fit, blur removed |
| `lumen_contrast` | float | fraction | Median fractional darkening of the limbs |
| `median_gap_px` | float | px | Median spacing between neighbouring apexes |
| `median_limb_left_px` | float | px | Median left (arterial-side) limb FWHM |
| `median_limb_right_px` | float | px | Median right (venous-side) limb FWHM |
| `median_apical_px` | float | px | Median apical FWHM |
| `median_total_px` | float | px | Median total loop width, outer edge to outer edge |
| `median_apical_um_est` | float | um | median_apical_px x um_per_px_est (ESTIMATE) |
| `median_total_um_est` | float | um | median_total_px x um_per_px_est (ESTIMATE) |

## `tables/loops.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `loop_id` | str |  | frame_id + '/L' + index left to right |
| `frame_id` | str |  | Frame |
| `session_id` | str |  | Session |
| `site_id` | str |  | Site |
| `view` | str |  | raw, enhanced or grey |
| `frame_usable` | bool |  | The frame passed the usability rule |
| `split` | str |  | train, val or test |
| `apex_x_px` | float | px | Apex column in the 640 x 480 frame |
| `apex_y_px` | float | px | Apex row |
| `resolved` | bool |  | Both limbs measured |
| `limb_left_px` | float | px | Left limb FWHM |
| `limb_right_px` | float | px | Right limb FWHM |
| `apical_px` | float | px | Apical FWHM |
| `total_px` | float | px | Total loop width |
| `depth_frac` | float | fraction | Darkening of the limbs relative to background |
| `limb_left_um_est` | float | um | ESTIMATE at um_per_px_est |
| `limb_right_um_est` | float | um | ESTIMATE at um_per_px_est |
| `apical_um_est` | float | um | ESTIMATE at um_per_px_est |
| `total_um_est` | float | um | ESTIMATE at um_per_px_est |

## `tables/sites.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `site_id` | str |  | Subject + hand + digit |
| `session_id` | str |  | Session |
| `view` | str |  | View the measurements come from |
| `split` | str |  | train, val or test |
| `measured_frames` | int |  | Measured frames (one per capture) |
| `usable_frames` | int |  | Usable frames |
| `usable_pct` | float | % | usable / measured |
| `loops_detected` | int |  | Loops detected on measured frames |
| `loops_resolved` | int |  | Loops resolved on usable frames |
| `limb_px_median` | float | px | Median limb FWHM |
| `apical_px_median` | float | px | Median apical FWHM |
| `total_px_median` | float | px | Median total width |
| `blur_sigma_px` | float | px | Median blur |
| `lumen_contrast` | float | fraction | Median limb contrast |
| `illum_strength` | float | fraction | Median lighting gradient |
| `glare_pct` | float | % | Median glare |
| `sharpness` | float | DN^2 | Median sharpness |
| `gap_px` | float | px | Median apex spacing |
| `apical_cv_pct_between_captures` | float | % | CV of per-capture median apical width (real repeatability) |

## `tables/paired_views.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `capture_id` | str |  | Session/subject-hand+digit-capture, seen in both views |
| `session_id` | str |  | Session |
| `site_id` | str |  | Site |
| `split` | str |  | train, val or test |
| `paired_by` | str |  | pixels or file label |
| `raw_n_apices` | int | count | Apexes detected (raw view) |
| `raw_n_loops_resolved` | int | count | Loops resolved (raw view) |
| `raw_sharpness` | float | DN^2 | Sharpness (raw view) |
| `raw_noise_dn` | float | DN | Noise (raw view) |
| `raw_blur_sigma_px` | float | px | Blur (raw view) |
| `raw_median_apical_px` | float | px | Median apical FWHM (raw view) |
| `raw_glare_pct` | float | % | Glare (raw view) |
| `enhanced_n_apices` | int | count | Apexes detected (enhanced view) |
| `enhanced_n_loops_resolved` | int | count | Loops resolved (enhanced view) |
| `enhanced_sharpness` | float | DN^2 | Sharpness (enhanced view) |
| `enhanced_noise_dn` | float | DN | Noise (enhanced view) |
| `enhanced_blur_sigma_px` | float | px | Blur (enhanced view) |
| `enhanced_median_apical_px` | float | px | Median apical FWHM (enhanced view) |
| `enhanced_glare_pct` | float | % | Glare (enhanced view) |

## `tables/splits.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `site_id` | str |  | Site |
| `session_id` | str |  | Session |
| `split` | str |  | train, val or test |
| `n_frames` | int |  | Frames at this site (all views) |
| `rank_key` | str |  | First 12 hex of sha256(site_id); sites are dealt to splits in this order |

## `calibration/ruler_photos.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `photo` | str |  | Ruler photo number |
| `um_per_px` | float | um/px | Scale at the RULER's zoom (one fine tick = pitch_um) |
| `se_um_per_px` | float | um/px | Standard error of that photo's fit |
| `pitch_px` | float | px | Fine-tick pitch from the tick-band autocorrelation |
| `tilt_deg` | float | deg | Best rotation |
| `long_tick_px` | float | px | Long-tick spacing (independent check, 10 pitches expected) |
| `long_over_10_fine` | float | ratio | long_tick_px / (10 x pitch_px); 1.0 = consistent |
| `um_per_px_old_peakpicking` | float | um/px | Superseded estimator (split ticks), kept for the record |
| `tick_blur_sigma_px` | float | px | Blur at the ruler's zoom, from folded tick profile |
| `tick_width_px` | float | px | Printed tick width |

## `calibration/noise_spectrum.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `band_px` | str | px | Difference-of-Gaussian band |
| `measured_dn` | float | DN | Blank-card frame-to-frame noise in this band (one frame) |
| `white_model_dn` | float | DN | What a white-noise model gives |
| `fitted_dn` | float | DN | White + correlated model used in the sims |

## `sims/scenarios.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `scenario_id` | str |  | session-view-scenario |
| `session_id` | str |  | Session whose frames set the parameters |
| `view` | str |  | View the parameters were measured on |
| `scenario` | str |  | anat_central / anat_low / anat_high (scale), white_noise_only (ablation), site_* (one site), ruler_zoom |
| `site_id` | str |  | Site for site_* scenarios, else empty |
| `um_per_px` | float | um/px | Pixel size simulated |
| `sigma_um` | float | um | Blur simulated |
| `contrast` | float | fraction | Lumen contrast |
| `illum_strength` | float | fraction | Lighting gradient |
| `photons` | float |  | Photons per pixel at full scale (white part of the noise) |
| `corr_noise` | str |  | JSON [[scale_px, std], ...] correlated noise, empty = white only |
| `texture_amp` | float | fraction | Skin texture amplitude used in S2/S4 |
| `scale_status` | str |  | anatomical_estimate or ruler (ruler_zoom only) |

## `sims/metrics.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `scenario_id` | str |  | Scenario |
| `mode` | str |  | quick (few replicates) or full |
| `s1_limb_mean_abs_bias_pct` | float | % | Mean |bias| of limb FWHM over the S1 truth grid at this blur and pixel |
| `s1_apical_mean_abs_bias_pct` | float | % | Same for apical width |
| `s1_total_mean_abs_bias_pct` | float | % | Same for total width |
| `s1_limb_worst_bias_pct` | float | % | Signed bias of the worst limb case |
| `s1_apical_worst_bias_pct` | float | % | Signed bias of the worst apical case |
| `s1_total_worst_bias_pct` | float | % | Signed bias of the worst total case |
| `s1_min_limb_under_10pct_um` | float | um | Smallest true limb measured within 10% |
| `s4_giant_error` | float | fraction | P(45 um read as giant or 55 um read as not), averaged |
| `s4_max_sigma_um_for_5pct` | float | um | Largest blur keeping that error <= 5% |
| `s4_p7_called_reduced` | float | fraction | P(true 7/mm called < 7/mm) |
| `s4_p5_called_normal` | float | fraction | P(true 5/mm called >= 7/mm) |
| `s4_density_est_9` | float | loops/mm | Mean estimate for a true 9/mm |
| `s7_icc` | float |  | Simulated repeat-capture ICC(2,1) |
| `s7_cv_pct` | float | % | Simulated repeat-capture CV of apical width |
| `platform_unstable` | str |  | Metrics that changed between two platforms (';'-separated); empty = stable or not checked |

## `sims/s1_width_bias.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `scenario_id` | str |  | Scenario |
| `kind` | str |  | limb, apical or total |
| `true_um` | float | um | True width |
| `sigma_um` | float | um | Blur |
| `pixel_um` | float | um | Pixel |
| `estimator` | str |  | fwhm (half-depth) or fixed threshold |
| `mean_um` | float | um | Mean measured |
| `sd_um` | float | um | SD measured |
| `bias_pct` | float | % | Mean bias |
| `n_ok` | int |  | Replicates measured |
| `bias_lo95` | float | % | 95% CI low |
| `bias_hi95` | float | % | 95% CI high |

## `sims/s2_density.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `scenario_id` | str |  | Scenario |
| `field_mm` | float | mm | Field width |
| `sigma_um` | float | um | Blur |
| `contrast` | float | fraction | Contrast |
| `true_density` | float | loops/mm | Truth |
| `rep` | int |  | Replicate |
| `est_density` | float | loops/mm | Estimate |
| `err_pct` | float | % | Error |

## `sims/s2_texture.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `scenario_id` | str |  | Scenario |
| `texture` | float | fraction | Texture amplitude |
| `true_density` | float | loops/mm | Truth |
| `rep` | int |  | Replicate |
| `est_density` | float | loops/mm | Estimate |

## `sims/s4_misclassification.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `scenario_id` | str |  | Scenario |
| `sigma_um` | float | um | Blur |
| `p_45_called_giant` | float | fraction | P(45 um read >= 50) |
| `p_55_called_not_giant` | float | fraction | P(55 um read < 50) |
| `p45_lo95` | float | fraction | CI |
| `p45_hi95` | float | fraction | CI |
| `p55_lo95` | float | fraction | CI |
| `p55_hi95` | float | fraction | CI |
| `n_each` | int |  | Trials |
| `err_rate` | float | fraction | Mean of the two |
| `mean_45` | float | um | Mean read for 45 |
| `mean_55` | float | um | Mean read for 55 |

## `sims/s6_propagation.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `scenario_id` | str |  | Scenario |
| `true_width_um` | float | um | True width |
| `assumption` | str |  | resistance, flow, velocity, pressure, ... |
| `exponent` | float |  | Power of radius |
| `mean_pct` | float | % | Mean error after propagation |
| `sd_pct` | float | % | SD |
| `p5_pct` | float | % | 5th pct |
| `median_pct` | float | % | Median |
| `p95_pct` | float | % | 95th pct |

## `sims/s7_repeatability.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `scenario_id` | str |  | Scenario |
| `icc` | float |  | ICC(2,1) |
| `ci_lo` | float |  | Bootstrap 95% low |
| `ci_hi` | float |  | Bootstrap 95% high |
| `icc_oracle_position` | float |  | ICC if the tool knew the true position |
| `cv_pct` | float | % | Repeat CV |
| `n_subjects` | int |  | Simulated subjects |

## `qa/reproducibility.csv`

| column | type | unit | meaning |
|---|---|---|---|
| `input_a` | str |  | Input zip |
| `input_b` | str |  | Byte-identical input zip |
| `output_files` | int |  | Outputs compared |
| `byte_identical` | int |  | Identical bytes |
| `float_equal` | int |  | Equal to 1e-9 |
| `real_differences` | str |  | Anything else |

## `llm/records.jsonl`

One JSON object per frame: `id`, `split`, `session_id`, `site_id`, `view`, `usable`, `unusable_reason`,
`metrics` (frame numbers), `loops` (per-loop numbers), `text` (the same facts as one paragraph).

## `llm/sft_{train,val,test}.jsonl`

Chat-format supervised examples: `messages` (system, user, assistant), `task`, `split`, `source` (the frame,
loop, scenario or capture the answer was computed from). Tasks: `frame_qc`, `loop_measurement`, `simulation_lookup`, `raw_vs_enhanced`.
Answers are generated by fixed templates from the pipeline's own numbers and rules.
