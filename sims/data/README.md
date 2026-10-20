# data/: where real captures go

Nothing here is committed except this file and empty folders (see `.gitignore`). **Never put
names, faces or identifiable metadata in filenames. Use participant codes (P01, P02).** Images stay
on the team laptop.

Save from the scan page with **Save**: it writes `<name>_raw.png` (untouched) and a processed view.
The code uses only the `_raw` files.

```
data/
  s0/                     real calibration, do this first
    slide/                10 captures of the 10 um slide at the fixed finger-rest height, two sessions
    flat/<level>/         blank white card, >= 2 frames per exposure level, >= 3 levels (low, mid, high)
                          LOCK exposure and white balance first, then change only the light or card distance
    gradient/             blank card at the working exposure (lighting gradient), 1 to 3 images
    edge/                 optional: razor edge (black card on white), for PSF sigma if tick width is unknown
    contrast.csv          optional: condition,contrast        (condition = no_oil | oil)
  s3/oil_pairs.csv        condition,contrast,sigma_um          one row per capture, no_oil and oil
  s5/slide_profile.csv    one column: a real 1D profile across the slide ticks (from the scan page crop)
  s5/manual_widths.csv    capillary_id,rater1_um,rater2_um,tool_um    two people measure blind, plus the tool
  s7/repeat_widths.csv    capture_id,width_um                 10 re-seated captures of the same capillary
```

## Order of work

1. `python run_s0.py` writes `results/scope_params.json` and tells you what is still ASSUMED.
2. `python run_sims.py all --scope results/scope_params.json` writes `results/measured/` once everything is measured.
3. S5 decides whether S1 to S4 may be reported. If `s5.sim_vs_real.passes` is false, fix the model first.

## Before S0 will work

- `configs/s0_config.json`: set `tick_width_um` to your slide's real dark-tick width. Without it, PSF sigma
  cannot be fitted from the ticks (use `data/s0/edge/` instead).
- Do the photon-transfer frames with fixed exposure. If the camera applies gamma or auto exposure the
  code reports `not linear` and leaves gain and read noise ASSUMED rather than guessing.
- S3 cannot identify depth, attenuation and blur growth separately; `configs/s3_config.json` holds the
  depth assumption. State it in the write-up.
