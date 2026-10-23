# Session 2026-10-21 (Sujay, 5 fingers x 6 captures)

Numbers only (no images are committed). Reproduce:

    python analyze_session.py <folder of the 60 PNGs> --out results/session_2026-10-21
    python validate_session_pipeline.py --sigma 1.9 --lumen 6 --total 22 --contrast 0.095 --illum 0.16
    python run_sims.py all --scope sessions/2026-10-21_sujay/scope_central.json --data <folder with real_images/> --out results/central

- `frames.csv`, `loops.csv`, `session_summary.json`: per-frame quality and per-loop widths, in PIXELS.
- `scope_central|low|high.json`: blur, contrast, lighting and noise MEASURED from the frames; um/px is an
  ANATOMICAL ESTIMATE (1.6, bounds 1.3 and 2.0) until the ruler photo is measured (S0).
- `validation_matched.json`: the pipeline on simulated frames that match this session (known truth).
- `measured_texture.json`: S4 density call with skin texture measured from these frames (0.0075 equivalent).
- `d1_gap.json`: synthetic vs real image gap, dataset look vs Celestron-matched look.
- `sims_central_summary.json`: full S1 to S9 run at the central scenario (S2/S4 density there used the old
  default texture 0.05; see measured_texture.json and the results doc for the corrected numbers).
