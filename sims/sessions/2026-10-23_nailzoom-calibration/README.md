# 2026-10-23: the calibration card at nail zoom

**Result: 1.8749 um per pixel at the nail zoom.** This replaces the anatomical estimate (1.6 um/px, bounds
1.3 to 2.0) that datasets v1 and v2 used. Every um value moves up by 1.8749 / 1.6 = 17.2%.

## Input

`calibration (6).zip` (sha256 `87dbc11727a6...`): 3 photos of the Captrail calibration card's crosshair and
5 photos of a plain white card, all at the zoom and height of the nail captures. The photos are in
`tests/fixtures/card_nailzoom/` (no people in them).

## Measurement

| photo | um/px | horizontal arm px/tick | vertical arm px/tick | vertical / horizontal | card blur sigma px |
| --- | --- | --- | --- | --- | --- |
| ruler-1 | 1.87487 | 53.335 | 53.346 | 1.00021 | 1.02 |
| ruler-2 | 1.87495 | 53.338 | 53.332 | 0.99989 | 1.05 |
| ruler-3 | 1.87498 | 53.333 | 53.336 | 1.00006 | 0.98 |
| **mean** | **1.87493 (SD 0.00006)** | | | **1.00005** | **1.02** |

- Method: `calibration.tick_ladder`. Each arm of the cross is levelled, every 0.1 mm tick is located to
  sub-pixel accuracy (10 to 11 ticks on the horizontal arm, 7 on the vertical), and a straight line through
  the tick positions gives the pitch. The rms residual is 0.04 to 0.10 px, which is 0.1 to 0.2% of the pitch.
  On synthetic cards with a known pitch (30 to 53 px, tilt up to 2.6 degrees) it is within 0.013%.
- Independent checks:
    - The old peak-picking estimator agrees at this zoom: 1.87488 to 1.87491 um/px.
    - The two arms agree, so the pixels are square.
- Not checked: the card's own printing accuracy against a certified stage micrometer. The low-zoom card
  photo agreed with the build guide's independent value to 0.13%.
- The earlier card photo (`ruler`, 6.102 um/px) was taken at 3.25x lower magnification. It is kept as a
  record and as the `ruler_zoom` what-if scenario only.

## What the nail images look like at this scale (234 measured frames, 9 inputs, 6 subjects)

| quantity | px | um | healthy reference |
| --- | --- | --- | --- |
| median apical width (FWHM) | 7.98 | 15.0 | apical lumen of roughly 10 to 20 um |
| median total loop width | 30.9 | 58.0 | |
| median spacing between detected apexes | 105 | 197 | 7 or more loops per mm, i.e. 143 um or less |
| optical blur, capillary limbs (shared-sigma fit) | 2.33 | 4.4 | |
| optical blur, card ticks | 1.02 | 1.9 | |

- **Widths fit healthy anatomy** at the measured scale.
- **Apex spacing does not.** About 5 detected loops per mm is below the 7 per mm normal cut-off, in healthy
  volunteers. The likeliest cause is the detector missing faint or out-of-focus loops; the S2 density sims
  show undercounting at this blur. It could also mean some captures were not at exactly the card's zoom.
  So **do not read capillary density from this data**. Re-shoot the card at the start and end of each
  session to rule out a zoom change.
- **Tissue adds blur.** The capillaries are blurrier than the card (2.33 vs 1.02 px). The skin scatters light
  on top of the optics, and the sims model this as the measured limb blur.

## Noise

The white card at nail zoom shows strong vignetting: the frame centre is 1.58x as bright as the edge. Its
per-pixel noise is 1.03 DN, and brightness drifts by 8.4 DN across the 5 frames. This card is used for the
noise model because it was shot at the same zoom as the nails. The earlier `white` zip is kept and reported.

## Pipeline changes

- Card photos are recognised by pixels at any zoom.
- A zip holding both card and white-card frames is processed as a calibration input.
- A card in the nail-zoom range (0.8 to 3.0 um/px) becomes the measured scale for every um value, skin
  texture estimate and sim scope. The `measured` scenario replaces `anat_low` / `anat_central` / `anat_high`.
- See `sims/README.md` and dataset `captrail-v3`.
