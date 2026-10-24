# Real calibration photos at nail zoom (2026-10-23)

The Captrail calibration card (0.1 mm small ticks, 1 mm long lines) and a plain white card, photographed
with the Jiusion USB microscope at the zoom and height used for the nail captures. No people in these images.

- `ruler-1..3.png`: the card's crosshair. Expected scale 1.8749 um/px (53.33 px per 0.1 mm tick, both arms).
- `whitecard-1..5.png`: blank white card at the same zoom (camera noise spectrum, vignetting).

Source: the team's `calibration (6).zip`, sha256 87dbc11727a603941aa692b65fed138e654b6aa99323c9c00fcd87bbd9c30dad.
`tests/test_card.py::test_real_card_photos` checks the pipeline against them.
