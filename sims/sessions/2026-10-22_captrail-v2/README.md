# How datasets/captrail-v2 was built (Oct 22, 2026)

## Inputs

- `captrail images.zip` (team collection): folders `person1`..`person4` x `-unenhanced`/`-enhanced`, `ruler/`, `white/`,
  plus four nested zips that are byte-identical copies of the folders (skipped).
- `sujay.zip` (first session, Oct 21). Its two byte-identical copies were not used.
- The earlier `unenhanced.zip` / `enhanced.zip` / `ruler.zip` / `white.zip` are byte-identical to the collection's
  `person1-*`, `ruler/` and `white/` folders, so the collection replaces them.

## Who is who (decided by the team, Oct 22)

- **Each folder is one person.** File prefixes are shifted inside the new folders (the `person2` folder holds
  `person1-...` files, `person3` holds `person2-...`, `person4` holds `person3-...`), so files are renamed
  `f<folder>p<prefix>-...` by `prepare_inputs.py`.
- **Every person is a different person**, including the partial `person2` inside the `person1` folder
  (left index and middle only). That gives 6 subjects: SUBJ-01..06 in the dataset (real names stay in the
  git-ignored `_private/`).

## Calibration

- `--pitch-um 100`: the Captrail calibration card has 0.1 mm small ticks. The three card photos give
  6.102 um/px; the build guide's own value is 6.094 (0.13% apart).
- That is the card's zoom. The nail captures were taken at about 4x more zoom (at 6.1 um/px loops would be
  ~190 um wide and ~1.5 per mm), so every `*_um_est` uses the anatomical estimate 1.6 um/px (1.3 to 2.0)
  until the card is photographed at the nail zoom.

## Commands

```
python prepare_inputs.py "captrail images.zip" --out inputs_v2 --also sujay.zip   # 213 files kept
python batch_run.py inputs_v2 --out results/batch_v2                             # pitch 100 is the default
python export_dataset.py results/batch_v2 --out datasets/captrail-v2
python export_dataset.py --check datasets/captrail-v2
```

## Where it ran

- Analysis (sessions, calibration, scopes): the Linux shell next to the team folder (numpy 2.2.6).
- All 575 quick-mode sim steps: one Linux machine (numpy 2.5.3), from byte-identical scope files, merged by
  cache key. The dataset export was reproduced byte for byte on both machines (manifest digest
  `966d41c6eba5e73c`). Rerunning everything on one machine with the commands above gives the same analysis;
  quick-mode S1 and S7 values can move slightly between platforms (see the dataset card).
