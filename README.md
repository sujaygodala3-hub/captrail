# Captrail

**A $32 nailfold microscope that images the tiny blood vessels at the base of your fingernail, measures them in micrometers, and flags unusual loops for a doctor to review.**

[![Live site](https://img.shields.io/badge/live%20site-captrail--app.vercel.app-8E1B24?style=for-the-badge)](https://captrail-app.vercel.app)
[![Microscope app](https://img.shields.io/badge/try%20it-microscope%20app-17110F?style=for-the-badge)](https://captrail-app.vercel.app/scan.html)
[![Build guide](https://img.shields.io/badge/build%20it-guide-B8434E?style=for-the-badge)](https://captrail-app.vercel.app/build.html)

> Research prototype for screening and referral. Not a diagnostic tool, and it does not replace a doctor.

## At a glance

| | |
| --- | --- |
| Build cost | **$32** (USB microscope, two 3D-printed parts, mineral oil, swabs) |
| Scale at nail zoom | **3.745 µm per pixel** (640 × 480 frame ≈ 2.4 × 1.8 mm) |
| Calibration | Coordinate cross on the calibration card: 0.1 mm per small tick, 1 mm per long line |
| Software | Runs in the browser. Nothing to install |
| AI | Trained loop-shape model (private backend) plus an optional Claude review (`claude-sonnet-5-5`) |
| Who decides | A clinician. Every result is a flag, never a diagnosis |

## What is in this repo

| Path | What it is |
| --- | --- |
| `index.html` | Landing page: purpose, how it works, hardware to software, results, team, open source |
| `build.html` | Parts list and step-by-step assembly guide |
| `scan.html` | Microscope app: live view, Best of 8, µm calibration, click-to-measure, AI analysis, voice commands |
| `site.css` / `site.js` | Shared styles, header and footer (add header pages in the `PAGES` list in `site.js`) |
| `api/` | Vercel functions: `analyze.js` (Claude review), `voice.js` (voice commands), `ml-bridge.js` (private trained model) |
| `captrail_print_files/` | Finger cradle and base board, `.stl` and `.gcode` |
| `sims/` | Microscope physics simulations, calibration code and tests |
| `tests/` | Automated tests for the model bridge |

## Calibration

Image the calibration card at the **same zoom and height you use on the nail**, click two marks a known distance apart on the Microscope page, and enter the distance in µm.

```
µm per pixel = known distance in µm / distance in pixels
```

Our setup reads **3.745 µm per pixel** at nail zoom. Recalibrate whenever zoom, height or focus changes.

## Environment variables

Set these in Vercel → Project → Settings → Environment Variables, then redeploy. For local testing, copy `.env.example` to `.env.local` (ignored by git).

| Name | Used by | Needed for |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | `api/analyze.js`, `api/voice.js` | AI review and voice answers |
| `CAPTRAIL_ML_BACKEND_URL` | `api/ml-bridge.js` | Trained model (optional, https only) |

## Run locally

```bash
npm install
npx vercel dev      # serves the site and the api/ functions, reads .env.local
```

For the static pages only, `npx serve .` works. The microscope page needs `localhost` or `https://` for camera access.

## Deploy

```bash
npx vercel deploy --prod
```

## Team

Four students from four Georgia high schools:
Sujay Godala (Denmark High School) · Smyan Reddy (Lambert High School) · Aryan Senthilkumar (Northview High School) · Ansh Mishra (Alliance Academy for Innovation)

## Credits

Finger rest design adapted from [Attune (dlx-designlab)](https://github.com/dlx-designlab/Attune). Built with NumPy, SciPy, OpenCV, scikit-learn, pandas, PyTorch, FastAPI and the Anthropic SDK.
