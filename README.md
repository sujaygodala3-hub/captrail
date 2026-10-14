# Captrail

A low-cost nailfold microscope that uses AI to flag early signs of autoimmune and vascular disease for a doctor to review.

**Live site:** https://captrail-seven.vercel.app

> Research prototype for screening and referral. Not a diagnostic tool, and it does not replace a doctor.

## Pages

| File | What it is |
| --- | --- |
| `index.html` | Home page: purpose, how it works, hardware → software, results, team |
| `build.html` | Parts list and step-by-step assembly guide |
| `scan.html` | Connect the Celestron USB microscope: live view, capture, µm calibration, measurement, AI analysis |
| `site.css` / `site.js` | Shared styles, header and footer (add header pages in the `PAGES` list in `site.js`) |

## Editing

- **Results:** fill in the `JOURNEY` list near the bottom of `index.html`.
- **AI model:** set `ANALYZE_ENDPOINT` in `scan.html` once the model is deployed.

## Run locally

It's a static site, so any static server works:

```bash
npx serve .
```

The microscope page needs `localhost` or `https://` for camera access.

## Deploy

```bash
npx vercel deploy --prod
```

## Team

Sujay Godala · Smyan Reddy · Aryan Senthilkumar · Ansh Mishra

Finger rest design: [Attune (dlx-designlab)](https://github.com/dlx-designlab/Attune).
