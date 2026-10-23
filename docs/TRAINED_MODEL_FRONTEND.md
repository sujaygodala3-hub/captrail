# CapTrail public frontend ↔ private trained model (research integration)

Frontend: sujaygodala3-hub/captrail (Vercel static site)  
Backend: aryansenthilkumar/CapTrail-Database (FastAPI, current research PR #2)

This feature branch adds a second **Trained capillary model** button to the existing microscope page. It deliberately leaves the existing Claude experimental visual-review function unchanged. The new pathway uses the real trained detector and returns actual source morphology candidates (bushy/crossing/hairpin/tortuous), not health or disease diagnoses.

## Two services, one browser origin

The Vercel frontend cannot host Python/PyTorch model inference in its static build. Deploy the CapTrail FastAPI research backend separately on secure persistent Python-compatible infrastructure. Configure the **server-only** Vercel environment variable `CAPTRAIL_ML_BACKEND_URL=https://your-private-backend.example` (origin only, no path). It is intentionally not in source code.

The same-origin Vercel function `api/ml-bridge.js` proxies only an explicit allowlist of operations: ready, auth-mode, login, logout, capabilities, analyze. It forwards only the `captrail_session` HttpOnly cookie to the backend; no tokens are stored in JavaScript/localStorage. The underlying CapTrail backend requires a valid authenticated session for uploads. The browser receives the same-origin cookie via the login response; strict SameSite rules and HTTPS must be used.

No configured backend = **503 / no upload**. Never replace this with a public shared backend token. Never allow the public website to proxy uploaded medical images anonymously to a globally authenticated service.

## Test after deploying both services

1. Open `scan.html` and verify existing USB microscope capture/upload still works, as does the existing optional Claude endpoint.
2. Select a captured image and enter a pseudonymous research participant ID; do not enter real names.
3. Confirm you have permission for private server storage.
4. Select **Trained capillary model**. If the backend has not been deployed, a descriptive error appears and no scan is uploaded.
5. Sign in with the *backend's* named user/password, or the private single-user research token if configured.
6. Retry. Confirm an actual `POST /api/analyses` to the separately hosted FastAPI service, a returned analysis, and rectangle overlays on the original image.
7. Verify that the new output states **unreviewed candidate detections**, not calibrated confidence, micrometer dimensions, healthy status or clinical diagnosis.
8. Verify authentication failure, server 503, batch identity safeguards, no cookies leaked cross-origin, no tokens in logs/storage, and no undesired data sent to the Claude API.
9. Verify Vercel request limit (~4 MB for this bridge) and hosted function timeout. For larger images/batches, use a proper authorized job/upload service.
10. Perform live browser integration/security review on the deployed environment before enabling broadly.

## Remaining blockers

- Separate HTTPS backend deployment, protected research user accounts, and secure environment-variable configuration.
- Live end-to-end browser acceptance and privacy/security approval.
- Validated anatomical accuracy and verified disease labels. This is a research integration, **not a customer-ready medical device**.

## Dataset grouping

The research dataset DOES contain source-level T2DM/control cohorts (published 126 T2DM people and 76 controls). The source tree reports 126 diabetic and 76 control *folders*, but it is not yet established that each folder maps one-to-one to a participant, especially for mixed-format healthy folders. A safe data pipeline preserves `source_cohort`, `source_group`, `subject_id_verified=false` pending author/acquisition verification and uses the most conservative defensible grouping for exploratory splits. Do not describe source cohort as an independently verified image-level diagnosis and do not train clinical T2DM prediction for customer use without independent clinical evaluation.
