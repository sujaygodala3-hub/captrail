# CapTrail V6 implementation and release gates

The production UX handoff was read in full (all eight files, README_FIRST and ASTRA_MASTER_PROMPT first). It contains a comprehensive acceptance checklist, not executable test programs. This change implements tests for the new workflow and preserves the existing simulator and backend suites.

## Delivered behavior

- Warm ivory/berry research workstation with dark optical stage, responsive upload queue, keyboard zoom/pan, morphology summaries, reduced-motion support, named-account modal and saved observations.
- JPEG/PNG magic-byte, encoded-size, dimension and aggregate-preview budgets before inference; no silent resize. Local file selection sends zero image bytes. Explicit storage consent and an explicit analyze action are required; training consent remains false.
- Real trained YOLO detector through a bounded same-origin bridge. Original image pixels and SVG geometry share one viewBox. Returned line vertices are animated without invented splines; replay does not re-run inference. Masks, detections and human annotations are distinct layers. Neural semantic fragments are not assigned to individual detector boxes or counted as capillaries.
- Four registered neural tracing comparisons, including V5 negative-regularized weights, retain checkpoint/image provenance and experimental status. Default detector and default ridge tracing are unchanged.
- Candidate inspection, morphology correction, exclusion, provisional notes and explicit manual centerline/boundary/apex annotation. Review revision conflicts fail rather than overwrite. Original predictions remain in the record.
- Authenticated session reopening, review history, JSON record export, standalone HTML report and SVG overlay export. Physical units are withheld unless a saved verified reference exists; detector box dimensions remain pixels, never vessel diameter.
- Microscope device selection/UVC controls, 40-frame stack, best-of-eight, enhancement, operator calibration and downloads remain in the capture studio. The former random capillary drawing was removed. Optional Anthropic commentary requires separate consent and no longer produces anatomical geometry or physical measurements.
- CSRF request markers, origin checks, host-only HttpOnly/SameSite cookies, streamed request limits and route allowlists. Login/logout and server session revocation are tested. No production credential is in source, browser storage or screenshots.

## Validation evidence

- Frontend: 33 passing Node contract/security tests; JavaScript syntax checks pass; production npm audit reports zero vulnerabilities at the recorded lockfile.
- Backend local selected suite: 98 passed, two skipped. The earlier full restore attempt failed 14 artifact/inventory tests because the private connector could not restore large binary artifacts; this is not reported as a passing local full suite. Backend GitHub CI has the complete artifact checkout and is the authority for those tests.
- Existing simulator suite: 84 passed on the initial checked-out revision. Newer main simulator input-preparation additions were preserved and must be included in the final CI result.
- Actual source detector + V5 neural model through the bridge: login, upload, image retrieval, review save/reopen, stale revision 409 and logout revocation passed. Public source image SHA `3468326aa909ea15d18bc003b2496a54976cb99549821274fcb36d1a6e8cf76f`; 12 detector candidates; 93 ridge fragments; 106 V5 neural fragments. Three upload-to-saved times were 789.6, 702.9 and 686.5 ms. These are small-sample CPU timings, not production P95 or accuracy evidence.
- GitHub container workflow at backend `575a13c880665dde936d75fd43a1221ff24d534e` passed Docker build, real-photo processing and the strict HIGH/CRITICAL vulnerability gate. Later changes through `bd22033` affect browser QA configuration only.
- Browser CI executes Chromium 1440×900, Firefox 1024×768 and WebKit 390×844/DPR3 using a real Python service and public source photograph. It covers consent, model inference, geometry mapping, layers, neural tracing, review/persistence, exports, reduced motion, responsive overflow and accessibility. Its latest result must be checked; do not infer success from test creation.
- The interactive managed browser rendered the new UI, then its file-picker and subsequent control calls timed out. Interactive sign-in/upload acceptance is not claimed from that session.

## Private preview

The temporary internal preview runs one Node gateway with a loopback-only Python backend. Named accounts, revocable sessions and AES-GCM storage are enabled. Credentials live in an ignored, mode-0600 file and are delivered separately. The managed route is HTTP inside the preview environment; it is not a persistent HTTPS customer deployment.

A lossless local float32 serialization is needed for this managed CPU environment. All 355 tensors were checked for numerical equality to the registered source checkpoint. Its local serialization SHA is `5f3d3e426bc480d0cc2b0e50b2ef49480a0973f4fa0c83d7f19f3e58d765f584`. This is not a newly trained model or an accuracy improvement. Production model registry and default checkpoint are unchanged.

## Release blockers and remaining scope

1. Persistent, access-controlled HTTPS Python hosting, encrypted backup/restore drill in that deployment, secure-cookie transport verification, account provisioning and a complete user-accepted private-preview walkthrough. No public deployment or merge is authorized by this change.
2. Independent expert masks/centerlines and adjudicated loop identities; untouched patient/device test collections; validated unfamiliar-image rejection; calibrated physical references and repeatability. No near-100% accuracy or anatomical completeness is established.
3. Source-folder identities and repeated captures remain unresolved. The registry records 8,368 classification files (4,956 diabetic-cohort and 3,412 healthy-cohort files), 126+76 folders, zero verified people, and a 1,673-file difference from the publication's 6,695 originals. Cohort folder names are not verified clinical labels.
4. Clinical diagnosis, healthy-status inference, disease-specific prediction, referral decisions and medical/customer approval remain unavailable.
5. Full clinical longitudinal comparison and verification of raw/enhanced spatial registration are not implemented in this frontend. Enhancements remain local previews and cannot be submitted as additional original acquisitions. The queue analyzes one selected original at a time.
6. Physical USB/UVC device behavior, GPU execution, broad slow-network/backend-restart/browser-refresh stress, large-session performance and all manual-review accessibility interactions still require final acceptance. Pixel-level browser regression baselines need expert-reviewed sign-off, not merely screenshot generation.
7. Server-side anatomical annotation/adjudication and calibration workflows already exist; this UI exposes provisional geometry correction and verified-reference display, not the complete multi-reader adjudication or calibration-approval administration.

PR #3 was already merged. The follow-up is draft frontend PR #6. Backend work continues in draft PR #2; neither development branch is merged by this task.
