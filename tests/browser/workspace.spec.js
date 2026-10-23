import {test,expect} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import fs from 'node:fs/promises';
const image=process.env.CAPTRAIL_SOURCE_IMAGE;
async function signIn(page){await page.locator('#sessionButton').click();await page.locator('#researchUser').fill('qa-researcher');await page.locator('#researchPassword').fill('test-only-public-source-fixture-2026');await page.locator('#submitSignIn').click();await expect(page.locator('#sessionButton')).toContainText('Sign out');}
test('real model, consent, precision, neural traces, review, persistence and exports',async({page},info)=>{
 test.skip(!image,'Requires the isolated backend QA fixture and public source image.');
 const errors=[];page.on('pageerror',e=>errors.push(e.message));let uploads=0;page.on('request',r=>{if(r.method()==='POST'&&r.url().includes('op=analyze'))uploads++;});
 await page.goto('/');await expect(page.locator('#modelStatus')).toContainText('Trained detector ready',{timeout:60000});
 await page.locator('#researchFiles').setInputFiles(image);await expect(page.locator('#vesselViewport')).toBeVisible();expect(uploads).toBe(0);
 await page.locator('#analyzeImage').click();await expect(page.locator('#researchMessage')).toContainText('Confirm permission');expect(uploads).toBe(0);
 await signIn(page);const session=(await page.context().cookies()).find(c=>c.name==='captrail_session');expect(session.httpOnly).toBe(true);expect(session.sameSite).toBe('Strict');expect(await page.evaluate(()=>Object.keys(localStorage).some(k=>/token|password|session/i.test(k)))).toBe(false);await page.locator('#participantId').fill('PUBLIC-SOURCE-QA');await page.locator('#storageConsent').check();
 const responsePromise=page.waitForResponse(r=>r.url().includes('op=analyze')&&r.request().method()==='POST');await page.locator('#analyzeImage').click();const response=await responsePromise;expect(response.status()).toBe(200);const a=(await response.json()).analyses[0];expect(a.provenance.inference_status).toBe('completed');expect(a.acquisition.consent_to_research_training).toBe(false);
 await expect(page.locator('#candidateCount')).toContainText('REGIONS');expect(await page.locator('.candidate-box').count()).toBe(a.detected_structures.capillary_instances.length);
 const first=a.detected_structures.capillary_instances[0];expect(first).toBeTruthy();await page.locator('#candidateSelect').selectOption(first.instance_id);await expect(page.locator('#candidateFacts')).toContainText('uncalibrated');
 const x=Number(await page.locator('.candidate-box').first().getAttribute('x'));expect(x).toBeCloseTo(first.bbox_xyxy[0],6);
 await page.locator('#zoomIn').click();await expect(page.locator('#zoomLevel')).toHaveText('150%');expect(Number(await page.locator('.candidate-box').first().getAttribute('x'))).toBe(x);await page.locator('#zoomFit').click();
 await page.locator('#showMasks').check();await page.locator('#replayTrace').click();expect(uploads).toBe(1);
 await page.locator('#traceModel').selectOption('v5_negative_regularized');await page.locator('#runTraceModel').click();await expect(page.locator('#researchMessage')).toContainText('Neural geometry loaded',{timeout:90000});expect(await page.locator('.trace-path').count()).toBeGreaterThan(0);
 await page.locator('#markCenterline').click();await page.locator('#researchStage').press('Enter');await page.locator('#researchStage').press('Shift+ArrowDown');await page.locator('#researchStage').press('Enter');await page.locator('#finishGeometry').click();
 await page.locator('#correctClass').selectOption('3');await page.locator('#reviewNotes').fill('Automated public-source QA. Not anatomical validation.');await page.locator('#saveReview').click();await expect(page.locator('#reviewState')).toContainText('Saved revision 1');await expect(page.locator('#reviewGeometryLayer path')).toHaveCount(1);
 await page.locator('#researchStage').scrollIntoViewIfNeeded();await page.screenshot({path:info.outputPath('actual-trained-model.png'),fullPage:true});
 for(const id of ['exportJson','exportReport','exportOverlay']){const d=page.waitForEvent('download');await page.locator('#'+id).click();const file=await d;await file.saveAs(info.outputPath(file.suggestedFilename()));expect((await fs.stat(info.outputPath(file.suggestedFilename()))).size).toBeGreaterThan(100);}
 const access=await new AxeBuilder({page}).include('.research-workspace').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();expect(access.violations.filter(v=>['critical','serious'].includes(v.impact))).toEqual([]);
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
 await page.reload();await expect(page.locator('#sessionButton')).toContainText('Sign out');await page.locator('.session-item').first().click();await expect(page.locator('#reviewState')).toContainText('Saved revision 1');await expect(page.locator('#reviewNotes')).toHaveValue('Automated public-source QA. Not anatomical validation.');
 await page.emulateMedia({reducedMotion:'reduce'});await page.locator('#replayTrace').click();expect(await page.locator('.trace-path').first().evaluate(e=>getComputedStyle(e).animationName)).toBe('none');
 await page.locator('#sessionButton').click();await expect(page.locator('#sessionButton')).toContainText('Sign in');await expect(page.locator('#sessionList')).toBeEmpty();expect(errors).toEqual([]);
});
test('malicious file rejected locally, responsive empty state and capture controls',async({page},info)=>{
 await page.goto('/');await page.locator('#researchFiles').setInputFiles({name:'image.png',mimeType:'image/png',buffer:Buffer.from('<svg><script>malicious()</script></svg>')});await expect(page.locator('#researchMessage')).toContainText('not a supported');
 await page.locator('#openCapture').click();await expect(page.locator('#btnStack')).toBeVisible();await expect(page.locator('#btnBurst')).toBeVisible();await expect(page.locator('#btnCal')).toBeVisible();await expect(page.locator('#externalReviewConsent')).not.toBeChecked();await page.locator('#captureStudio>summary').click();
 await page.screenshot({path:info.outputPath('empty-workspace.png'),fullPage:true});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});

test('failure states retain original, clear overlays, and recover without automatic uploads',async({page})=>{
 test.skip(!image,'Requires isolated real backend fixture.');
 await page.goto('/');await expect(page.locator('#modelStatus')).toContainText('Trained detector ready',{timeout:60000});await page.locator('#researchFiles').setInputFiles(image);await signIn(page);await page.locator('#participantId').fill('FAILURE-QA');await page.locator('#storageConsent').check();
 const original=await page.locator('#researchImage').getAttribute('href');
 // Status handling is a deterministic fault-injection test, not new model evidence.
 for(const code of [400,403,413,422,429,500,503]){
  await page.route('**/api/ml-bridge?op=analyze',route=>route.fulfill({status:code,contentType:'application/json',body:JSON.stringify({detail:'Injected failure '+code})}));
  await page.locator('#analyzeImage').click();await expect(page.locator('#researchMessage')).toContainText('Injected failure '+code);await expect(page.locator('#modelWorking')).toBeHidden();await expect(page.locator('#boxLayer')).toBeEmpty();expect(await page.locator('#researchImage').getAttribute('href')).toBe(original);await page.unroute('**/api/ml-bridge?op=analyze');
 }
 await page.context().setOffline(true);await expect(page.locator('#researchMessage')).toContainText('offline');expect(await page.locator('#researchImage').getAttribute('href')).toBe(original);await page.context().setOffline(false);
 await page.locator('#sessionButton').click();
});


test('capture architecture: simulated live camera, local voice, original transfer',async({page,browserName},info)=>{
 test.skip(browserName!=='chromium','Synthetic device regression runs in Chromium; physical USB and speech hardware remain manual checks.');
 let remoteVoice=0,uploads=0;page.on('request',r=>{if(r.url().includes('/api/voice'))remoteVoice++;if(r.url().includes('op=analyze'))uploads++;});
 // Explicit test-only speech recognizer. Production uses the browser Web Speech API.
 await page.addInitScript(()=>{window.SpeechRecognition=class{start(){setTimeout(()=>{this.onresult?.({results:[[{transcript:'capture'}]]});this.onend?.();},50);}stop(){this.onend?.();}};});
 const response=await page.goto('/');expect(response.headers()['permissions-policy']).toContain('microphone=(self)');
 await page.locator('#openCapture').click();await page.locator('#resSel').selectOption('640x480');await page.locator('#btnConnect').click();
 await expect(page.locator('#btnCapture')).toBeEnabled();expect(await page.locator('#video').evaluate(v=>v.srcObject.getVideoTracks()[0].readyState)).toBe('live');
 await page.locator('#micBtn').click();await expect(page.locator('#voiceReply')).toContainText('Captured.');await expect(page.locator('#btnLive')).toBeEnabled();expect(remoteVoice).toBe(0);
 const dimensions=await page.evaluate(async()=>{const f=await window.CapTrailCapture.original();const b=await createImageBitmap(f);return {width:b.width,height:b.height,type:f.type};});expect(dimensions).toEqual({width:640,height:480,type:'image/png'});
 await page.locator('#useCapture').click();await expect(page.locator('#vesselViewport')).toBeVisible();expect(uploads).toBe(0);
 await page.locator('#openCapture').click();await page.locator('#btnLive').click();await expect(page.locator('#btnCapture')).toBeEnabled();await expect(page.locator('#btnBurst')).toBeEnabled();await expect(page.locator('#btnStack')).toBeEnabled();
 await page.screenshot({path:info.outputPath('simulated-camera-regression.png'),fullPage:true});
});
