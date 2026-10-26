import {test} from 'node:test';
import assert from 'node:assert/strict';
import {calibrationProvenanceNotice} from '../research/calibration.js';

test('operator microscope report is shown alongside conflicting metadata',()=>{
 const message=calibrationProvenanceNotice({provenance_review_required:true,operator_reported_microscope_capture:true});
 assert.match(message,/Operator-reported microscope capture/);
 assert.match(message,/does not prove the image is synthetic/);
 assert.match(message,/review required before physical verification/);
});
test('metadata alone and older backend flags never assert synthetic origin',()=>{
 for(const proposal of [{provenance_review_required:true},{synthetic_evidence:true}]){
  assert.match(calibrationProvenanceNotice(proposal),/metadata alone does not establish image origin/);
 }
 assert.equal(calibrationProvenanceNotice({provenance_review_required:false}),null);
 assert.equal(calibrationProvenanceNotice(null),null);
});
