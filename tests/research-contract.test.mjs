import test from 'node:test';import assert from 'node:assert/strict';
import {imagePoint,box,polyline,traceGeometry,pathData,effectiveInstances,reviewInstances,pairKey,validateFile} from '../research/contract.js';
const a={image:{width:640,height:480},detected_structures:{capillary_instances:[{instance_id:'c001',class_id:2,class_name:'hairpin',bbox_xyxy:[20,30,80,100],confidence:.8}]}};
for(const dpr of [1,2,3])for(const rect of [{left:0,top:0,width:800,height:600},{left:5,top:40,width:1000,height:400},{left:10,top:20,width:358,height:500}])test(`shared SVG coordinate roundtrip DPR ${dpr}, viewport ${rect.width}x${rect.height}`,()=>{
 const view={x:90,y:45,w:320,h:240},scale=Math.min(rect.width/view.w,rect.height/view.h),p=[250,130];
 const screen=[rect.left+(rect.width-view.w*scale)/2+(p[0]-view.x)*scale,rect.top+(rect.height-view.h*scale)/2+(p[1]-view.y)*scale];
 const got=imagePoint(...screen,rect,view);assert.ok(Math.hypot(got[0]-p[0],got[1]-p[1])<1e-9); // CSS pixels: DPR must not multiply coordinates.
});
test('box geometry excludes NaN, flipped, negative and out-of-bounds',()=>{for(const b of [[0,0,NaN,1],[3,4,2,5],[-1,2,4,5],[0,0,641,200]])assert.equal(box(b,640,480),false);});
test('trace animation path reproduces every returned vertex without inventing a spline',()=>{assert.equal(pathData([[3,4],[10,25],[16,8]]),'M3 4 L10 25 L16 8');assert.equal(polyline([[2,3],[Infinity,3]],640,480),false);});
test('unknown coordinate spaces and invalid fragments are withheld',()=>{assert.throws(()=>traceGeometry({coordinate_space:'normalized',fragments:[]},640,480));assert.throws(()=>traceGeometry({coordinate_space:'original_image_pixels',fragments:[{centerline:[[1,2],[1000,2]]}]},640,480));});
test('review layer is distinct and source predictions immutable',()=>{const r=structuredClone(a);r.review={status:'provisional',instances:[{...reviewInstances(a)[0],class_ids:[3],excluded:false}]};assert.equal(effectiveInstances(r)[0].class_name,'tortuous');assert.equal(effectiveInstances(a)[0].class_name,'hairpin');r.review.status='rejected';assert.deepEqual(effectiveInstances(r),[]);});
test('raw/enhanced filenames retain one acquisition key',()=>{assert.equal(pairKey('person1-Lindex-2-n.png').key,pairKey('person1-Lindex-2-e.jpg').key);assert.equal(pairKey('person1-Lindex-2-e.jpg').representation,'enhanced');});
test('reject forged MIME bytes and oversize files before upload',async()=>{await assert.rejects(validateFile(new File(['<svg><script>x</script></svg>'],'image.png',{type:'image/png'})));await assert.rejects(validateFile(new File([new Uint8Array(4*1024*1024)],'large.jpg',{type:'image/jpeg'})));});
