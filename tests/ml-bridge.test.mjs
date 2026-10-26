import assert from 'node:assert/strict';
import test from 'node:test';
import {GET,POST} from '../api/ml-bridge.js';
const make=(u,method='GET',init={})=>new Request(u,{method,...init});
test('gateway refuses unconfigured backend, no upload',async()=>{
 const prior=process.env.CAPTRAIL_ML_BACKEND_URL;delete process.env.CAPTRAIL_ML_BACKEND_URL;
 try {const r=await GET(make('https://captrail.example/api/ml-bridge?op=ready'));assert.equal(r.status,503)}
 finally{if(prior!==undefined)process.env.CAPTRAIL_ML_BACKEND_URL=prior}
});
test('gateway prevents method hopping',async()=>{
 const r=await GET(make('https://captrail.example/api/ml-bridge?op=analyze'));
 assert.equal(r.status,405);
});
test('rejects cross-site state requests',async()=>{
 const r=await POST(make('https://captrail.example/api/ml-bridge?op=login','POST',{headers:{origin:'https://evil.example','content-type':'application/json'},body:'{}'}));
 assert.equal(r.status,403);
});
test('ready is proxied to configured HTTPS origin only',async()=>{
 const original=globalThis.fetch, prior=process.env.CAPTRAIL_ML_BACKEND_URL;
 process.env.CAPTRAIL_ML_BACKEND_URL='https://secure-backend.example';
 try {
   globalThis.fetch=async(url,opts)=>{assert.equal(url,'https://secure-backend.example/ready');assert.equal(opts.method,'GET');return new Response('{"status":"ready"}',{headers:{'content-type':'application/json'}})};
   const r=await GET(make('https://captrail.example/api/ml-bridge?op=ready'));
   assert.equal(r.status,200);assert.equal((await r.json()).status,'ready');
 }finally{globalThis.fetch=original;if(prior===undefined)delete process.env.CAPTRAIL_ML_BACKEND_URL;else process.env.CAPTRAIL_ML_BACKEND_URL=prior}
});
test('POST without the same-origin marker is refused before reaching the backend',async()=>{
 const original=globalThis.fetch, prior=process.env.CAPTRAIL_ML_BACKEND_URL;let called=false;
 process.env.CAPTRAIL_ML_BACKEND_URL='https://secure-backend.example';
 try {
   globalThis.fetch=async()=>{called=true;return new Response('{}',{headers:{'content-type':'application/json'}})};
   const r=await POST(make('https://captrail.example/api/ml-bridge?op=analyze','POST',{headers:{'content-type':'multipart/form-data; boundary=x'},body:'--x--'}));
   assert.equal(r.status,403);assert.equal(called,false);
 }finally{globalThis.fetch=original;if(prior===undefined)delete process.env.CAPTRAIL_ML_BACKEND_URL;else process.env.CAPTRAIL_ML_BACKEND_URL=prior}
});
test('POST with the marker forwards it to the backend',async()=>{
 const original=globalThis.fetch, prior=process.env.CAPTRAIL_ML_BACKEND_URL;
 process.env.CAPTRAIL_ML_BACKEND_URL='https://secure-backend.example';
 try {
   globalThis.fetch=async(url,opts)=>{assert.equal(url,'https://secure-backend.example/api/analyses');assert.equal(opts.headers['x-captrail-request'],'1');return new Response('{"analyses":[]}',{headers:{'content-type':'application/json'}})};
   const r=await POST(make('https://captrail.example/api/ml-bridge?op=analyze','POST',{headers:{'x-captrail-request':'1','content-type':'multipart/form-data; boundary=x'},body:'--x--'}));
   assert.equal(r.status,200);
 }finally{globalThis.fetch=original;if(prior===undefined)delete process.env.CAPTRAIL_ML_BACKEND_URL;else process.env.CAPTRAIL_ML_BACKEND_URL=prior}
});
