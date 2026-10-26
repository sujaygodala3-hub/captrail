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

test('mutations require a CSRF marker even when Origin is omitted',async()=>{
 assert.equal((await POST(make('https://captrail.example/api/ml-bridge?op=login','POST',{body:'{}'}))).status,403);
});
test('cookie upload forwards the marker, only the session cookie, and preserves bytes',async()=>{
 const original=globalThis.fetch,prior=process.env.CAPTRAIL_ML_BACKEND_URL;process.env.CAPTRAIL_ML_BACKEND_URL='https://secure-backend.example';
 try{globalThis.fetch=async(u,o)=>{assert.equal(u,'https://secure-backend.example/api/analyses');assert.equal(o.headers['x-captrail-request'],'1');assert.equal(o.headers.cookie,'captrail_session=test-session');assert.equal(new TextDecoder().decode(o.body),'original-bytes');return new Response('{}',{headers:{'content-type':'application/json'}});};
 const r=await POST(make('https://captrail.example/api/ml-bridge?op=analyze','POST',{headers:{'x-captrail-request':'1',cookie:'other=private; captrail_session=test-session'},body:'original-bytes'}));assert.equal(r.status,200);
 }finally{globalThis.fetch=original;if(prior===undefined)delete process.env.CAPTRAIL_ML_BACKEND_URL;else process.env.CAPTRAIL_ML_BACKEND_URL=prior;}
});
test('unknown-length streamed uploads stop at the byte budget before upstream',async()=>{
 const prior=process.env.CAPTRAIL_ML_BACKEND_URL;process.env.CAPTRAIL_ML_BACKEND_URL='https://secure-backend.example';const original=globalThis.fetch;
 try{globalThis.fetch=()=>{throw new Error('Must not send oversized bytes');};const r=await POST(new Request('https://captrail.example/api/ml-bridge?op=analyze',{method:'POST',headers:{'x-captrail-request':'1'},body:new ReadableStream({start(c){c.enqueue(new Uint8Array(4*1024*1024+1));c.close();}}),duplex:'half'}));assert.equal(r.status,413);}finally{globalThis.fetch=original;if(prior===undefined)delete process.env.CAPTRAIL_ML_BACKEND_URL;else process.env.CAPTRAIL_ML_BACKEND_URL=prior;}
});
test('record IDs cannot inject upstream paths',async()=>{assert.equal((await GET(make('https://captrail.example/api/ml-bridge?op=detail&id=../secrets'))).status,400);});
for(const status of [400,401,403,413,422,429,500,503])test(`gateway preserves backend error ${status}`,async()=>{
 const prior=process.env.CAPTRAIL_ML_BACKEND_URL,original=globalThis.fetch;process.env.CAPTRAIL_ML_BACKEND_URL='https://secure-backend.example';
 try{globalThis.fetch=async()=>new Response('{"detail":"test error"}',{status,headers:{'content-type':'application/json'}});const r=await GET(make('https://captrail.example/api/ml-bridge?op=ready'));assert.equal(r.status,status);assert.equal((await r.json()).detail,'test error');}finally{globalThis.fetch=original;if(prior===undefined)delete process.env.CAPTRAIL_ML_BACKEND_URL;else process.env.CAPTRAIL_ML_BACKEND_URL=prior;}
});
test('login rewrites cookie to host-only HttpOnly SameSite Strict Secure',async()=>{const prior=process.env.CAPTRAIL_ML_BACKEND_URL,original=globalThis.fetch;process.env.CAPTRAIL_ML_BACKEND_URL='https://secure-backend.example';try{globalThis.fetch=async()=>new Response('{}',{headers:{'content-type':'application/json','set-cookie':'captrail_session=test; Domain=backend.example; Path=/bad; SameSite=None'}});const r=await POST(make('https://captrail.example/api/ml-bridge?op=login','POST',{headers:{'x-captrail-request':'1'},body:'{}'}));assert.equal(r.headers.get('set-cookie'),'captrail_session=test; Path=/; HttpOnly; SameSite=Strict; Max-Age=28800; Secure');}finally{globalThis.fetch=original;if(prior===undefined)delete process.env.CAPTRAIL_ML_BACKEND_URL;else process.env.CAPTRAIL_ML_BACKEND_URL=prior;}});

test('logout accepts the backend quoted empty deletion cookie and clears host cookie',async()=>{const prior=process.env.CAPTRAIL_ML_BACKEND_URL,original=globalThis.fetch;process.env.CAPTRAIL_ML_BACKEND_URL='https://secure-backend.example';try{globalThis.fetch=async()=>new Response('{"ok":true}',{headers:{'content-type':'application/json','set-cookie':'captrail_session=""; expires=Thu, 01 Jan 1970 00:00:00 GMT; Max-Age=0; Path=/'}});const r=await POST(make('https://captrail.example/api/ml-bridge?op=logout','POST',{headers:{'x-captrail-request':'1'},body:'{}'}));assert.equal(r.status,200);assert.match(r.headers.get('set-cookie'),/^captrail_session=; Path=\/; HttpOnly; SameSite=Strict; Max-Age=0; Secure$/);}finally{globalThis.fetch=original;if(prior===undefined)delete process.env.CAPTRAIL_ML_BACKEND_URL;else process.env.CAPTRAIL_ML_BACKEND_URL=prior;}});
for(const [op,method,path,binary] of [['calibrations','GET','/api/calibrations'],['calibration-proposal','POST','/api/calibration-proposals'],['save-calibration','POST','/api/calibrations'],['participants','GET','/api/participants'],['create-participant','POST','/api/participants'],['examinations','GET','/api/examinations'],['calibration-image','GET','/api/calibrations/'+ 'a'.repeat(32)+'/image',true],['archive-participant','POST','/api/participants/'+'a'.repeat(32)],['consent','POST','/api/analyses/'+'a'.repeat(32)+'/training-consent'],['anatomy-readings','GET','/api/analyses/'+'a'.repeat(32)+'/anatomy/comparison']])test(`V7 allowlisted operation ${op} preserves method and identity`,async()=>{
 const original=fetch,prior=process.env.CAPTRAIL_ML_BACKEND_URL;process.env.CAPTRAIL_ML_BACKEND_URL='https://backend.example';
 try{globalThis.fetch=async(url,o)=>{assert.equal(url,'https://backend.example'+path);assert.equal(o.method,method);return new Response(binary?new Uint8Array([137,80,78,71]):'{}',{headers:{'content-type':binary?'image/png':'application/json'}});};const r=await (method==='GET'?GET:POST)(new Request('https://captrail.example/api/ml-bridge?op='+op+'&id='+'a'.repeat(32),{method,headers:{'x-captrail-request':'1'},...(method==='POST'?{body:'{}'}:{})}));assert.equal(r.status,200);assert.equal(r.headers.get('cache-control'),'no-store');}finally{globalThis.fetch=original;if(prior===undefined)delete process.env.CAPTRAIL_ML_BACKEND_URL;else process.env.CAPTRAIL_ML_BACKEND_URL=prior;}
});
test('production HTTP is denied except explicit fixed private-compose model service',async()=>{const before={url:process.env.CAPTRAIL_ML_BACKEND_URL,env:process.env.NODE_ENV,internal:process.env.CAPTRAIL_INTERNAL_NETWORK},original=fetch;try{process.env.NODE_ENV='production';process.env.CAPTRAIL_ML_BACKEND_URL='http://model:8000';delete process.env.CAPTRAIL_INTERNAL_NETWORK;assert.equal((await GET(make('https://captrail.example/api/ml-bridge?op=ready'))).status,503);process.env.CAPTRAIL_INTERNAL_NETWORK='1';globalThis.fetch=async(u,o)=>{assert.equal(u,'http://model:8000/ready');assert.equal(o.headers['x-forwarded-proto'],'https');return new Response('{}',{headers:{'content-type':'application/json'}});};assert.equal((await GET(make('https://captrail.example/api/ml-bridge?op=ready'))).status,200);process.env.CAPTRAIL_ML_BACKEND_URL='http://other-host:8000';assert.equal((await GET(make('https://captrail.example/api/ml-bridge?op=ready'))).status,503);}finally{globalThis.fetch=original;for(const [key,value] of Object.entries({CAPTRAIL_ML_BACKEND_URL:before.url,NODE_ENV:before.env,CAPTRAIL_INTERNAL_NETWORK:before.internal})){if(value===undefined)delete process.env[key];else process.env[key]=value;}}});
