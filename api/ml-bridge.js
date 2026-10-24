/* Same-origin, authenticated bridge. Upstream location is server configuration only. */
const MiB=1024*1024;
export const MAX_BODY=4*MiB;
const fixed={ready:['GET','/ready'],'auth-mode':['GET','/api/auth-mode'],me:['GET','/api/me'],login:['POST','/api/login'],logout:['POST','/api/logout'],capabilities:['GET','/api/capabilities'],analyze:['POST','/api/analyses'],sessions:['GET','/api/analyses'],models:['GET','/api/segmentation-models'],calibrations:['GET','/api/calibrations'],'calibration-proposal':['POST','/api/calibration-proposals'],'save-calibration':['POST','/api/calibrations'],participants:['GET','/api/participants'],'create-participant':['POST','/api/participants'],examinations:['GET','/api/examinations'],compare:['POST','/api/compare']};
const records={detail:['GET',''],image:['GET','/image'],review:['POST','/review'],history:['GET','/history'],segmentation:['POST','/segmentation-proposal'],trace:['POST','/trace-proposal'],consent:['POST','/training-consent'],'anatomy-readings':['GET','/anatomy/comparison'],'anatomy-adjudication':['GET','/anatomy/adjudicated'],'save-adjudication':['POST','/anatomy/adjudicate'],remove:['DELETE',''],anatomy:['GET','/anatomy/me'],'save-anatomy':['POST','/anatomy/me']};
const reply=(data,status)=>new Response(JSON.stringify(data),{status,headers:{'content-type':'application/json','cache-control':'no-store','x-content-type-options':'nosniff'}});
function baseUrl(){
 try{const p=new URL(process.env.CAPTRAIL_ML_BACKEND_URL);if(p.username||p.password||p.search||p.hash||p.pathname!=='/')return null;
 const internal=process.env.CAPTRAIL_INTERNAL_NETWORK==='1'&&p.origin==='http://model:8000';
 if(p.protocol!=='https:'&&!internal&&!(process.env.NODE_ENV!=='production'&&p.protocol==='http:'&&['localhost','127.0.0.1'].includes(p.hostname)))return null;return p.origin;}catch{return null;}
}
async function boundedBody(request){
 const length=request.headers.get('content-length');
 if(length!==null&&(!/^\d+$/.test(length)||Number(length)>MAX_BODY))throw Object.assign(new Error(),{status:413});
 if(!request.body)return undefined;
 const reader=request.body.getReader(),chunks=[];let total=0;
 try{while(true){const {value,done}=await reader.read();if(done)break;total+=value.byteLength;if(total>MAX_BODY){await reader.cancel();throw Object.assign(new Error(),{status:413});}chunks.push(value);}}
 finally{reader.releaseLock();}
 const body=new Uint8Array(total);let offset=0;for(const c of chunks){body.set(c,offset);offset+=c.length;}return body;
}
async function proxy(request,method){
 const url=new URL(request.url),origin=request.headers.get('origin'),op=url.searchParams.get('op');
 if((origin&&origin!==url.origin)||request.headers.get('sec-fetch-site')==='cross-site')return reply({error:'Cross-site requests are not permitted.'},403);
 let route=fixed[op];
 if(records[op]){const id=url.searchParams.get('id');if(!/^[a-f0-9]{32}$/.test(id||''))return reply({error:'Invalid analysis identifier.'},400);route=[records[op][0],'/api/analyses/'+id+records[op][1]];}
 const other={'calibration-image':['GET','/api/calibrations/','/image'],'calibration-record':['GET','/api/calibrations/',''],'remove-calibration':['DELETE','/api/calibrations/',''],'archive-participant':['POST','/api/participants/','']};
 if(other[op]){const id=url.searchParams.get('id');if(!/^[a-f0-9]{32}$/.test(id||''))return reply({error:'Invalid record identifier.'},400);const [verb,prefix,suffix]=other[op];route=[verb,prefix+id+suffix];}
 if(!route||route[0]!==method)return reply({error:'Unsupported trained-model operation.'},405);
 if(method!=='GET'&&request.headers.get('x-captrail-request')!=='1')return reply({error:'Missing same-origin request marker.'},403);
 const upstream=baseUrl();if(!upstream)return reply({error:'The research model is temporarily unavailable. Your photo has not been sent.'},503);
 const binary=['image','calibration-image'].includes(op);
 const headers={accept:binary?'image/png':'application/json'};
 if(upstream==='http://model:8000'&&process.env.CAPTRAIL_INTERNAL_NETWORK==='1')headers['x-forwarded-proto']=url.protocol==='https:'?'https':'http';
 const cookie=(request.headers.get('cookie')||'').split(';').map(x=>x.trim()).find(x=>/^captrail_session=[A-Za-z0-9._-]+$/.test(x));if(cookie)headers.cookie=cookie;
 let body;
 try{if(method!=='GET'){headers['x-captrail-request']='1';headers['content-type']=request.headers.get('content-type')||'application/json';body=await boundedBody(request);}}
 catch(e){return reply({error:'This request exceeds the 4 MiB limit. Select fewer or smaller originals; no automatic resizing is applied.'},e.status||400);}
 let path=route[1];if(op==='image')path+='?enhanced='+String(url.searchParams.get('enhanced')==='true');
 let response;
 try{response=await fetch(upstream+path,{method,headers,body,redirect:'manual',signal:AbortSignal.timeout(['analyze','segmentation'].includes(op)?120000:20000)});}
 catch{return reply({error:'The model connection was interrupted. Reopen saved sessions before retrying an upload; it may have finished on the server.'},503);}
 if(response.status>=300&&response.status<400)return reply({error:'Unexpected backend redirect.'},502);
 const type=response.headers.get('content-type')||'';
 if(!(type.includes('application/json')||(binary&&response.ok&&type.startsWith('image/png'))))return reply({error:'The model returned an unreadable response.'},response.status>=400?response.status:502);
 const out=new Headers({'content-type':type,'cache-control':'no-store','x-content-type-options':'nosniff','referrer-policy':'no-referrer'});
 if(op==='login'||op==='logout'){
  const session=response.headers.get('set-cookie');
  if(session){const match=op==='logout'&&response.ok?['','']:session.match(/^captrail_session=([A-Za-z0-9._-]*)(?:;|$)/);if(!match)return reply({error:'Invalid session response.'},502);
   out.set('set-cookie',`captrail_session=${match[1]}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${op==='logout'?0:28800}${url.protocol==='https:'?'; Secure':''}`);}
 }
 return new Response(response.body,{status:response.status,headers:out});
}
export const GET=r=>proxy(r,'GET');export const POST=r=>proxy(r,'POST');export const DELETE=r=>proxy(r,'DELETE');
