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
}
export const GET=r=>proxy(r,'GET');export const POST=r=>proxy(r,'POST');export const DELETE=r=>proxy(r,'DELETE');
