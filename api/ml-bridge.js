/* Vercel same-origin bridge to a separately hosted CapTrail Python model.
   No public access to raw credentials; backend still authenticates every upload.
   Disabled unless CAPTRAIL_ML_BACKEND_URL is explicitly configured. */
const routes={
  'ready':{method:'GET',path:'/ready'},
  'auth-mode':{method:'GET',path:'/api/auth-mode'},
  'login':{method:'POST',path:'/api/login'},
  'logout':{method:'POST',path:'/api/logout'},
  'capabilities':{method:'GET',path:'/api/capabilities'},
  'analyze':{method:'POST',path:'/api/analyses'}
};
const reply=(data,status)=>new Response(JSON.stringify(data),{status,headers:{'content-type':'application/json','cache-control':'no-store','x-content-type-options':'nosniff'}});
const baseUrl=()=>{
  const source=process.env.CAPTRAIL_ML_BACKEND_URL;
  if(!source)return null;
  let parsed;
  try{parsed=new URL(source)}catch{return null}
  if(parsed.username||parsed.password||parsed.search||parsed.hash)return null;
  if(parsed.protocol!=='https:'&&!(process.env.NODE_ENV!=='production'&&parsed.protocol==='http:'&&['localhost','127.0.0.1'].includes(parsed.hostname)))return null;
  if(parsed.pathname!=='/'&&parsed.pathname!=='')return null;
  return parsed.origin;
};
async function proxy(request,method){
  const url=new URL(request.url);
  const origin=request.headers.get('origin');
  if(origin&&origin!==url.origin)return reply({error:'Cross-site requests are not permitted.'},403);
  const op=url.searchParams.get('op');
  const route=routes[op];
  if(!route||route.method!==method)return reply({error:'Unsupported trained-model operation.'},405);
  const upstream=baseUrl();
  if(!upstream)return reply({error:'Trained model not configured. Deploy the private Python API over HTTPS and set CAPTRAIL_ML_BACKEND_URL in Vercel.'},503);
  // Same-origin mutation marker (required by the CapTrail backend): demand it from the browser, then forward it.
  if(method==='POST'&&request.headers.get('x-captrail-request')!=='1')return reply({error:'Missing same-origin request marker.'},403);
  const length=Number(request.headers.get('content-length')||0);
  if(length>4*1024*1024)return reply({error:'Image exceeds 4 MB gateway request limit.'},413);
  let response;
  try{
    const headers={accept:'application/json'};
    const cookie=(request.headers.get('cookie')||'').split(';').map(s=>s.trim()).find(s=>s.startsWith('captrail_session='));
    if(cookie)headers.cookie=cookie;
    if(method==='POST'){headers['content-type']=request.headers.get('content-type')||'application/octet-stream';headers['x-captrail-request']='1';}
    response=await fetch(upstream+route.path,{
      method,headers,body:method==='POST'?request.body:undefined,
      ...(method==='POST'?{duplex:'half'}:{}),redirect:'manual',
      signal:AbortSignal.timeout(op==='analyze'?120000:20000)
    });
  }catch{return reply({error:'Trained model host unavailable or timed out.'},503)}
  if(response.status>=300&&response.status<400)return reply({error:'Unexpected backend redirect.'},502);
  const responseHeaders=new Headers({'cache-control':'no-store','x-content-type-options':'nosniff'});
  const type=response.headers.get('content-type')||'application/json';
  if(!type.includes('application/json'))return reply({error:'Unexpected backend response format.'},502);
  responseHeaders.set('content-type','application/json');
  if(op==='login'||op==='logout'){
    const session=response.headers.get('set-cookie');
    if(session&&/^(captrail_session=)/i.test(session))responseHeaders.set('set-cookie',session);
  }
  return new Response(response.body,{status:response.status,headers:responseHeaders});
}
export const GET=request=>proxy(request,'GET');
export const POST=request=>proxy(request,'POST');
