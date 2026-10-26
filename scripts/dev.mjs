// Local private preview: one managed web listener and a loopback-only Python model.
import http from 'node:http';
import {spawn} from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Readable} from 'node:stream';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const flag=(name,fallback)=>{const i=process.argv.indexOf(name);return i>=0?process.argv[i+1]:fallback;};
const port=Number(flag('--port',process.env.PORT||3000)),host=flag('--host','0.0.0.0');
const envFile=process.env.CAPTRAIL_PREVIEW_ENV||path.join(root,'.preview-runtime/config.json');
let config={};try{config=JSON.parse((await fs.readFile(envFile,'utf8')).replaceAll('@site',root));}catch(e){if(process.env.NODE_ENV!=='production')await fs.writeFile(path.join(root,'.preview-status.json'),JSON.stringify({configuration_error:e.code,uid:process.getuid?.(),config_path:envFile,root,cwd:process.cwd()}));}
if(config.backend_dir)await fs.writeFile(path.join(root,'.preview-status.json'),JSON.stringify({configuration_loaded:true,uid:process.getuid?.()}));
let backend;

if(config.backend_dir&&config.python){
 const backendPort=Number(config.backend_port||8765);
 process.env.CAPTRAIL_ML_BACKEND_URL='http://127.0.0.1:'+backendPort;
 backend=spawn(config.python,['-m','captrail.run'],{cwd:config.backend_dir,env:{...process.env,...config.environment,PORT:String(backendPort),CAPTRAIL_HOST:'127.0.0.1',CAPTRAIL_TRUSTED_PROXY_IPS:'127.0.0.1',ATEN_CPU_CAPABILITY:'default'},stdio:['ignore','ignore','pipe']});
 let modelError='';backend.stderr.on('data',data=>{modelError=(modelError+data.toString()).slice(-5000);fs.writeFile(path.join(root,'.preview-python-status.json'),JSON.stringify({model_error:modelError}));});
 backend.on('error',async e=>{await fs.writeFile(path.join(root,'.preview-status.json'),JSON.stringify({model_spawn_error:e.code}));});
 backend.on('exit',async code=>{await fs.writeFile(path.join(root,'.preview-status.json'),JSON.stringify({model_exit:code}));});
}
const bridge=await import('../api/ml-bridge.js');
const optionalHandlers={
 '/api/voice':await import('../api/voice.js'),
 '/api/analyze':await import('../api/analyze.js'),
};
const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.woff2':'font/woff2','.ico':'image/x-icon'};
const server=http.createServer(async(req,res)=>{
 try{
 const incoming=new URL(req.url,'http://localhost');
 const publicHost=req.headers.host;
 // Managed preview terminates HTTPS; backend trusts only this loopback BFF.
 const protocol=req.headers['x-forwarded-proto']==='https'?'https':'http';
 if(incoming.pathname==='/api/ml-bridge'||optionalHandlers[incoming.pathname]){
  const headers=new Headers();for(const [k,v] of Object.entries(req.headers))if(v!==undefined)headers.set(k,Array.isArray(v)?v.join(','):v);
  const r=new Request(`${protocol}://${publicHost}${req.url}`,{method:req.method,headers,...(!['GET','HEAD'].includes(req.method)?{body:Readable.toWeb(req),duplex:'half'}:{})});
  const route=incoming.pathname==='/api/ml-bridge'?bridge:optionalHandlers[incoming.pathname];
  const handler=route[req.method];const result=handler?await handler(r):new Response('Method not allowed',{status:405});
  res.writeHead(result.status,Object.fromEntries(result.headers));if(result.body)Readable.fromWeb(result.body).pipe(res);else res.end();return;
 }
 if(incoming.pathname.startsWith('/api/')){res.writeHead(503,{'content-type':'application/json'});res.end(JSON.stringify({error:'Optional external commentary is not configured in this private preview.'}));return;}
 const pathname=decodeURIComponent(incoming.pathname);const relative=pathname==='/'?'scan.html':pathname.replace(/^\//,'');
 const p=path.resolve(root,relative);
 if(!p.startsWith(root+path.sep)||relative.split('/').some(s=>s.startsWith('.'))||!mime[path.extname(p)]||relative.startsWith('api/')||relative.startsWith('scripts/')||relative.startsWith('tests/')){res.writeHead(404);res.end();return;}
 const body=await fs.readFile(p);res.writeHead(200,{'content-type':mime[path.extname(p)],'cache-control':'no-store','x-content-type-options':'nosniff','referrer-policy':'no-referrer','x-frame-options':'DENY','permissions-policy':'camera=(self), microphone=(self)'});res.end(body);
 }catch{res.writeHead(404);res.end('Not found');}
});
server.listen(port,host,()=>console.log('CapTrail private preview listener ready'));
if(backend){let probes=0;const timer=setInterval(async()=>{try{const r=await fetch(process.env.CAPTRAIL_ML_BACKEND_URL+'/ready',{signal:AbortSignal.timeout(15000)});const data=await r.json();await fs.writeFile(path.join(root,'.preview-readiness.json'),JSON.stringify({status:r.status,...data}));if(r.ok||++probes>=3)clearInterval(timer);}catch{if(++probes>=3)clearInterval(timer);}},16000);}
for(const sig of ['SIGINT','SIGTERM'])process.on(sig,()=>{backend?.kill('SIGTERM');server.close(()=>process.exit());});
