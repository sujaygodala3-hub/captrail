export const endpoint=(op,id,extra={})=>'/api/ml-bridge?'+new URLSearchParams({op,...(id?{id}:{}),...extra});
export async function request(op,{id,method='GET',body,signal}={}){
 let r;try{r=await fetch(endpoint(op,id),{method,body,signal,credentials:'same-origin',cache:'no-store',headers:{'X-Captrail-Request':'1',...(body&&!(body instanceof FormData)?{'Content-Type':'application/json'}:{})}});}catch(e){if(e.name==='AbortError')throw e;throw new Error('Connection interrupted. Check saved sessions before retrying; your upload may have finished.');}
 let data;try{data=await r.json();}catch{throw new Error('The research service returned an unreadable response.');}
 if(!r.ok)throw Object.assign(new Error(typeof data.detail==='string'?data.detail:data.error||`Request failed (${r.status}).`),{status:r.status});return data;
}
export const json=(op,body,id)=>request(op,{method:'POST',id,body:JSON.stringify(body)});
