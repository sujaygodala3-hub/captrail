/* Dedicated trained CapTrail API adapter. No tokens are persisted or logged. */
(function (root) {
  'use strict';
  const bridge='/api/ml-bridge';
  async function request(op,{method='GET',body}={}) {
    let response;
    try { response=await fetch(bridge+'?op='+encodeURIComponent(op),{
      method,body,credentials:'same-origin',
      // x-captrail-request marks a same-origin mutation; the bridge and backend require it on POST.
      headers:{...(method==='GET' ? {} : {'x-captrail-request':'1'}),...(body instanceof FormData || !body ? {} : {'content-type':'application/json'})},
      cache:'no-store'
    }); }
    catch { const error=new Error('Cannot reach the trained-model gateway. Check backend hosting.'); error.status=503; throw error; }
    let data;
    try { data=await response.json(); }
    catch { const error=new Error('Gateway returned an invalid response');error.status=response.status;throw error; }
    if(!response.ok){const error=new Error(typeof data?.detail==='string'?data.detail:data?.error||data?.message||'Model request failed');error.status=response.status;throw error;}
    return data;
  }
  function toOverlay(reply,width,height) {
    const analysis=reply?.analyses?.[0];
    if(!analysis)throw new Error('Model response did not include an analysis.');
    const w=analysis.image?.width||width,h=analysis.image?.height||height;
    const sx=width/w,sy=height/h;
    const entries=analysis.detected_structures?.capillary_instances||[];
    const findings=entries.filter(x=>Array.isArray(x.bbox_xyxy)&&x.bbox_xyxy.length===4).map(x=>{
      const [x1,y1,x2,y2]=x.bbox_xyxy;
      return {
        label:x.class_name||'vessel candidate',
        confidence:null, // Uncalibrated detector scores must NOT appear as medical percentages.
        box:[x1*sx,y1*sy,Math.max(0,(x2-x1)*sx),Math.max(0,(y2-y1)*sy)]
      };
    });
    const status=analysis.provenance?.inference_status||'unknown';
    const version=analysis.provenance?.model_version||'unknown';
    const quality=analysis.quality?.technical_rejection?' Image quality insufficient; model abstained.':'';
    const note='Trained model '+version+' ('+status+'). '+findings.length+' detector candidates. Box geometry is not vessel diameter. No diagnosis, healthy-status assessment, or physical-unit measurement is available.'+quality;
    return {findings,width_um:null,density_per_mm:null,note,model_result:analysis};
  }
  root.CapTrailML={
    ready: async ()=>{
      try {return {ok:true,data:await request('ready')};}
      catch(e){return {ok:false,error:e.message,status:e.status};}
    },
    login: payload=>request('login',{method:'POST',body:JSON.stringify(payload)}),
    analyze:(blob,metadata)=>{
      const fd=new FormData();
      fd.append('files',blob,'captrail_capture.jpg');
      fd.append('metadata',JSON.stringify(metadata));
      fd.append('trace','true');
      return request('analyze',{method:'POST',body:fd});
    },
    toOverlay
  };
})(window);
