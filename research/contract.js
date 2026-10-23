// Shared, pure geometry contract: no anatomical interpolation or smoothing.
export const CLASSES=['bushy','crossing','hairpin','tortuous'];
export const COLORS=['#E7A06B','#B9A7F2','#74D4CC','#F38598'];
export function dimensions(analysis){const {width:w,height:h}=analysis?.image||{};if(!Number.isInteger(w)||!Number.isInteger(h)||w<16||h<16||w*h>16000000)throw new Error('Invalid model image dimensions.');return {w,h};}
export function point(p,w,h){return Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&p[0]>=0&&p[1]>=0&&p[0]<=w&&p[1]<=h;}
export function polyline(points,w,h,min=2){return Array.isArray(points)&&points.length>=min&&points.length<=4096&&points.every(p=>point(p,w,h));}
export function box(b,w,h){return Array.isArray(b)&&b.length===4&&b.every(Number.isFinite)&&b[0]>=0&&b[1]>=0&&b[2]>b[0]&&b[3]>b[1]&&b[2]<=w&&b[3]<=h;}
export function traceGeometry(trace,w,h){
 if(!trace)return {fragments:[],masks:[]};
 if(trace.coordinate_space!=='original_image_pixels')throw new Error('Unknown trace coordinate system. Overlay withheld.');
 const fragments=trace.fragments||[],masks=trace.mask_contours||[];
 if(fragments.length>256||masks.length>256||fragments.some(f=>!polyline(f.centerline,w,h))||masks.some(p=>!polyline(p,w,h,3)))throw new Error('Invalid trace geometry. Overlay withheld.');
 return {fragments,masks};
}
export function modelInstances(a){const {w,h}=dimensions(a);const v=a.detected_structures?.capillary_instances||[];
 if(v.length>1000||v.some(i=>!box(i.bbox_xyxy,w,h)||!CLASSES.includes(i.class_name)))throw new Error('Invalid detection geometry. Overlay withheld.');return v;}
export function effectiveInstances(a){modelInstances(a);if(!a.review)return modelInstances(a);if(a.review.status==='rejected')return [];
 const {w,h}=dimensions(a);if(a.review.instances.some(i=>!box(i.bbox_xyxy,w,h)||(i.centerline&&!polyline(i.centerline,w,h))||(i.polygon&&!polyline(i.polygon,w,h,3))||(i.apex&&!point(i.apex,w,h))))throw new Error('Invalid review geometry.');
 return a.review.instances.filter(i=>!i.excluded).map(i=>({...i,class_name:i.class_ids.map(k=>CLASSES[k]).join(', ')||'unclassified',class_id:i.class_ids.length===1?i.class_ids[0]:null,confidence:null}));}
export function reviewInstances(a){return a.review?structuredClone(a.review.instances):modelInstances(a).map(i=>({instance_id:i.instance_id,bbox_xyxy:[...i.bbox_xyxy],class_ids:[i.class_id],centerline:null,polygon:null,apex:null,distal_row:false,excluded:false}));}
export function lengthPx(points){return points.slice(1).reduce((s,p,i)=>s+Math.hypot(p[0]-points[i][0],p[1]-points[i][1]),0);}
export function imagePoint(clientX,clientY,rect,view){const scale=Math.min(rect.width/view.w,rect.height/view.h);return [view.x+(clientX-rect.left-(rect.width-view.w*scale)/2)/scale,view.y+(clientY-rect.top-(rect.height-view.h*scale)/2)/scale];}
export function pathData(points,closed=false){return points.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' ')+(closed?'Z':'');}
export function pairKey(name){const m=name.match(/^(person\d+-[LR](?:index|middle|ring|pinky|thumb)-\d+)-([ne])\.(png|jpe?g|tiff?|bmp)$/i);return m?{key:m[1].toLowerCase(),representation:m[2].toLowerCase()==='n'?'original':'enhanced'}:{key:name,representation:'original'};}
export async function validateFile(file){
 if(file.size<16||file.size>3.8*1024*1024)throw new Error('Choose a JPEG or PNG under 3.8 MiB. Originals are never silently resized.');
 const b=new Uint8Array(await file.slice(0,16).arrayBuffer());const png=[137,80,78,71,13,10,26,10].every((x,i)=>b[i]===x),jpeg=b[0]===255&&b[1]===216&&b[2]===255;
 if(!png&&!jpeg)throw new Error('This file is not a supported JPEG or PNG image.');
 return png?'image/png':'image/jpeg';
}
