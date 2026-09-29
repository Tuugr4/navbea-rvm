import {useEffect,useState} from 'react';
import {CircleNotch,ClockCounterClockwise,Gauge,WarningCircle} from '@phosphor-icons/react';

type Report={at:string;durationSeconds:number;size:string;mode:string|null;liveModel:string|null;stillModel:string|null;cpu:string|null;gpu:string|null;liveProvider:string|null;photoProvider:string|null;photoModel:string|null;
 live:{meanMs:number;p95Ms:number;fps:number};photo:{firstSeconds:number;seconds:number;runsSeconds:number[];stillModel?:{name:string;provider:string|null;report?:{used?:string;fallback?:string}|null;error?:string|null}};subjects?:{detectMs:number;scoreMs?:number}};
export type BenchmarkStatus={running:boolean;last:Report|null};
export type EngineEvent={at:string;type:string;message:string};
export type BenchmarkBridge={benchmark():Promise<BenchmarkStatus>;benchmarkRun():Promise<BenchmarkStatus>;events():Promise<{events:EngineEvent[]}>};

const unit=(provider:string|null)=>!provider?'—':provider==='CPUExecutionProvider'?'işlemci':provider==='DmlExecutionProvider'?'ekran kartı':provider==='CUDAExecutionProvider'?'ekran kartı (CUDA)':provider.replace('ExecutionProvider','');
const mode=(value:string|null)=>value==='gpu'?'Ekran kartı':value==='auto'?'Otomatik':value==='cpu'?'İşlemci':'Varsayılan';
const time=(iso:string)=>new Date(iso).toLocaleString('tr-TR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'});

export default function BenchmarkSection({bridge,disabled}:{bridge?:BenchmarkBridge;disabled:boolean}){
 const [status,setStatus]=useState<BenchmarkStatus|null>(null),[events,setEvents]=useState<EngineEvent[]>([]),[pending,setPending]=useState(false),[error,setError]=useState('');
 useEffect(()=>{if(!bridge)return;let closed=false;const read=()=>Promise.all([bridge.benchmark(),bridge.events()]).then(([b,e])=>{if(!closed){setStatus(b);setEvents(e.events);}}).catch(()=>undefined);void read();const timer=setInterval(read,2000);return()=>{closed=true;clearInterval(timer);};},[bridge]);
 const run=async()=>{if(!bridge||pending)return;setPending(true);setError('');try{setStatus(await bridge.benchmarkRun());}catch(error){setError(message(error));}finally{setPending(false);}};
 const r=status?.last;const running=pending||!!status?.running;
 return <section className="model-section benchmark-section"><div className="section-heading"><h2>Test</h2><p>Seçili canlı model, fotoğraf modeli ve işlem birimiyle maskeleme sürelerini ölçer. Fotoğraf, kameranın gerçek fotoğraf çözünürlüğünde denenir; kamera veya müşteri görüntüsü kullanılmaz. Test sırasında canlı görüntü yaklaşık bir dakika durur.</p></div>
  <div className="device-apply"><button className="primary" disabled={!bridge||disabled||running} onClick={()=>void run()}>{running?<><CircleNotch className="spin"/>Test ediliyor…</>:<><Gauge/>Test et</>}</button></div>
  {error&&<p className="notice" role="alert"><WarningCircle/>{error}</p>}
  {r&&<div className="benchmark-result">
   <div><small>Canlı maske</small><strong>{r.live.meanMs} ms</strong><em>{r.live.fps} FPS · en yavaş %5: {r.live.p95Ms} ms · {unit(r.liveProvider)}</em></div>
   <div><small>Fotoğraf maskesi</small><strong>{r.photo.seconds.toLocaleString('tr-TR')} sn</strong><em>ilk çekim {r.photo.firstSeconds.toLocaleString('tr-TR')} sn · {r.size.replace('x',' × ')} · {r.photoModel?'ResNet50':r.stillModel&&r.stillModel!=='rvm'?r.stillModel:'RVM'} · {unit(r.photoProvider)}</em>
    {r.photo.stillModel?.report?.fallback&&<em className="model-warning">Fotoğraf modeli kullanılamadı ({r.photo.stillModel.report.fallback}); RVM ile tamamlandı.</em>}</div>
   {r.subjects&&<div><small>Kişi seçimi</small><strong>{((r.subjects.detectMs+(r.subjects.scoreMs||0))/1000).toLocaleString('tr-TR',{maximumFractionDigits:1})} sn</strong><em>tespit {r.subjects.detectMs} ms{r.subjects.scoreMs!=null?` · puanlama ${r.subjects.scoreMs} ms`:''}</em></div>}
   <p className="hardware-line">{time(r.at)} · işlem birimi: {mode(r.mode)} · canlı model: {r.liveModel||'—'} · fotoğraf modeli: {r.stillModel||'—'}{r.gpu?` · ${r.gpu}`:r.cpu?` · ${r.cpu}`:''} · test {r.durationSeconds} sn sürdü</p>
  </div>}
  {!!events.length&&<details className="engine-events" open><summary><ClockCounterClockwise/> Son olaylar</summary><ol>{[...events].reverse().slice(0,25).map((item,index)=><li key={index} className={`event-${item.type}`}><time>{time(item.at)}</time><span>{item.message}</span></li>)}</ol></details>}
 </section>;
}
function message(error:unknown){const raw=error instanceof Error?error.message:String(error);return raw.replace(/^(?:Error: )?Error invoking remote method '[^']+':\s*(?:Error:\s*)?/,'');}
