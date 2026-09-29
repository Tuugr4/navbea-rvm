import {useEffect,useState,type ReactElement} from 'react';
import {CheckCircle,CircleNotch,Cpu,Info,Lightning,Monitor,WarningCircle} from '@phosphor-icons/react';

type Mode='auto'|'gpu'|'cpu';
export type DeviceStatus={available:boolean;mode:Mode;explicit:boolean;gpuAvailable:boolean;
 hardware:{cpu:{name:string|null;logical:number;physical:number|null};memoryGiB:number;gpus:{name:string;discrete:boolean;videoMemory:string|null;inUse:boolean}[]};
 running:{live:string|null;photo:string|null;adapter:string|null;photoFallback:string|null;liveFallback:string|null};switching:boolean;activeSessions:number};
export type DeviceBridge={device():Promise<DeviceStatus>;deviceAction(id:Mode):Promise<DeviceStatus>};

const OPTIONS:{id:Mode;title:string;text:string;icon:ReactElement}[]=[
 {id:'auto',title:'Otomatik',text:'Ekran kartı varsa onu, yoksa işlemciyi kullanır. Ekran kartı hata verirse fotoğraf işlemcide tamamlanır.',icon:<Lightning/>},
 {id:'gpu',title:'Ekran kartı (GPU)',text:'Canlı maske ve fotoğraf maskesi ekran kartında çalışır. En hızlısı; kart başlatılamazsa ayar geri alınır.',icon:<Monitor/>},
 {id:'cpu',title:'İşlemci (CPU)',text:'Her şey işlemcide çalışır. En uyumlu seçenek; büyük fotoğraf modelinde fotoğraf başına birkaç saniye ekler.',icon:<Cpu/>},
];
const unit=(provider:string|null)=>!provider?'—':provider==='CPUExecutionProvider'?'İşlemci':provider==='DmlExecutionProvider'?'Ekran kartı (DirectML)':provider==='CUDAExecutionProvider'?'Ekran kartı (CUDA)':provider.replace('ExecutionProvider','');

export default function DeviceSection({bridge,disabled}:{bridge?:DeviceBridge;disabled:boolean}){
 const [status,setStatus]=useState<DeviceStatus|null>(null),[choice,setChoice]=useState<Mode|null>(null),[pending,setPending]=useState(false),[error,setError]=useState('');
 useEffect(()=>{if(!bridge)return;let closed=false;const read=()=>bridge.device().then(value=>{if(!closed)setStatus(value);}).catch(()=>undefined);void read();const timer=setInterval(read,3000);return()=>{closed=true;clearInterval(timer);};},[bridge]);
 const selected=choice??status?.mode??'cpu';
 const apply=async()=>{if(!bridge||pending||!choice)return;setPending(true);setError('');try{setStatus(await bridge.deviceAction(choice));setChoice(null);}catch(error){setError(message(error));}finally{setPending(false);}};
 const busy=pending||disabled||!!status?.switching||(status?.activeSessions??0)>0;
 const hw=status?.hardware;
 return <section className="model-section device-section"><div className="section-heading"><h2>İşlem birimi</h2><p>Maskelemenin ekran kartında mı işlemcide mi çalışacağını seç. Değişiklik, çekim oturumu ve canlı önizleme kapalıyken uygulanır.</p></div>
  {hw&&<div className="device-hardware">
   <div><Cpu/><span><small>İşlemci</small><strong>{hw.cpu.name||'Bilinmiyor'}</strong><em>{hw.cpu.physical?`${hw.cpu.physical} çekirdek · `:''}{hw.cpu.logical} iş parçacığı · {hw.memoryGiB} GB RAM</em></span></div>
   {hw.gpus.length?hw.gpus.map(gpu=><div key={gpu.name} className={gpu.inUse?'in-use':''}><Monitor/><span><small>Ekran kartı{gpu.inUse?' · kullanılıyor':''}</small><strong>{gpu.name}</strong><em>{gpu.discrete?'Harici':'Dahili'}{gpu.videoMemory?` · ${gpu.videoMemory} video belleği`:''}</em></span></div>)
    :<div><Monitor/><span><small>Ekran kartı</small><strong>Kullanılabilir ekran kartı bulunamadı</strong><em>Maskeleme işlemcide çalışır.</em></span></div>}
  </div>}
  <div className="model-grid">{OPTIONS.map(option=>{const off=option.id==='gpu'&&status!=null&&!status.gpuAvailable;return <article key={option.id} className={`model-card ${status?.mode===option.id?'active':''} ${selected===option.id?'selected':''}`}>
   <div className="model-title"><h3>{option.icon} {option.title}</h3>{status?.mode===option.id&&<span>Etkin</span>}</div><p>{option.text}</p>
   <div className="model-actions"><button className={selected===option.id?'primary':''} disabled={busy||off} onClick={()=>setChoice(option.id)}>{off?'Ekran kartı yok':selected===option.id?<><CheckCircle/>Seçili</>:'Seç'}</button></div>
  </article>;})}</div>
  {choice&&choice!==status?.mode&&<div className="device-apply"><button className="primary" disabled={busy} onClick={()=>void apply()}>{pending?<><CircleNotch className="spin"/>Uygulanıyor ve test ediliyor…</>:'Uygula'}</button><button disabled={pending} onClick={()=>setChoice(null)}>Vazgeç</button></div>}
  {!!status?.activeSessions&&<p className="notice"><Info/>Aktif çekim oturumu var. İşlem birimini değiştirmek için oturumun bitmesini bekle.</p>}
  {error&&<p className="notice" role="alert"><WarningCircle/>{error}</p>}
  {status?.running&&<p className="hardware-line">Şu an · canlı maske: <strong>{unit(status.running.live)}</strong> · fotoğraf maskesi: <strong>{unit(status.running.photo)}</strong>{status.running.adapter?` · ${status.running.adapter}`:''}</p>}
  {status?.running.photoFallback&&<p className="model-warning">Fotoğraf maskesi ekran kartında çalışamadı, işlemciye geçildi: {status.running.photoFallback}</p>}
  {!status?.available&&<div className="notice"><Info/>İşlem birimi seçimi RVM hizmetine bağlanınca görünür.</div>}
 </section>;
}
function message(error:unknown){const raw=error instanceof Error?error.message:String(error);return raw.replace(/^(?:Error: )?Error invoking remote method '[^']+':\s*(?:Error:\s*)?/,'');}
