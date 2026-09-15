import {useEffect,useRef,useState} from 'react';
import {Camera,CircleNotch,Stop} from '@phosphor-icons/react';
import './foreground-preview.css';

export type PreviewFrame={status:'waiting'}|{status:'frame';sequence:string;width:number;height:number;sourceAtEpochMs:number;jpeg:Uint8Array;mask:Uint8Array};
export type PreviewBridge={previewStart():Promise<{started:boolean}>;previewStop():Promise<unknown>;previewFrame():Promise<PreviewFrame>};
export default function ForegroundPreview({bridge,disabled,onActiveChange}:{bridge?:PreviewBridge;disabled:boolean;onActiveChange:(active:boolean)=>void}) {
  const canvas=useRef<HTMLCanvasElement>(null), generation=useRef(0);
  const [running,setRunning]=useState(false),[starting,setStarting]=useState(false),[status,setStatus]=useState('Kapalı'),[error,setError]=useState('');
  const notify=useRef(onActiveChange);notify.current=onActiveChange;
  const stop=async()=>{generation.current++;setStarting(false);setRunning(false);setStatus('Kapalı');notify.current(false);const c=canvas.current;c?.getContext('2d')?.clearRect(0,0,c.width,c.height);try{await bridge?.previewStop();}catch(error){setError(String(error));}};
  const start=async()=>{if(!bridge||starting||running)return;const id=++generation.current;setStarting(true);setError('');setStatus('Kamera ve RVM hazırlanıyor…');notify.current(true);try{const result=await bridge.previewStart();if(id!==generation.current)return;if(result.started){setRunning(true);setStatus('Güncel görüntü bekleniyor…');}else notify.current(false);}catch(error){if(id===generation.current){setError((error instanceof Error?error.message:String(error)).replace(/^(?:Error: )?Error invoking remote method '[^']+':\s*(?:Error:\s*)?/,''));setStatus('Bağlantı kurulamadı');notify.current(false);}}finally{if(id===generation.current)setStarting(false);}};
  useEffect(()=>()=>{generation.current++;void bridge?.previewStop?.().catch(()=>{});},[bridge]);
  useEffect(()=>{
    if(!running||!bridge)return;
    let closed=false,timer=0,expiry=0,lastSequence='';
    const clear=()=>{const c=canvas.current;c?.getContext('2d')?.clearRect(0,0,c.width,c.height);setStatus('Güncel görüntü bekleniyor…');};
    const poll=async()=>{
      try{
        const frame=await bridge.previewFrame();if(closed)return;
        if(frame.status==='frame'&&frame.sequence!==lastSequence&&Date.now()-frame.sourceAtEpochMs>=-100&&Date.now()-frame.sourceAtEpochMs<=700){
          const bitmap=await createImageBitmap(new Blob([new Uint8Array(frame.jpeg)],{type:'image/jpeg'}));
          try{
            if(closed||Date.now()-frame.sourceAtEpochMs>700)return;
            if(frame.mask.length!==frame.width*frame.height||Math.abs(bitmap.width/bitmap.height-frame.width/frame.height)>.03)throw Error('Kamera ve maske boyutları uyuşmuyor.');
            const c=canvas.current;if(!c)return;const context=c.getContext('2d',{willReadFrequently:true});if(!context)throw Error('Önizleme çizilemedi.');
            if(c.width!==frame.width||c.height!==frame.height){c.width=frame.width;c.height=frame.height;}
            context.clearRect(0,0,c.width,c.height);context.drawImage(bitmap,0,0,c.width,c.height);
            const pixels=context.getImageData(0,0,c.width,c.height);for(let i=0;i<frame.mask.length;i++)pixels.data[i*4+3]=frame.mask[i];context.putImageData(pixels,0,0);
            lastSequence=frame.sequence;setStatus('Canlı · arka plan kaldırıldı');clearTimeout(expiry);expiry=window.setTimeout(clear,Math.max(1,700-(Date.now()-frame.sourceAtEpochMs)));
          }finally{bitmap.close();}
        }
      }catch(error){if(!closed){clear();setError(error instanceof Error?error.message:String(error));setRunning(false);notify.current(false);void bridge.previewStop().catch(()=>{});}}
      finally{if(!closed)timer=window.setTimeout(poll,125);}
    };
    void poll();return()=>{closed=true;clearTimeout(timer);clearTimeout(expiry);const c=canvas.current;c?.getContext('2d')?.clearRect(0,0,c.width,c.height);};
  },[running,bridge]);
  return <section className="foreground-preview"><div className="foreground-heading"><div><h2>Arka plansız canlı görüntü</h2><p>Kameradaki kişi, mevcut RVM maskesiyle gösterilir.</p></div><button disabled={!bridge||(!running&&!starting&&disabled)} onClick={()=>void (running||starting?stop():start())}>{starting?<CircleNotch className="spin"/>:running?<Stop/>:<Camera/>}{starting?'İptal et':running?'Önizlemeyi kapat':'Önizlemeyi aç'}</button></div>
    <div className="foreground-stage"><canvas ref={canvas} width={1280} height={720} aria-label="Arka planı kaldırılmış kamera görüntüsü"/>{!running&&<div className="foreground-empty"><Camera/><span>{starting?'Kamera hazırlanıyor…':'Kişi burada görünecek'}</span></div>}</div>
    <div className="foreground-caption"><span role="status">{status}</span><small>Dama deseni şeffaf alanı gösterir · Tanılama önizlemesi en fazla 8 FPS</small></div>
    {error&&<p role="alert" className="notice">{error}</p>}
    <p className="foreground-help">Önce Camera Runtime’da kamerayı doğrulayıp kaydet. Model değiştirmeden önce önizlemeyi kapat.</p>
  </section>;
}
