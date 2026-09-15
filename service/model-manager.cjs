const fs=require('node:fs/promises');const path=require('node:path');const crypto=require('node:crypto');const os=require('node:os');
const catalog=require('../models/catalog.json');
const UPSTREAM='https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/';
const allowedHosts=new Set(['github.com','release-assets.githubusercontent.com','objects.githubusercontent.com']);
async function hashFile(file){const hash=crypto.createHash('sha256');const stream=require('node:fs').createReadStream(file);for await(const chunk of stream)hash.update(chunk);return hash.digest('hex');}
class ModelManager {
 constructor({dataRoot,bundledRoot,probe,catalogue=catalog,fetcher=fetch,memory=()=>({total:os.totalmem(),free:os.freemem()})}){
  Object.assign(this,{dataRoot,bundledRoot,probe,catalogue,fetcher,memory});this.root=path.join(dataRoot,'model-cache');this.settingsFile=path.join(dataRoot,'model-selection.json');
  this.active={id:catalogue.defaultModel,device:'cpu'};this.available=new Set();this.download=null;this.lastError=null;this.lastProbe=null;this.switching=false;
 }
 entry(id){const item=this.catalogue.models.find(item=>item.id===id);if(!item)throw Error('Bilinmeyen model.');return item;}
 file(item){return path.join(item.bundled?this.bundledRoot:this.root,item.file);}
 limits(id=this.active.id){const item=this.entry(id);return {maxNativePixels:item.architecture==='resnet50'?24000000:64000000,minimumMemoryGiB:item.architecture==='resnet50'?8:4};}
 async safeRoot(){await fs.mkdir(this.root,{recursive:true,mode:0o700});const info=await fs.lstat(this.root);if(info.isSymbolicLink()||!info.isDirectory())throw Error('Model dizini güvenli değil.');}
 async verified(item){try{const stat=await fs.lstat(this.file(item));return stat.isFile()&&!stat.isSymbolicLink()&&stat.nlink===1&&stat.size===item.bytes&&await hashFile(this.file(item))===item.sha256;}catch(error){if(error.code==='ENOENT')return false;throw error;}}
 async initialize(){
  await this.safeRoot();
  for(const entry of await fs.readdir(this.root,{withFileTypes:true})){if(!entry.isFile()||entry.isSymbolicLink())continue;const item=this.catalogue.models.find(item=>entry.name.startsWith(item.file+'.'));if(!item||!/^[a-f0-9-]{36}\.partial$/.test(entry.name.slice(item.file.length+1)))continue;const file=path.join(this.root,entry.name);const stat=await fs.lstat(file);if(Date.now()-stat.mtimeMs>3600000)await fs.unlink(file);}
  for(const item of this.catalogue.models)if(await this.verified(item))this.available.add(item.id);
  if(!this.available.has(this.catalogue.defaultModel))throw Error('Paketlenmiş varsayılan model doğrulanamadı.');
  try{const saved=JSON.parse(await fs.readFile(this.settingsFile,'utf8'));this.entry(saved.id);if(this.available.has(saved.id)&&['cpu','auto'].includes(saved.device))this.active={id:saved.id,device:saved.device};else this.lastError='Seçilen model kullanılamadı; mevcut varsayılan model korunuyor.';}catch(error){if(error.code!=='ENOENT')this.lastError='Model seçimi okunamadı; varsayılan model korunuyor.';}
 }
 snapshot(){const memory=this.memory();return {defaultModel:this.catalogue.defaultModel,active:this.active,hardware:{memoryGiB:Math.round(memory.total/1073741824),freeMemoryGiB:Math.round(memory.free/1073741824),logicalCores:os.availableParallelism?.()||os.cpus().length,platform:process.platform,arch:process.arch},models:this.catalogue.models.map(item=>({...item,maxNativePixels:this.limits(item.id).maxNativePixels,downloaded:this.available.has(item.id),active:item.id===this.active.id,recommended:item.id===this.catalogue.defaultModel,memoryRecommended:memory.total>=item.recommendedMemoryGiB*1073741824,note:item.bundled?'Mevcut model · varsayılan':item.precision==='fp16'?'Daha küçük model. Hız donanıma bağlıdır; etkinleştirmeden önce test edilir.':'Daha ağır model. Daha fazla bellek ve işlem süresi gerekebilir.'})),download:this.download?{...this.download,controller:undefined}:null,switching:this.switching,lastError:this.lastError,lastProbe:this.lastProbe};}
 startDownload(id){
  const item=this.entry(id);if(item.bundled||this.available.has(id))return this.snapshot();
  if(this.download&&['downloading','verifying'].includes(this.download.phase)){if(this.download.id===id)return this.snapshot();throw Error('Başka bir model indiriliyor.');}
  const job={id,phase:'downloading',received:0,total:item.bytes,controller:new AbortController(),startedAt:Date.now()};this.download=job;this.lastError=null;
  this.downloadPromise=this.fetchModel(item,job).catch(error=>{job.phase=job.controller.signal.aborted?'cancelled':'failed';job.error=error.message;this.lastError=error.message;});
  return this.snapshot();
 }
 cancelDownload(){this.download?.controller?.abort();return this.snapshot();}
 async fetchModel(item,job){
  await this.safeRoot();const space=await fs.statfs(this.root);if(Number(space.bavail)*Number(space.bsize)<item.bytes*2+64*1024*1024)throw Error('Model indirmek için yeterli disk alanı yok.');
  let url=UPSTREAM+item.file,response;
  for(let redirects=0;redirects<6;redirects++){
   const target=new URL(url);if(target.protocol!=='https:'||!allowedHosts.has(target.hostname))throw Error('Model indirme adresi güvenilir değil.');
   response=await this.fetcher(url,{redirect:'manual',signal:AbortSignal.any([job.controller.signal,AbortSignal.timeout(900000)]),headers:{'User-Agent':'Navbea-RVM-ModelManager'}});
   if([301,302,303,307,308].includes(response.status)){const location=response.headers.get('location');await response.body?.cancel();if(!location)throw Error('Model indirme yönlendirmesi geçersiz.');url=new URL(location,url).href;continue;}break;
  }
  if(!response?.ok||!response.body){await response?.body?.cancel();throw Error('Model indirilemedi. İnternet bağlantısını kontrol edin.');}
  if(Number(response.headers.get('content-length')||item.bytes)!==item.bytes){await response.body.cancel();throw Error('Model dosyası boyutu beklenenle eşleşmiyor.');}
  const temporary=path.join(this.root,item.file+'.'+crypto.randomUUID()+'.partial');let handle;const reader=response.body.getReader();
  try{
   handle=await fs.open(temporary,'wx',0o600);const hash=crypto.createHash('sha256');
   while(true){let timer;const packet=await Promise.race([reader.read(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('Model indirmesi durdu. İnternet bağlantısını kontrol edip tekrar deneyin.')),30000);})]).finally(()=>clearTimeout(timer));if(packet.done)break;const chunk=packet.value;if(job.controller.signal.aborted)throw Error('İndirme iptal edildi.');job.received+=chunk.length;if(job.received>item.bytes)throw Error('Model boyutu sınırı aşıldı.');hash.update(chunk);let offset=0;while(offset<chunk.length){const result=await handle.write(chunk,offset,chunk.length-offset);if(!result.bytesWritten)throw Error('Model dosyası yazılamadı.');offset+=result.bytesWritten;}}
   job.phase='verifying';await handle.sync();await handle.close();handle=null;
   if(job.received!==item.bytes||hash.digest('hex')!==item.sha256)throw Error('Model bütünlüğü doğrulanamadı. Mevcut model değiştirilmedi.');
   if(job.controller.signal.aborted)throw Error('İndirme iptal edildi.');
   await fs.rename(temporary,this.file(item));this.available.add(item.id);job.phase='ready';job.completedAt=Date.now();
  }finally{await reader.cancel().catch(()=>{});reader.releaseLock();await handle?.close();await fs.unlink(temporary).catch(error=>{if(error.code!=='ENOENT')throw error;});}
 }
 async activate(id){
  if(this.switching)throw Error('Model doğrulaması sürüyor.');const item=this.entry(id);
  if(!this.available.has(id)||!await this.verified(item))throw Error('Model önce indirilmeli ve doğrulanmalı.');
  if(id===this.active.id)return this.snapshot();
  if(!item.bundled){const memory=this.memory();if(memory.total<this.limits(id).minimumMemoryGiB*1073741824||memory.free<1073741824)throw Error('Bu model için yeterli kullanılabilir bellek yok. Mevcut model korunuyor.');}
  this.switching=true;this.lastError=null;
  try{
   const result=await this.probe(this.file(item),item.bundled?'cpu':'auto');
   if(!result?.verified)throw Error('Model çalışma testi geçilemedi.');
   const selection={id,device:item.bundled?'cpu':'auto'};const temporary=this.settingsFile+'.'+crypto.randomUUID()+'.tmp';
   try{await fs.writeFile(temporary,JSON.stringify(selection),{mode:0o600,flag:'wx'});await fs.rename(temporary,this.settingsFile);}finally{await fs.unlink(temporary).catch(error=>{if(error.code!=='ENOENT')throw error;});}
   this.active=selection;this.lastProbe={...result,modelId:id,at:new Date().toISOString()};return this.snapshot();
  }catch(error){this.lastError=error.message;throw error;}finally{this.switching=false;}
 }
 async restore(selection,reason){this.entry(selection.id);this.active=selection;this.lastError=reason;const temporary=this.settingsFile+'.'+crypto.randomUUID()+'.tmp';try{await fs.writeFile(temporary,JSON.stringify(selection),{mode:0o600,flag:'wx'});await fs.rename(temporary,this.settingsFile);}catch(error){this.lastError=reason+' Seçim diske yazılamadı: '+error.code;}finally{await fs.unlink(temporary).catch(()=>{});}}
}
module.exports={ModelManager,hashFile};
