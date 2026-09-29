const fs=require('node:fs/promises');const path=require('node:path');const http=require('node:http');const {execFile}=require('node:child_process');
const catalogs={live:require('../models/catalog.json'),still:require('../models/still-catalog.json')};
const ids={...Object.fromEntries(Object.entries(catalogs).map(([kind,catalog])=>[kind,new Set(catalog.models.map(item=>item.id))])),device:new Set(['auto','gpu','cpu'])};
const prefixes={live:'/v1/models',still:'/v1/still-models',device:'/v1/device'};
function check(action,id,kind='live'){if(!ids[kind]||!['download','activate','cancel'].includes(action)||(kind==='device'&&action!=='activate')||(action==='cancel'?![undefined,null,''].includes(id):!ids[kind].has(id)))throw Error('Geçersiz model işlemi.');}
async function request({dataRoot,control,role='client'},pathname,body,timeoutMs){
 const token=(await fs.readFile(path.join(dataRoot,role+'-token.txt'),'utf8')).trim();
 return new Promise((resolve,reject)=>{const payload=body?Buffer.from(JSON.stringify(body)):null;const req=http.request({socketPath:control,path:pathname,method:payload?'POST':'GET',headers:{['x-navbea-'+role+'-token']:token,...(payload?{'content-type':'application/json','content-length':payload.length}:{})},timeout:timeoutMs||(payload?210000:10000)},res=>{const chunks=[];let size=0;res.on('data',chunk=>{size+=chunk.length;if(size>1024*1024)req.destroy(Error('Model yanıtı çok büyük.'));else chunks.push(chunk);});res.on('end',()=>{try{const value=JSON.parse(Buffer.concat(chunks).toString());if(res.statusCode>=400)reject(Error(value.error||'Model işlemi başarısız.'));else resolve(value);}catch(error){reject(error);}});});req.on('error',reject);req.on('timeout',()=>req.destroy(Error('Model işlemi zaman aşımına uğradı.')));if(payload)req.write(payload);req.end();});
}
async function administer(options,action,id,kind='live'){
 check(action,id,kind);
 if(options.platform==='linux')return new Promise((resolve,reject)=>execFile('/usr/bin/pkexec',['/usr/lib/navbea-rvm/resources/deploy/linux/model-admin',kind,action,id||''],{timeout:220000,maxBuffer:1024*1024},(error,out,stderr)=>{if(error)return reject(Error(stderr.trim()||'Yönetici izni verilmedi.'));try{const value=JSON.parse(out);if(value.error)reject(Error(value.error));else resolve(value);}catch{reject(Error('Model yönetimi yanıtı okunamadı.'));}}));
 try{return await request({...options,role:'admin'},prefixes[kind]+'/'+action,{id});}
 catch(error){
  if(options.platform!=='win32'||!['EACCES','EPERM'].includes(error.code))throw error;
  const executable="'"+options.executable.replaceAll("'","''")+"'";
  // Start-Process drops empty arguments, so the optional id goes last.
  const code=`try { $p=Start-Process -FilePath ${executable} -ArgumentList '--model-admin','${kind}','${action}','${id||''}' -Verb RunAs -PassThru -Wait; exit $p.ExitCode } catch { exit 1 }`;
  const environment=Object.fromEntries(Object.entries(process.env).filter(([key])=>!/^(?:NAVBEA_|RVM_|PYTHON|NODE_OPTIONS$|NODE_PATH$|ELECTRON_RUN_AS_NODE$)/i.test(key)));
  await new Promise((resolve,reject)=>execFile(path.join(options.systemRoot,'System32/WindowsPowerShell/v1.0/powershell.exe'),['-NoProfile','-NonInteractive','-EncodedCommand',Buffer.from(code,'utf16le').toString('base64')],{windowsHide:true,timeout:240000,env:environment},error=>error?reject(Error('Model yönetimi tamamlanamadı. Yönetici iznini ve servis durumunu kontrol edin.')):resolve()));
  return request(options,prefixes[kind]);
 }
}
function adminArguments(values){const [first,...rest]=values;return Object.hasOwn(prefixes,first)?{kind:first,action:rest[0],id:rest[1]}:{kind:'live',action:first,id:rest[0]};}
module.exports={request,administer,check,prefixes,adminArguments};
