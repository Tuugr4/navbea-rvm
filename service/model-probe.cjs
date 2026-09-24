const {execFile}=require('node:child_process');
function runModelProbe(options,model,device='cpu',signal){
 return new Promise((resolve,reject)=>execFile(options.python,['-I','-B',options.script,'--model',model,'--device',device,'--threads',String(Math.max(1,Math.min(2,require('node:os').availableParallelism?.()||2))),'--still-max-edge',String(options.stillMaxEdge||1024),'--check-model'],{windowsHide:true,timeout:90000,maxBuffer:1024*1024,signal},(error,stdout,stderr)=>{
  if(error)return reject(Error('Model bu çalışma ortamında doğrulanamadı. '+String(stderr||error.message).slice(-600)));
  try{const result=JSON.parse(stdout.trim().split(/\r?\n/).at(-1));if(!result.verified)throw Error('Model doğrulanmadı');resolve(result);}catch{reject(Error('Model testi geçerli sonuç vermedi.'));}
 }));
}
function runStillProbe(options,model,deadlineMs,signal){
 return new Promise((resolve,reject)=>execFile(options.python,['-I','-B',options.script,'--still-model',model,'--still-device',options.stillDevice||'auto','--check-still-model'],{windowsHide:true,timeout:180000,maxBuffer:1024*1024,signal},(error,stdout,stderr)=>{
  if(error)return reject(Error('Fotoğraf modeli bu bilgisayarda çalıştırılamadı. '+String(stderr||error.message).slice(-600)));
  let result;
  try{result=JSON.parse(stdout.trim().split(/\r?\n/).at(-1));if(!result.verified)throw Error('Model doğrulanmadı');}catch{return reject(Error('Fotoğraf modeli testi geçerli sonuç vermedi.'));}
  // Captures wait for the model only up to the deadline; a model that cannot
  // meet it would add the full wait to every photo and still fall back to RVM.
  if(!(result.probeMs<=deadlineMs))return reject(Error(`Bu bilgisayarda maske ${(result.probeMs/1000).toFixed(1)} sn sürdü; ${(deadlineMs/1000).toFixed(0)} sn sınırını aşıyor. Mevcut model korunuyor.`));
  resolve(result);
 }));
}
module.exports={runModelProbe,runStillProbe};
