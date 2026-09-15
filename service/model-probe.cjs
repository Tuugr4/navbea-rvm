const {execFile}=require('node:child_process');
function runModelProbe(options,model,device='cpu',signal){
 return new Promise((resolve,reject)=>execFile(options.python,['-I','-B',options.script,'--model',model,'--device',device,'--threads',String(Math.max(1,Math.min(2,require('node:os').availableParallelism?.()||2))),'--still-max-edge',String(options.stillMaxEdge||1024),'--check-model'],{windowsHide:true,timeout:90000,maxBuffer:1024*1024,signal},(error,stdout,stderr)=>{
  if(error)return reject(Error('Model bu çalışma ortamında doğrulanamadı. '+String(stderr||error.message).slice(-600)));
  try{const result=JSON.parse(stdout.trim().split(/\r?\n/).at(-1));if(!result.verified)throw Error('Model doğrulanmadı');resolve(result);}catch{reject(Error('Model testi geçerli sonuç vermedi.'));}
 }));
}
module.exports={runModelProbe};
