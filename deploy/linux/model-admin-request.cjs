const {request,check}=require('../../service/model-control-client.cjs');
const [action,id]=process.argv.slice(2);
(async()=>{if(process.getuid?.()!==0)throw Error('Administrator required');check(action,id);const protocol=require('../../local-media-protocol/index.cjs');const result=await request({dataRoot:'/var/lib/navbea/rvm',control:protocol.endpointPaths('rvm','linux',{NAVBEA_RUNTIME_DIR:'/run/navbea'}).control,role:'admin'},'/v1/models/'+action,{id});process.stdout.write(JSON.stringify(result)+'\n');})().catch(error=>{process.stderr.write(error.message+'\n');process.exitCode=1;});
