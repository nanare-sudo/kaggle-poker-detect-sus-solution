"""Full S5/J50 and Reserve/J50 inference, offline; raw inputs plus model weights.

The shared hand computation produces both evidence variants. No legacy model
outputs, selected-pair lists or submission CSVs are inputs. --plan is read-only.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(a):
    root=HERE/'package';s5=HERE/'s5';cpu=str(a.cpu_python.absolute());gpu=str(a.gpu_python.absolute())
    core=[cpu,'-B',str(root/'run.py'),'--data',str(a.data.resolve()),'--cpu-python',cpu,'--gpu-python',gpu,'--jobs',str(a.jobs)]
    stages=[(s,core+['--stage',s]) for s in ['core','math','policy','neural','features','contexts','tabicl-inputs','tabicl','select']]
    stages += [('typed',[cpu,'-B',str(s5/'typed_inference.py'),'--tables','all','--jobs',str(min(a.jobs,3))]),
               ('reserve-picks',[cpu,'-B',str(s5/'materialize.py'),'reserve']),
               ('reserve-csv',core+['--stage','assemble']),('reserve-valid',core+['--stage','validate']),
               ('oc-statistics',[cpu,'-B',str(s5/'rebuild_oc_statistics.py'),'--cache',str(root/'cache')]),
               ('oc-base',[cpu,'-B',str(s5/'materialize.py'),'baseline'])]
    for s in ['base','detectors','zdep','finish']:
        stages.append(('oc-route-'+s,[cpu,'-B',str(s5/'oc_routes.py'),s,'--cache',str(root/'cache'),
                                     '--pair-baseline',str(s5/'pair_baseline.npz')]))
    stages += [('oc-hands',[cpu,'-B',str(s5/'oc_inference.py'),'--cache',str(root/'cache'),'--pairs',str(s5/'oc_routes/pairs_oc_block.csv')]),
               ('s5-csv',[cpu,'-B',str(s5/'materialize.py'),'assemble']),
               ('s5-valid',[cpu,'-B',str(root/'src/validate_submission.py'),str(s5/'result/S5_J50_risk.csv'),
                            '--data',str(a.data.resolve()),'--out',str(s5/'result/independent_validation.json')])]
    stages.append(('selected-reference',[cpu,'-B',str(HERE/'export_selected_reference.py')]))
    if a.plan:
        print(json.dumps(dict(stages=[dict(stage=s,command=c) for s,c in stages],submits=False),indent=2));return
    assert not any(p.is_symlink() for p in (root/'cache').glob('*')),'Refuse a read-only shared cache workspace'
    manifest=HERE/'SOLUTION_MANIFEST.json';spec=json.loads(manifest.read_text())
    for name,digest in spec['files'].items():assert sha(HERE/name)==digest,name
    fingerprint=sha(manifest);receipts=HERE/'solution_receipts';receipts.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for stage,cmd in stages:
        receipt=receipts/(stage+'.json')
        if receipt.exists():assert json.loads(receipt.read_text())['manifest_sha256']==fingerprint;continue
        started=time.time();print('SOLUTION STAGE',stage,flush=True)
        subprocess.run(cmd,cwd=HERE,env=env,check=True)
        receipt.write_text(json.dumps(dict(stage=stage,command=cmd,seconds=time.time()-started,manifest_sha256=fingerprint),indent=2)+'\n')
    for file,expected in [(root/'result/Reserve_J50_risk.csv','0497940682d7efce936ac16d186621cb63c45c3053b679b39ead07482dbd9316'),
                          (s5/'result/S5_J50_risk.csv','cb25c90604e5c5bc8be323c9b3a83fb12f153f8b69775191467446ff847ffd8b')]:
        assert sha(file)==expected,(str(file),sha(file))
    print('Output CSV hashes match the released predictions.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--cpu-python',type=Path,default=Path(sys.executable))
    p.add_argument('--gpu-python',type=Path,default=Path(sys.executable));p.add_argument('--jobs',type=int,default=3);p.add_argument('--plan',action='store_true')
    a=p.parse_args();assert 1<=a.jobs<=4;main(a)
