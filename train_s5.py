"""Train the typed evidence models and OC bin weights after the base system."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(a):
    base=a.base_run.resolve();out=a.out.resolve();cpu=str(a.cpu_python.absolute())
    stages=[('typed-fit',[cpu,'-B',str(out/'s5/train_typed.py'),'--inputs',str(base/'reserve_inputs/rank'),
              '--cache',str(base/'current/cache'),'--out',str(out/'typed_fits')]),
            ('install-typed',None),
            ('typed',[cpu,'-B',str(out/'s5/typed_inference.py'),'--tables','all','--jobs',str(a.jobs)]),
            ('oc-statistics',[cpu,'-B',str(out/'s5/rebuild_oc_statistics.py'),'--cache',str(base/'current/cache')]),
            ('oc-base',[cpu,'-B',str(out/'s5/materialize.py'),'baseline'])]
    for name in ['base','detectors','zdep','finish']:
        stages.append(('oc-route-'+name,[cpu,'-B',str(out/'s5/oc_routes.py'),name,'--cache',str(base/'current/cache'),
                                      '--pair-baseline',str(out/'s5/pair_baseline.npz')]))
    stages += [('oc-weight-fit',[cpu,'-B',str(out/'s5/calibrate_oc.py'),'--cache',str(base/'current/cache'),'--newly-trained']),
               ('install-oc',None),
               ('oc-hands',[cpu,'-B',str(out/'s5/oc_inference.py'),'--cache',str(base/'current/cache'),
                            '--pairs',str(out/'s5/oc_routes/pairs_oc_block.csv')]),
               ('assemble',[cpu,'-B',str(out/'s5/materialize.py'),'assemble','--newly-trained']),
               ('validate',[cpu,'-B',str(base/'current/src/validate_submission.py'),str(out/'s5/result/S5_J50_risk.csv'),
                            '--data',str(a.data.resolve()),'--out',str(out/'s5/result/independent_validation.json')])]
    if a.plan:
        print(json.dumps(dict(stages=[dict(stage=s,command=c) for s,c in stages],scope='S5 training plan'),indent=2));return
    assert base!=out and not out.is_relative_to(base) and not base.is_relative_to(out)
    complete=json.loads((base/'CURRENT_TRAINING_COMPLETE.json').read_text());assert complete['complete']
    assert sha(base/'current/result/Reserve_J50_risk.csv')==complete['csv_sha256']
    assert not out.exists();out.mkdir(parents=True)
    # The link reads this task's own completed training runtime. S5 writes only
    # below its separate output directory; no base fits are replaced.
    (out/'package').symlink_to(base/'current',target_is_directory=True)
    s5=out/'s5';s5.mkdir()
    for p in (HERE/'s5').glob('*.py'):shutil.copy2(p,s5/p.name)
    shutil.copytree(HERE/'s5/assets',s5/'assets')
    source=s5/'source';source.mkdir()
    for name in ['tdlib.py','oc_evidence_final.py']:shutil.copy2(HERE/'s5/source'/name,source/name)
    for folder in ['accumulators','route_original']:
        shutil.copytree(HERE/'s5/source'/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__'))
    records=[];started=time.time();logs=out/'logs';logs.mkdir()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NPY_DISABLE_CPU_FEATURES='X86_V4')
    for name,command in stages:
        t=time.time();print('S5 TRAIN',name,flush=True)
        if name=='install-typed':
            report=json.loads((out/'typed_fits/MODEL_CHECK.json').read_text());assert len(report['models'])==18
            for model in report['models']:
                p=out/'typed_fits'/model['file'];assert sha(p)==model['sha256'];shutil.copy2(p,s5/'assets/typed'/p.name)
        elif name=='install-oc':shutil.copy2(s5/'oc_calibration/calib_norho.json',s5/'assets/calib_norho.json')
        else:
            with (logs/(name+'.log')).open('w') as f:subprocess.run(command,cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        records.append(dict(stage=name,seconds=time.time()-t,command=command))
        (out/'TRAINING_PROGRESS.json').write_text(json.dumps(records,indent=2)+'\n')
    report=dict(complete=True,seconds=time.time()-started,base_completion_sha256=sha(base/'CURRENT_TRAINING_COMPLETE.json'),
                csv_sha256=sha(s5/'result/S5_J50_risk.csv'),stages=records,submission_performed=False)
    (out/'S5_TRAINING_COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--cpu-python',type=Path,default=Path(sys.executable))
    p.add_argument('--jobs',type=int,default=3);p.add_argument('--plan',action='store_true');a=p.parse_args()
    assert 1<=a.jobs<=3;main(a)
