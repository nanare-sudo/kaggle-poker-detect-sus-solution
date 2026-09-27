"""Train the action references, history models, P18n, behavior and Reserve evidence models."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time

HERE=Path(__file__).resolve().parent
BASE=['core','math','stats','train-classic','train-q2','install-policies','policy',
      'train-neural','install-neural','neural','pair-inputs','pair-heads',
      'instance-inputs','instance-final','instance-teachers','pool-inputs','pool-heads',
      'evidence-inputs','evidence-heads','assemble-assets']
TAIL=['action-current','init-current','pair-views','original-heads','rclean-heads',
      'reserve-views','reserve-matrices','reserve-heads','install-current',
      'features','contexts','tabicl-inputs','tabicl','select','reserve','assemble','validate']
STAGES=['base:'+s for s in BASE]+TAIL

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def initialize(root,data):
    current=root/'current';assert not current.exists()
    package=HERE/'package'
    spec=json.loads((package/'MANIFEST.json').read_text())
    for name, expected in spec['files'].items():
        relative=Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        source=package/relative
        assert sha(source)==expected, name
        target=current/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    shutil.copy2(package/'MANIFEST.json',current/'MANIFEST.json')
    legacy=root/'legacy'
    assert not (current/'cache').exists();(current/'cache').mkdir()
    for p in (legacy/'cache').iterdir():
        if p.is_file():(current/'cache'/p.name).symlink_to(p.resolve())
    # Install weights fitted in the preceding training stages.
    for p in (legacy/'assets').rglob('*'):
        if p.is_file():
            target=current/'assets'/p.relative_to(legacy/'assets');target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
    frozen=json.loads((HERE/'package/assets/recipe.json').read_text())
    recipe=json.loads((current/'assets/recipe.json').read_text())
    assert recipe['tabicl_features']==frozen['tabicl_features']
    recipe['tabicl_features192']=frozen['tabicl_features192']
    (current/'assets/recipe.json').write_text(json.dumps(recipe,indent=2))
    for f in range(2):shutil.copy2(root/f'action_current/RCLEAN/model_noply_f{f}.pt',current/f'assets/neural/model_rclean_f{f}.pt')
    for name in ['surprise','paggr','pfold']:
        shutil.copy2(root/f'action_current/RCLEAN/act_{name}_q2nnnoply.npy',current/f'cache/act_{name}_rclean.npy')
    shutil.copy2(data/'sample_submission.csv',current/'raw_sample_submission.csv')

def install(root):
    current=root/'current';assets=current/'assets'
    for arm,target in [('original','bothshort'),('RCLEAN','rclean')]:
        report=json.loads((root/f'head_fits/{arm}/REPRODUCTION.json').read_text())
        assert [r['seed'] for r in report['results']]==list(range(42,47))
        for seed in range(42,47):shutil.copy2(root/f'head_fits/{arm}/seed{seed}.txt',assets/f'{target}/seed{seed}.txt')
    fitted=root/'reserve_fits';manifest=json.loads((fitted/'model_manifest.json').read_text())
    assert len(list(fitted.glob('CLASSIC_inner*.txt')))==5 and len(list(fitted.glob('NN_EVENTS_inner*.txt')))==5
    assert not json.loads((fitted/'specification.json').read_text())['smoke']
    for name,expected in manifest.items():
        assert sha(fitted/name)==expected
        shutil.copy2(fitted/name,assets/'reserve'/name)
    learned={str(p.relative_to(assets)):sha(p) for p in assets.rglob('*') if p.is_file()}
    (root/'CURRENT_FITTED_ASSETS.json').write_text(json.dumps(learned,indent=2))

def commands(a):
    root=a.out.resolve();legacy=root/'legacy';current=root/'current';train=root/'reserve_inputs'
    cpu=str(a.cpu_python.absolute());gpu=str(a.gpu_python.absolute());data=a.data.resolve()
    cmd={}
    for stage in BASE:
        cmd['base:'+stage]=[cpu,'-B',str(HERE/'training_base/train.py'),'--data',str(data),'--out',str(legacy),
            '--cpu-python',cpu,'--gpu-python',gpu,'--jobs',str(a.jobs),'--stage',stage]
    cmd['action-current']=[gpu,'-B',str(HERE/'train_action_current.py'),'--coordinate','--root',str(legacy),'--out',str(root/'action_current')]
    cmd['init-current']=None
    common=['--root',str(current)]
    cmd['pair-views']=[cpu,'-B',str(HERE/'build_current_pair_views.py'),*common,'--out',str(root/'pair_views'),'--tables','all','--jobs',str(a.jobs)]
    for arm,stage in [('original','original-heads'),('RCLEAN','rclean-heads')]:
        cmd[stage]=[cpu,'-B',str(HERE/'fit_current_heads.py'),arm,*common,'--inputs',str(root/'pair_views'),'--out',str(root/f'head_fits/{arm}')]
    cmd['reserve-views']=[cpu,'-B',str(HERE/'build_reserve_training.py'),'build',*common,'--out',str(train),'--tables','all','--jobs',str(a.jobs)]
    cmd['reserve-matrices']=[cpu,'-B',str(HERE/'build_reserve_training.py'),'collect',*common,'--out',str(train)]
    cmd['reserve-heads']=[cpu,'-B',str(HERE/'train_reserve.py'),'--nested',str(train/'nested'),'--rankdata',str(train/'rank'),'--windowdata',str(train/'window'),'--out',str(root/'reserve_fits')]
    cmd['install-current']=None
    cmd['features']=[cpu,'-B',str(current/'pipeline/current_pairs.py'),'--tables','all','--jobs',str(a.jobs)]
    cmd['contexts']=[cpu,'-B',str(current/'src/tabicl_contexts.py'),'--tables','400','--jobs',str(a.jobs)]
    for stage,operation in [('tabicl-inputs','prepare'),('select','select'),('assemble','assemble')]:
        cmd[stage]=[cpu,'-B',str(current/'pipeline/current_finish.py'),operation,*common]
    cmd['tabicl']=[gpu,'-B',str(current/'pipeline/tabicl_pairs.py'),*common]
    cmd['reserve']=[cpu,'-B',str(current/'pipeline/current_reserve.py'),'--tables','all','--jobs',str(a.jobs)]
    cmd['validate']=[cpu,'-B',str(current/'src/validate_submission.py'),str(current/'result/Reserve_J50_risk.csv'),'--data',str(data),
        '--out',str(current/'result/independent_validation.json')]
    assert set(cmd)==set(STAGES);return cmd

def main(a):
    cmd=commands(a)
    source={p.name:sha(p) for p in [Path(__file__),HERE/'build_current_pair_views.py',HERE/'fit_current_heads.py',
        HERE/'build_reserve_training.py',HERE/'train_reserve.py',HERE/'train_action_current.py']}
    source['training_base_manifest']=sha(HERE/'training_base/TRAINING_MANIFEST.json')
    source['inference_manifest']=sha(HERE/'package/MANIFEST.json')
    fingerprint=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    if a.plan:
        print(json.dumps(dict(stages=STAGES,commands=cmd,source=source,source_sha256=fingerprint,
            scope='Training plan'),indent=2));return
    root=a.out.resolve();assert root!=HERE and not HERE.is_relative_to(root)
    root.mkdir(parents=True,exist_ok=True);marker=root/'CURRENT_TRAINING_ROOT.json'
    if marker.exists():assert json.loads(marker.read_text())==dict(source_sha256=fingerprint,data=str(a.data.resolve()))
    else:
        assert not list(root.iterdir());marker.write_text(json.dumps(dict(source_sha256=fingerprint,data=str(a.data.resolve())),indent=2))
    receipts=root/'current_receipts';receipts.mkdir(exist_ok=True);logs=root/'current_logs';logs.mkdir(exist_ok=True)
    for stage in STAGES if a.stage=='all' else [a.stage]:
        path=receipts/(stage.replace(':','_')+'.json')
        if path.exists():assert json.loads(path.read_text())['source_sha256']==fingerprint;continue
        for previous in STAGES[:STAGES.index(stage)]:assert (receipts/(previous.replace(':','_')+'.json')).exists()
        started=time.time();print('CURRENT TRAINING START',stage,flush=True)
        if stage=='action-current':(root/'action_current').mkdir()
        if stage=='init-current':initialize(root,a.data.resolve())
        elif stage=='install-current':install(root)
        else:
            env=dict(os.environ);env.pop('PYTHONPATH',None)
            if stage in ['action-current','tabicl']:
                env.pop('NPY_DISABLE_CPU_FEATURES',None);env.pop('POKER_CHEN_ORDER_FILE',None)
            elif not stage.startswith('base:'):
                env['NPY_DISABLE_CPU_FEATURES']='X86_V4';env['POKER_CHEN_ORDER_FILE']=str(root/'current/pipeline/chen_order_cpu.json')
            with (logs/(stage.replace(':','_')+'.log')).open('w') as log:
                subprocess.run(cmd[stage],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        path.write_text(json.dumps(dict(stage=stage,command=cmd[stage],source_sha256=fingerprint,seconds=time.time()-started),indent=2))
        print('CURRENT TRAINING COMPLETE',stage,round(time.time()-started),flush=True)
    if all((receipts/(s.replace(':','_')+'.json')).exists() for s in STAGES):
        report=dict(complete=True,source_sha256=fingerprint,csv_sha256=sha(root/'current/result/Reserve_J50_risk.csv'),
            byte_identity_with_retained_submission_not_assumed=True,scope='Fresh current recipe, stochastic neural training, no competition submission')
        (root/'CURRENT_TRAINING_COMPLETE.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cpu-python',type=Path,default=Path(sys.executable));p.add_argument('--gpu-python',type=Path,default=Path(sys.executable))
    p.add_argument('--jobs',type=int,default=3);p.add_argument('--stage',choices=['all']+STAGES,default='all');p.add_argument('--plan',action='store_true')
    a=p.parse_args();assert 1<=a.jobs<=3;main(a)
