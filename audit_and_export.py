"""Audit the public article against the local analysis source, without refitting.

Usage: python3 audit_and_export.py /path/to/AuditoryCategorizationSwat
Requires numpy, scipy, pandas, matplotlib and Node.js. Original analysis code stays untouched.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]).resolve()
WEB = ROOT / 'docs/auditory_categorization_learning'
(ROOT / 'audit').mkdir(exist_ok=True)
sys.path.insert(0, str(SOURCE))
os.chdir(SOURCE)
import reinforcementLearningFns as rlf
import clusteringFns as clf
import helperFns as helper
import plotFns as plotting

data = json.loads((WEB / 'results.json').read_text())
checks = []
sources = set()

def check(name, actual, expected, tolerance=1e-12):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b)), name
    error = float(np.nanmax(np.abs(a-b))) if np.any(np.isfinite(a)) else 0.0
    assert error <= tolerance, (name, error)
    checks.append({'check': name, 'max_abs_error': error, 'tolerance': tolerance, 'values': int(a.size)})

def read_csv(name):
    sources.add(name)
    return list(csv.DictReader((SOURCE/name).open()))

def mouse(mouse_id):
    name=f'data/Trajectories/with_bias_learning/{mouse_id}_trainingDataBias.pkl'
    sources.add(name)
    with (SOURCE/name).open('rb') as f:return pickle.load(f)

def sessions(d):
    result=[];start=0
    for size in d['dayLength']:
        end=start+int(size);correct=np.asarray(d['correct'])[start:end];cat=np.asarray(d['answer'])[start:end]
        result.append({'end':end,'n':int(size),'accuracy':float(correct.mean()),'low':float(correct[cat==1].mean()),'high':float(correct[cat==2].mean())})
        start=end
    assert start==len(d['correct'])
    return result

for mid, entry in data['mice'].items():
    d=mouse(mid);acc=np.asarray(d['correct']);cat=np.asarray(d['answer']);raw_y=np.asarray(d['y'])
    choice=(raw_y==raw_y.max()).astype(int)
    check(mid+' normalized choices agree with recorded correctness',choice==(cat-1),acc,0)
    check(mid+' published trial count',entry['n'],len(acc),0)
    for i,b in enumerate(entry['bins']):
        start=i*200;end=min(start+200,len(acc));a=acc[start:end];c=cat[start:end]
        check(mid+f' original website block {i}',[b['accuracy'],b['low'],b['high']],[a.mean(),a[c==1].mean(),a[c==2].mean()],5.000001e-6)
    exact=sessions(d)
    check(mid+' saved session means',[s['accuracy'] for s in entry['sessions']],[s['accuracy'] for s in exact],5.000001e-6)
    entry['sessions']=exact

# Figure1.ipynb specifies these seven animals, cutoff=60, untilTesting=True.
data['paperMice']={}
for mid in ['GS027','GS037','JC028','JC039','JC044','JC052','JC061']:
    raw=helper.getD(mid,keyword='training',cutoff=60,sessionCutoff=None,txt=None,untilTesting=True,basePath=str(SOURCE/'data/MouseData'))
    grouped=pd.DataFrame({'session':raw['session'],'correct':raw['correct']}).groupby('session')['correct'].mean().to_numpy()
    ss=sessions(raw);check(mid+' Figure1 session aggregation',[s['accuracy'] for s in ss],grouped)
    data['paperMice'][mid]={'n':len(raw['correct']),'sessions':ss}

static=read_csv('thesis/static_glm_history_model_comparison.csv')
dynamic=read_csv('thesis/psytrack_history_model_comparison.csv')
for row in data['history']:
    sr={r['model']:float(r['held_out_log_loss']) for r in static if r['mouse']==row['mouse']}
    dr=next(r for r in dynamic if r['mouse']==row['mouse'])
    check(row['mouse']+' static history gains',row['static'],[sr['Stimulus + bias']-sr[k] for k in ['+ previous choice','+ WSLS','+ previous choice + WSLS']])
    check(row['mouse']+' dynamic history gains',row['dynamic'],[float(dr[k]) for k in ['gain_choice_vs_base','gain_wsls_vs_base','gain_full_vs_base']])
comparison=next(r for r in read_csv('thesis/psytrack_all_old_mice_comparison.csv') if r['mouse']=='GS029')
hmm=read_csv('thesis/glmhmm_one_weight_GS029_comparison.csv')
check('GS029 saved cross-validation scores',[data['comparison'][k] for k in ['static','dynamic','hmm']],[float(comparison['static_one_cv_log_loss']),float(comparison['psytrack_one_cv_log_loss']),float(next(r for r in hmm if r['K']=='3')['held_out_log_loss'])])
for a,b in zip(data['comparison']['states'],[r for r in read_csv('thesis/glmhmm_one_weight_all_old_mice_K3_states.csv') if r['mouse']=='GS029']):
    check('GS029 full-data state '+str(a['state']),list(a.values()),[float(b[k]) for k in a])

d=mouse('GS027');cues=np.asarray(d['answer'])-1;choices=np.asarray(d['y'])-1;params=data['params'];N=len(cues)
fit_rows=read_csv('RL_Data/260802_RL_Data/fits_4param_260802.csv')
fit=min((r for r in fit_rows if r['index']=='GS027'),key=lambda r:float(r['nLL']))
check('GS027 2026 parameter selection',[params[k] for k in params],[float(fit[k]) for k in params],0)
actual_nll=rlf.fitLearning([params[k] for k in ['bias','alpha','beta','init_Q']],{},choices,cues)
check('GS027 likelihood, original fitLearning',actual_nll,float(fit['nLL']),1e-9)
check('GS027 exact public cues',[int(x) for x in data['simulationData']['cue']],cues,0)
check('GS027 exact public choices',[int(x) for x in data['simulationData']['choice']],choices,0)

def mp(p,n):return {'N':n,'bias':p['bias'],'alpha':p['alpha'],'beta':p['beta'],'init_Q':[p['init_Q'],-p['init_Q']],'max_num_cts':3}
stream=random.Random(2021)
data['randomStream']=[stream.random() for _ in range(2*N)]
data['simulationSeed']=2021
data['validSimulationTrials']=N-1
random.seed(2021)
stim,ch,reward,q,ph=rlf.simulateLearning(mp(params,N),stimCat=cues)
reference={'choice':ch[:-1].tolist(),'reward':reward[:-1].tolist(),'q':q[:,:-1].T.tolist(),'p':ph[:-1].tolist()}

# Run the actual browser module, not a separate hand-written implementation.
variants=[('default',params),('fast learning',{**params,'alpha':.01}),('positive bias',{**params,'bias':.2}),('negative initial value',{**params,'init_Q':-.25}),('no cross update',{**params,'beta':1000})]
runner="const m=require(process.argv[1]);let s='';process.stdin.on('data',x=>s+=x);process.stdin.on('end',()=>{const d=JSON.parse(s),r=m.simulate(d.cues,d.params,d.stream),c=m.categoryCurves(d.cues,r.map(x=>x.choice));process.stdout.write(JSON.stringify({choice:r.map(x=>x.choice),reward:r.map(x=>x.reward),q:r.map(x=>x.after),p:r.map(x=>x.p),low:c.low,high:c.high}));});"
for name,p in variants:
    random.seed(2021);st,cc,rr,qq,pp=rlf.simulateLearning(mp(p,N),stimCat=cues)
    spec={'cues':data['simulationData']['cue'],'params':{'alpha':p['alpha'],'bias':p['bias'],'init':p['init_Q'],'coupling':math.exp(-p['beta'])},'stream':data['randomStream']}
    js=json.loads(subprocess.check_output(['node','-e',runner,str(WEB/'model.js')],input=json.dumps(spec).encode()))
    for k,expected in [('choice',cc[:-1]),('reward',rr[:-1]),('q',qq[:,:-1].T),('p',pp[:-1])]:check('browser vs original '+name+' '+k,js[k],expected,0 if k in ('choice','reward') else 1e-12)
    hh,ll,bh,bl,*_=rlf.generateLearningCurves(st[:-1],cc[:-1],pp[:-1])
    check('browser vs original '+name+' low rolling curve',[np.nan if x is None else x for x in js['low']],bl)
    check('browser vs original '+name+' high rolling curve',[np.nan if x is None else x for x in js['high']],bh)

# Quantify the old browser's divergent stochastic trajectory against the same source seed.
q0=np.array([params['init_Q'],-params['init_Q']]);seed=29;legacy=[]
for c in cues:
    p=1/(1+math.exp(-5*((q0[1] if c else -q0[0])+2*params['bias'])));seed=(1664525*seed+1013904223)&0xffffffff;a=int(seed/4294967296<p);r=int(a==c);legacy.append(a);delta=r-q0[a];q0[a]+=params['alpha']*delta;q0[1-a]-=math.exp(-params['beta'])*params['alpha']*delta
changed=int(np.sum(np.array(legacy[:-1])!=np.array(reference['choice'])))

# Recompute Figure 3 diagnostics with the actual simulator, not NumPy's alternate RNG.
random.seed(2021);rollouts=[]
for _ in range(50):rollouts.append(rlf.simulateLearning(mp(params,N),stimCat=cues)[2][:-1])
rollouts=np.asarray(rollouts);q0=np.array([params['init_Q'],-params['init_Q']]);conditional=[]
for c,a in zip(cues,choices):
    p=1/(1+math.exp(-5*((q0[1] if c else -q0[0])+2*params['bias'])));conditional.append(p if c else 1-p);delta=int(c==a)-q0[int(a)];q0[int(a)]+=params['alpha']*delta;q0[1-int(a)]-=math.exp(-params['beta'])*params['alpha']*delta
diagnostic=[]
for i in range(0,N-1,200):
    j=min(i+200,N-1);means=rollouts[:,i:j].mean(axis=1)
    diagnostic.append({'trial':j,'observed':float(np.mean(choices[i:j]==cues[i:j])),'conditional':float(np.mean(conditional[i:j])),'simulation':float(means.mean()),'lo':float(np.quantile(means,.1)),'hi':float(np.quantile(means,.9))})
data['diagnostic']={'mouse':'GS027','nll':actual_nll,'runs':50,'seed':2021,'bins':diagnostic}

# Source-notebook reference: preserve the published plotting pipeline exactly.
# The low trace is 1 - low accuracy, i.e. P(high response | low stimulus).
data['referenceTraces']={}
for mid,num,fit_file in [('GS027',4,'RL_Data/260802_RL_Data/fits_4param_260802.csv'),('JC025',3,'data/RL_Data/fits_3param_081123.csv'),('JC059',4,'data/RL_Data/fits_4param_231023.csv')]:
    dd=mouse(mid);cu=np.asarray(dd['answer'],dtype=float)-1.;n=len(cu);fr=next(r for r in read_csv(fit_file) if r['index']==mid and r['']=='0');p={k:float(fr[k]) for k in ['bias','alpha','init_Q']};p['beta']=float(fr['beta']) if num==4 else 1000
    acc=np.asarray(dd['correct']);lo=np.where(cu==0,acc,np.nan);hi=np.where(cu==1,acc,np.nan)
    random.seed(2021);los=[];his=[]
    for _ in range(50):
        st,cc,rr,qq,pp=rlf.simulateLearning(mp(p,n),stimCat=cu)
        los.append(np.where(cu==0,1-cc,np.nan));his.append(np.where(cu==1,cc,np.nan))
    los=np.array(los);his=np.array(his)
    raw=clf.smoothLearningTraces(lo,hi,nPoints=400,smoothF=5,smooth2=5)[1]
    mean=clf.smoothLearningTraces(los.mean(axis=0),his.mean(axis=0),nPoints=400,smoothF=5,smooth2=5)[1]
    sem=clf.smoothLearningTraces(los.std(axis=0)/np.sqrt(50),his.std(axis=0)/np.sqrt(50),nPoints=400,smoothF=5,smooth2=5)[1];sem[:400]=1-sem[:400]
    def clean(xs):return [float(v) if np.isfinite(v) else None for v in xs]
    data['referenceTraces'][mid]={'n':n,'parameters':p,'fitFile':fit_file,'runs':50,'seed':2021,'observedLow':clean(raw[:400]),'observedHigh':clean(raw[400:800]),'simulatedLow':clean(mean[:400]),'simulatedHigh':clean(mean[400:800]),'semLow':clean(sem[:400]),'semHigh':clean(sem[400:800])}
    original_fig,original_ax=plotting.plotFitSimulations({'sim_low':los,'sim_high':his,'acc_low':lo,'acc_high':hi},50,400,5,5)
    for line,key in zip(original_ax.lines[:4],['observedLow','observedHigh','simulatedLow','simulatedHigh']):
        check(mid+' exported trace vs actual plotFitSimulations '+key,[np.nan if x is None else x for x in data['referenceTraces'][mid][key]],line.get_ydata(),0)
    original_fig.savefig(ROOT/'audit'/f'original-{mid}.png',dpi=180,bbox_inches='tight')
    plt.close(original_fig)

sources.update(['reinforcementLearningFns.py','clusteringFns.py','plotFns.py','Figure1.ipynb','Figure4.ipynb','helperFns.py'])
data['sources']=[{'path':s,'sha256':hashlib.sha256((SOURCE/s).read_bytes()).hexdigest()} for s in sorted(sources)]
data['audit']={'date':'2026-09-29','checks':len(checks),'originalSourceCommit':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),'sourceWorkingTreeHashes':True,'simulationSeed':2021,'validSimulationTrials':N-1,'historicalPaperRandomStateAvailable':False}
(WEB/'results.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
(ROOT/'audit/reference-run.json').write_text(json.dumps(reference,separators=(',',':')))
report={'checks':checks,'initial_findings':{'old_browser_choices_different_from_seeded_source':changed,'trials_compared':N-1,'GS027_analysis_trials':N,'GS027_Figure1_trials':data['paperMice']['GS027']['n'],'old_nll_matches_original':True,'historical_notebook_only_seeds_numpy_not_python_random':True,'paper_fig4_example_mice':['JC025','JC059'],'current_demo_mouse':'GS027'},'limitations':['A newly seeded rerun is reproducible; the historical paper random realizations cannot be guaranteed without their Python random state.','CV values and history gains were checked against saved CSVs; the model fits were not rerun.','The reference smoother outputs high-response probability for both stimulus categories, despite an accuracy ylabel in plotFns.py.','Figure1.ipynb includes more sessions than the criterion-truncated thesis inputs for several mice.']}
(ROOT/'audit/report.json').write_text(json.dumps(report,indent=2))
(WEB/'audit-report.json').write_text(json.dumps(report,indent=2))
(WEB/'reference-run.json').write_text(json.dumps(reference,separators=(',',':')))
print(json.dumps({'checks_passed':len(checks),'max_browser_error':max(c['max_abs_error'] for c in checks if c['check'].startswith('browser')),'initial_findings':report['initial_findings']},indent=2))
