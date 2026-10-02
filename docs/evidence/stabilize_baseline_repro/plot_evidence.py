"""Render selected diagnostic CSVs without simulator or learner execution."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('docs/evidence/stabilize_baseline_repro')
r=json.loads((p/'results.json').read_text())
rows=list(csv.DictReader((p/'stabilize_repro_3k/early_diagnostics.csv').open()))
pol=list(csv.DictReader((p/'stabilize_repro_3k/policy_diagnostics.csv').open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
a=axs[0]
x=[int(z['environment_steps']) for z in rows]
for i,color in enumerate(['#176a9e','#cf621f']):
 a.plot(x,[float(z[f'validation_bc_drift_mse_{i}']) for z in rows],color=color,label=f'Actor {i}')
a.axvline(1000,color='#555555',ls='--',lw=1,label='Actors start after step 1000')
a.axvline(2000,color='#777777',ls=':',lw=1,label='Std head released after step 2000')
a.set(xlabel='Training environment steps',ylabel='Bounded mean-action MSE vs BC',title='Original failing fine-tune: validation drift',xlim=(0,3000),ylim=(-.008,.27))
a.legend(fontsize=8,loc='upper left');a.grid(alpha=.2)
a=axs[1]
full=[(0,.7)]+[(int(z['environment_steps']),float(z['success_rate'])) for z in pol if z['mode']=='deterministic' and z['episodes']=='10']
a.scatter(*zip(*full),color='#176a9e',marker='o',label='Deterministic / full 10')
for mode,marker,color in [('deterministic','x','#176a9e'),('stochastic','+','#cf621f')]:
 sub=[z for z in pol if z['mode']==mode and z['episodes']=='2']
 a.scatter([int(z['environment_steps']) for z in sub],[float(z['success_rate'])+(.014 if mode=='stochastic' else -.014) for z in sub],marker=marker,color=color,label=f'{mode.title()} / prefix 2')
a.axvline(1000,color='#555555',ls='--',lw=1)
a.text(0.04,.1,'Prefix markers offset by ±0.014 for visibility\nTwo-episode probes do not establish 0/10',transform=a.transAxes,fontsize=8)
a.set(xlabel='Training environment steps',ylabel='Ever-success fraction',title='Paired success; different sample sizes',ylim=(-.08,.85));a.grid(alpha=.2);a.legend(fontsize=8,loc='upper right')
a=axs[2]
names=['control','a_critic_only','b_initial_std','c_actor_lr']
labels=['Control','A: critic-only','B: std = -3','C: actor LR']
for i,color in enumerate(['#176a9e','#cf621f']):
 vals=[r['runs'][f'stabilize_{n}_2k']['final_checkpoint_drift'][f'validation_bc_drift_mse_{i}'] for n in names]
 a.bar([j+(-.18 if i==0 else .18) for j in range(4)],vals,width=.34,color=color,label=f'Actor {i}')
a.set_xticks(range(4),labels,rotation=15)
a.set(ylabel='Bounded mean-action MSE vs BC',title='Isolated controls: exact final 2k checkpoint',ylim=(0,.28));a.legend(fontsize=8);a.grid(axis='y',alpha=.2)
a.text(.03,.94,'All four: ever-success 0/10, final-success 0/10',transform=a.transAxes,fontsize=8)
fig.suptitle('REL301m seed-0 audit — reproducible collapse; no treatment establishes recovery',fontsize=13)
fig.savefig(p/'diagnostic_overview.png',dpi=160)
fig.savefig(p/'diagnostic_overview.pdf')
print('Saved diagnostic_overview.png / .pdf')
