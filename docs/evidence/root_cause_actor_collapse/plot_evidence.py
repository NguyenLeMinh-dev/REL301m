"""Render the completed root-cause diagnostics; no model or simulator imports."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
output=root
output.mkdir(parents=True,exist_ok=True)
results=json.loads((root/'results.json').read_text())
colors={'restored':'#176a9e','A_fixed_critic':'#cf621f','C_no_Q_actor_gradient':'#28884a',
        'D_no_BC':'#a4538d','snapshot':'#777777','E_no_entropy':'#aa8628','F_BC_only':'#222222','B_fixed_actor':'#689fab'}
labels={'restored':'Restored control','A_fixed_critic':'A: fixed critic','C_no_Q_actor_gradient':'C: no Q gradient',
        'D_no_BC':'D: no BC','snapshot':'Snapshot teammates','E_no_entropy':'E: no entropy',
        'F_BC_only':'F: BC only','B_fixed_actor':'B: fixed actor'}
fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
for i in range(2):
 ax=axes[0,i]
 for name in colors:
  if name=='B_fixed_actor':continue
  with (root/'fork_drift.csv').open(newline='') as stream:
   rows=[row for row in csv.DictReader(stream) if row['variant']==name]
  ax.semilogy(range(1,101),[float(r[f'validation_bc_drift_{i}']) for r in rows],color=colors[name],
              label=labels[name],ls='--' if name=='snapshot' else '-',lw=1.4)
 ax.set(title=f'Actor {i}: deterministic validation drift',xlabel='Controlled update window',ylabel='Bounded action MSE vs original BC')
 ax.text(.02,.97,'Fixed actor: exactly zero drift',transform=ax.transAxes,va='top',fontsize=9)
 ax.grid(alpha=.2)
handles,legend_labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,legend_labels,fontsize=9,loc='lower center',bbox_to_anchor=(.5,-.055),ncol=4)
ax=axes[1,0]
for name in ['restored','A_fixed_critic','B_fixed_actor','C_no_Q_actor_gradient','F_BC_only']:
 r=results['forks'][name]
 xs=[0,1,10,100];ys=[.7]+[r['evaluations'][str(n)]['success_rate'] for n in xs[1:]]
 ax.plot(xs,ys,marker='o',color=colors[name],label=labels[name],lw=1)
ax.set(xlabel='Controlled update window',ylabel='Ever-success fraction',title='Same paired 10 states at each checkpoint',ylim=(-.04,.82))
ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
ax.text(.02,.97,'Markers are measured; lines guide the eye.\nDo not pool repeated paired states.',transform=ax.transAxes,fontsize=8,va='top')
ax=axes[1,1]
trace=[json.loads(line) for line in (root/'actor_updates.jsonl').read_text().splitlines()]
first=[r for r in trace if r['actor_update']==1]
for i,r in enumerate(first):
 ax.bar([j+(-.18 if i==0 else .18) for j in range(3)],
        [r['gradients'][name]['gradient_l2'] for name in ['Q','entropy','BC']],width=.34,
        color=['#176a9e','#cf621f'][i],label=f'Actor {i}')
ax.set_xticks(range(3),['Q','Entropy','Weighted BC'])
ax.set_yscale('log');ax.set(ylabel='Exact minibatch gradient L2',title='First actor update: component gradients')
ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
ax.text(.32,.85,'Actual Adam delta L2: 0.08585 / 0.08575\nMax analytic delta error: 9.54e-7',transform=ax.transAxes,va='top',fontsize=9)
fig.suptitle('REL301m causal audit — ROOT CAUSE NOT YET ESTABLISHED',fontsize=14)
fig.savefig(output/'causal_overview.png',dpi=160,bbox_inches='tight')
fig.savefig(output/'causal_overview.pdf',bbox_inches='tight')
print('Saved causal_overview PNG/PDF')
