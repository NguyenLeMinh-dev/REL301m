"""Read-only gradient decomposition at the last frozen checkpoint; no optimizer steps."""
import json
from pathlib import Path
import torch
from rel301m.algorithms.masac import MASAC
from rel301m.imitation.demo_finetune import DemonstrationReplay
from rel301m.utils.console import install_panda_warning_filter
install_panda_warning_filter()
root=Path.cwd()
checkpoint=root/'experiments/phase3/bc_finetune_10k_seed0/checkpoint_0001000.pt'
model,metadata=MASAC.load(checkpoint,device='cpu',load_optimizers=False)
model.q1.requires_grad_(False);model.q2.requires_grad_(False)
model.configure_log_std()
demos=DemonstrationReplay(root/'data/demonstrations/two_arm_lift_scripted',
    root/'experiments/phase3/bc_seed42/best.pt',model,metadata['env_config'],0)
batch=demos.replay.sample(256,'cpu')
result=dict(checkpoint=str(checkpoint.relative_to(root)),gradient_probe_seed=42,
    training_demo_only=True, optimizer_steps_executed=0, caveat='fixed demonstration batch, not reconstructed historical minibatch')
torch.set_num_threads(1)
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(42)
    for i,actor in enumerate(model.actors):
        own,logp=actor.sample(batch[f'o{i}'])
        with torch.no_grad(): other,_=model.actors[1-i].sample(batch[f'o{1-i}'])
        joint=torch.cat([own,other] if i==0 else [other,own],dim=-1)
        q_loss=-torch.minimum(model.q1(batch['s'],joint),model.q2(batch['s'],joint)).mean()
        entropy_loss=(model.alpha[i].detach()*logp).mean()
        mean,_=actor(batch[f'o{i}'])
        bc_loss=.5*(actor.action_bias+actor.action_scale*mean.tanh()-batch[f'a{i}']).square().mean()
        gradients={}
        for name,loss in [('negative_Q',q_loss),('entropy',entropy_loss),('weighted_BC',bc_loss),
                          ('total',q_loss+entropy_loss+bc_loss)]:
            grad=torch.autograd.grad(loss,list(actor.parameters()),retain_graph=True)
            gradients[name]=dict(loss=loss.item(),gradient_l2=torch.cat([g.reshape(-1) for g in grad]).norm().item())
        result[f'agent_{i}']=gradients
output=root/'experiments/phase3/stabilize_baseline_audit/gradient_decomposition.json'
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
