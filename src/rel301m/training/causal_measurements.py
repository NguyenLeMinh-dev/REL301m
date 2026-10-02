"""Train-only critic ranking, action derivatives and finite-horizon return comparisons."""
from copy import deepcopy
import json

import numpy as np
import torch
from torch.nn import functional as F

from rel301m.algorithms.masac import MASAC
from .causal_tools import bounded_mean, discounted_returns
from .causal_forks import q_action_metrics


def fixed_training_states(trace, device, gamma, count=128):
    indices=json.loads((trace.output/'probe_indices.json').read_text())['train'][:count]
    rows=[]
    for episode in trace.split['train']:
        data=trace.episodes[episode]
        returns=discounted_returns(data['r'],gamma)
        rows.extend(dict(episode=episode,timestep=t,G=float(returns[t]),
                         **{key:data[key][t] for key in ['s','o0','o1','a0','a1']}) for t in range(len(returns)))
    chosen=[rows[index] for index in indices]
    tensors={key:torch.as_tensor(np.stack([row[key] for row in chosen]),device=device)
             for key in ['s','o0','o1','a0','a1']}
    metadata=[dict(flat_train_index=index,episode=row['episode'],timestep=row['timestep'],
                   finite_horizon_return_to_go=row['G']) for index,row in zip(indices,chosen)]
    return tensors,metadata


def actor_Q_descent_jvp(actor, observations, gradient_vector):
    parameters=dict(actor.named_parameters())
    tangents={};offset=0
    for name,parameter in parameters.items():
        count=parameter.numel()
        tangents[name]=-gradient_vector[offset:offset+count].reshape(parameter.shape).to(parameter)
        offset+=count
    def means(values):
        mean,_=torch.func.functional_call(actor,values,(observations,))
        return actor.action_bias+actor.action_scale*mean.tanh()
    _,direction=torch.func.jvp(means,(parameters,),(tangents,))
    return direction.detach()


def critic_audit(trace, checkpoint, bc_checkpoint, device):
    pre,_=MASAC.load(checkpoint,device=device)
    raw_bc,_=MASAC.load(bc_checkpoint,device=device,load_optimizers=False)
    fixed,metadata=fixed_training_states(trace,device,pre.config['gamma'])
    expert=torch.cat([fixed['a0'],fixed['a1']],dim=-1)
    bc=torch.cat([bounded_mean(actor,fixed[f'o{i}']) for i,actor in enumerate(raw_bc.actors)],dim=-1)
    actions={'expert':expert,'BC_mean':bc}
    devices=list(range(torch.cuda.device_count())) if str(device)=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(20261002)
        with torch.no_grad():
            actions['raw_BC_stochastic']=torch.cat([actor.sample(fixed[f'o{i}'])[0] for i,actor in enumerate(raw_bc.actors)],dim=-1)
            actions['configured_BC_stochastic']=torch.cat([actor.sample(fixed[f'o{i}'])[0] for i,actor in enumerate(pre.actors)],dim=-1)
    for n in [1,10,100]:
        model,_=MASAC.load(trace.output/f'after_{n:03d}.pt',device=device,load_optimizers=False)
        actions[f'actor_after_{n}']=torch.cat([bounded_mean(actor,fixed[f'o{i}']) for i,actor in enumerate(model.actors)],dim=-1)
    values_by_action={};rows=[]
    for name,action in actions.items():
        values,gradients=q_action_metrics(pre,fixed['s'],action)
        values_by_action[name]=values
        for j,meta in enumerate(metadata):
            rows.append(dict(**meta,action_variant=name,action=action[j].cpu().tolist(),
                Q1=float(values[0][j]),Q2=float(values[1][j]),min_Q=float(values[2][j]),
                dQ1_da=gradients[0][j].cpu().tolist(),dQ2_da=gradients[1][j].cpu().tolist(),
                dminQ_da=gradients[2][j].cpu().tolist(),
                mse_to_BC=float((action[j]-bc[j]).square().mean()),
                mse_to_expert=float((action[j]-expert[j]).square().mean())))
    _,bc_gradients=q_action_metrics(pre,fixed['s'],bc)
    perturbations=[]
    generator=torch.Generator(device=device);generator.manual_seed(20261004)
    offset=0
    for i,width in enumerate(pre.action_dims):
        actor_artifact=torch.load(trace.output/f'actor_001_{i}.pt',map_location='cpu',weights_only=True)
        jvp=actor_Q_descent_jvp(pre.actors[i],fixed[f'o{i}'],actor_artifact['component_gradients']['Q'])
        directions={'toward_expert':expert[:,offset:offset+width]-bc[:,offset:offset+width],
                    'away_from_expert':bc[:,offset:offset+width]-expert[:,offset:offset+width],
                    'dQ_da':bc_gradients[2][:,offset:offset+width],
                    'exact_actor_Q_parameter_descent_jvp':jvp}
        for n in [1,10,100]:
            directions[f'actor_delta_{n}']=actions[f'actor_after_{n}'][:,offset:offset+width]-bc[:,offset:offset+width]
        for n in range(2): directions[f'random_{n}']=torch.randn((len(bc),width),device=device,generator=generator)
        low,high=(torch.as_tensor(a,device=device,dtype=bc.dtype) for a in pre.action_specs[i])
        for name,direction in directions.items():
            unit=F.normalize(direction,dim=-1)
            for epsilon in [.001,.01,.05,.1]:
                perturbed=bc.clone()
                perturbed[:,offset:offset+width]=(bc[:,offset:offset+width]+epsilon*unit).clamp(low,high)
                with torch.no_grad():
                    q1,q2=pre.q1(fixed['s'],perturbed),pre.q2(fixed['s'],perturbed)
                    delta=torch.minimum(q1,q2)-values_by_action['BC_mean'][2]
                for j,meta in enumerate(metadata):
                    perturbations.append(dict(**meta,agent=i,direction=name,epsilon=epsilon,
                        delta_min_Q=float(delta[j]),
                        action_distance_from_BC=float((perturbed[j]-bc[j]).norm()),
                        delta_expert_mse=float((perturbed[j]-expert[j]).square().mean()-(bc[j]-expert[j]).square().mean())))
        offset+=width
    summaries={}
    for name,values in values_by_action.items():
        gt=torch.tensor([meta['finite_horizon_return_to_go'] for meta in metadata],device=device,dtype=values[2].dtype).reshape(-1,1)
        summaries[name]=dict(min_Q_mean=float(values[2].mean()),finite_G_mean=float(gt.mean()),
            Q_minus_finite_G_mean=float((values[2]-gt).mean()),
            Q_finite_G_mse=float((values[2]-gt).square().mean()),
            fraction_Q_higher_than_BC=float((values[2]>values_by_action['BC_mean'][2]).float().mean()),
            action_mse_to_BC=float((actions[name]-bc).square().mean()),
            action_mse_to_expert=float((actions[name]-expert).square().mean()))
    grouped={}
    for i in range(2):
        for direction in ['away_from_expert','exact_actor_Q_parameter_descent_jvp','actor_delta_100']:
            for epsilon in [.001,.01,.05,.1]:
                selected=[row for row in perturbations if row['agent']==i and row['direction']==direction and row['epsilon']==epsilon]
                grouped[f'agent{i}:{direction}:eps{epsilon}']=dict(
                    mean_delta_Q=float(np.mean([row['delta_min_Q'] for row in selected])),
                    fraction_Q_increases=float(np.mean([row['delta_min_Q']>0 for row in selected])),
                    fraction_Q_increases_while_expert_MSE_worsens=float(np.mean([
                        row['delta_min_Q']>0 and row['delta_expert_mse']>0 for row in selected])))
    (trace.output/'critic_action_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (trace.output/'critic_perturbation_rows.json').write_text(json.dumps(perturbations,indent=2)+'\n')
    return dict(fixed_train_states=len(metadata),critic_checkpoint=str(checkpoint),
                policy_ranking=summaries,perturbation_summary=grouped,
                finite_G_limitation='Recorded teacher rewards only; no unknown post-horizon continuation and no SAC entropy terms. Not an unbiased estimate of stochastic soft-policy Q.',
                derivatives='Per-state derivatives of Q1/Q2/min Q wrt joint actions; teammate held at BC mean during single-agent perturbations.',
                jvp='Exact update-1 Q parameter gradient mapped through each pre-update actor mean Jacobian; update-1 actor gradient used the Q after the env1001 critic step.')
