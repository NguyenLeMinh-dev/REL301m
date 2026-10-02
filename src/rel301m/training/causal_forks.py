"""Explicit diagnostic objectives; never selected by production MASAC training."""
from copy import deepcopy
import torch
from torch.nn import functional as F

from .causal_tools import component_gradient, cosine


def diagnostic_update(model, batch, demo_batch, variant='restored', measure=True):
    """One controlled replay update; settings remain unchanged in model.config."""
    allowed={'restored','A_fixed_critic','B_fixed_actor','C_no_Q_actor_gradient','D_no_BC','snapshot','E_no_entropy','F_BC_only'}
    if variant not in allowed:
        raise ValueError(f'Unknown diagnostic fork: {variant}')
    model.configure_log_std()
    # Every variant consumes the same target-policy draws. RNG is restored per recorded update by caller.
    target=model.critic_target(batch)
    joint=torch.cat((batch['a0'],batch['a1']),dim=-1)
    q1,q2=model.q1(batch['s'],joint),model.q2(batch['s'],joint)
    loss_q=F.mse_loss(q1,target)+F.mse_loss(q2,target)
    if variant!='A_fixed_critic':
        model.q_optimizer.zero_grad(set_to_none=True)
        loss_q.backward()
        model.q_optimizer.step()
        model.critic_optimizer_updates+=1
        model.q_optimizer.zero_grad(set_to_none=True)
    diagnostics=[]
    if variant!='B_fixed_actor':
        snapshot=deepcopy(model.actors) if variant=='snapshot' else None
        log_probs=[]
        model.q1.requires_grad_(False);model.q2.requires_grad_(False)
        try:
            for i,optimizer in enumerate(model.actor_optimizers):
                action,log_prob=model.actors[i].sample(batch[f'o{i}'])
                other=1-i
                with torch.no_grad():
                    other_action,_=(snapshot if snapshot is not None else model.actors)[other].sample(batch[f'o{other}'])
                actions=[None,None];actions[i]=action;actions[other]=other_action
                q=torch.minimum(model.q1(batch['s'],torch.cat(actions,dim=-1)),
                                model.q2(batch['s'],torch.cat(actions,dim=-1)))
                entropy=(model.alpha[i].detach()*log_prob).mean()
                q_term=-q.mean()
                actor=model.actors[i]
                mean,_=actor(demo_batch[f'o{i}'])
                predicted=actor.action_bias+actor.action_scale*mean.tanh()
                bc=model.config['fine_tune']['lambda_bc']*F.mse_loss(predicted,demo_batch[f'a{i}'])
                # Keep the original reduction/order in the restored control.
                if variant=='C_no_Q_actor_gradient':
                    loss=entropy
                elif variant=='E_no_entropy':
                    loss=q_term
                elif variant=='F_BC_only':
                    loss=torch.zeros((),device=model.device)
                else:
                    loss=(model.alpha[i].detach()*log_prob-q).mean()
                if variant!='D_no_BC': loss=loss+bc
                if measure:
                    q_vector,q_metrics=component_gradient(q_term,actor)
                    bc_vector,bc_metrics=component_gradient(bc,actor)
                    entropy_vector,entropy_metrics=component_gradient(entropy,actor)
                    diagnostics.append(dict(agent=i,Q=q_metrics,BC=bc_metrics,entropy=entropy_metrics,
                                            Q_vs_BC_cosine=cosine(q_vector,bc_vector),
                                            entropy_vs_BC_cosine=cosine(entropy_vector,bc_vector)))
                optimizer.zero_grad(set_to_none=True)
                loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
                model.actor_optimizer_updates[i]+=1
                log_probs.append(log_prob.detach())
        finally:
            model.q1.requires_grad_(True);model.q2.requires_grad_(True)
        alpha_loss=sum(-(model.log_alpha[i]*(log_probs[i]+model.target_entropy[i])).mean() for i in range(2))
        model.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward();model.alpha_optimizer.step();model.alpha_optimizer_updates+=1
    else:
        with torch.no_grad():
            # Match the native critic-only branch's diagnostic sample draws.
            for i,actor in enumerate(model.actors): actor.sample(batch[f'o{i}'])
    if variant!='A_fixed_critic': model.soft_update()
    model.updates+=1
    return dict(q_loss=float(loss_q.detach()),components=diagnostics)


def q_action_metrics(model,state,joint_action):
    action=joint_action.detach().clone().requires_grad_(True)
    q1,q2=model.q1(state,action),model.q2(state,action)
    minimum=torch.minimum(q1,q2)
    gradients=[torch.autograd.grad(q.sum(),action,retain_graph=True)[0].detach() for q in [q1,q2,minimum]]
    return [q.detach() for q in [q1,q2,minimum]],gradients


def local_bc_gradient(model,states,observations,expert_joint):
    from .causal_tools import bounded_mean
    bc=torch.cat([bounded_mean(actor,obs) for actor,obs in zip(model.actors,observations)],dim=-1)
    values,gradients=q_action_metrics(model,states,bc)
    result=dict(min_Q_mean=float(values[2].mean()))
    offset=0
    for i,width in enumerate(model.action_dims):
        gradient=gradients[2][:,offset:offset+width]
        correction=expert_joint[:,offset:offset+width]-bc[:,offset:offset+width]
        result[f'dQ_da_norm_{i}']=float(gradient.norm(dim=-1).mean())
        result[f'dQ_toward_expert_cosine_{i}']=float(F.cosine_similarity(gradient,correction,dim=-1).mean())
        offset+=width
    return result
