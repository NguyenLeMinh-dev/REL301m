from copy import deepcopy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from rel301m.algorithms.masac import MASAC
from rel301m.envs.robosuite_factory import PROJECT_ROOT
from rel301m.imitation.bc import normalized_action_loss
from rel301m.imitation.warm_start import warm_start_from_bc
from rel301m.training.baseline_audit import BaselineAudit, BCReferencePolicy, actor_sha
from rel301m.training.train import load_config


def learner(seed=7):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    cfg = load_config(PROJECT_ROOT/'configs/experiment/smoke.yaml')['algo']
    cfg['hidden_dims'] = [32, 32]
    return MASAC([66,66],119,[(-np.ones(7),np.ones(7))]*2,cfg)


def batch():
    data={key:torch.randn(8,width) for key,width in dict(o0=66,o1=66,s=119,a0=7,a1=7,r=1,
          next_o0=66,next_o1=66,next_s=119).items()}
    data.update(done=torch.zeros(8,1),timeout=torch.zeros(8,1))
    return data


@pytest.mark.parametrize('log_prob,direction', [(20.,1),(-20.,-1)])
def test_temperature_optimizer_direction_against_target_entropy(log_prob,direction,monkeypatch):
    model=learner()
    def loss(i,data,**kwargs):
        return sum(p.sum()*0 for p in model.actors[i].parameters()), torch.full((8,1),log_prob)
    monkeypatch.setattr(model,'actor_loss',loss)
    initial=model.log_alpha.detach().clone()
    model.update(batch())
    assert torch.equal(model.target_entropy,torch.tensor([-7.,-7.]))
    # dL/dlog_alpha = estimated_entropy - target_entropy.
    assert ((model.log_alpha.detach()-initial)*direction>0).all()
    assert model.alpha_optimizer_updates == 1


def test_exact_optimizer_counts_and_legacy_checkpoint_inference(tmp_path):
    model=learner()
    data=batch()
    model.update(data,update_actor=False)
    model.update(data,update_actor=False)
    model.update(data)
    expected=dict(critic_optimizer_updates=3,actor_optimizer_updates_0=1,
                  actor_optimizer_updates_1=1,actor_optimizer_steps_total=2,alpha_optimizer_updates=1)
    assert model.optimizer_counts()==expected
    path=tmp_path/'model.pt'
    model.save(path)
    restored,_=MASAC.load(path,load_optimizers=False)
    assert restored.optimizer_counts()==expected
    payload=torch.load(path,weights_only=True)
    del payload['optimizer_update_counts']
    torch.save(payload,path)
    restored,_=MASAC.load(path,load_optimizers=False)
    assert restored.optimizer_counts()==expected
    assert restored.updates==3


def test_sequential_actor_update_is_explicitly_preserved(monkeypatch):
    model=learner()
    before=actor_sha(model)[0]
    observed={}
    original=model.actor_loss
    def loss(i,data,**kwargs):
        observed[i]=actor_sha(model)[0]
        return original(i,data,**kwargs)
    monkeypatch.setattr(model,'actor_loss',loss)
    model.update(batch())
    assert observed[0]==before and observed[1]!=before


def test_bc_does_not_supervise_std_head_but_shared_features_can_change_std():
    model=learner()
    actor=model.actors[0]
    obs=torch.randn(16,66)
    weights=actor.net[-1].weight[7:].detach().clone()
    biases=actor.net[-1].bias[7:].detach().clone()
    original_std=actor(obs)[1].detach().clone()
    loss=normalized_action_loss(actor,obs,torch.ones(16,7)*.8)
    loss.backward()
    assert torch.count_nonzero(actor.net[-1].weight.grad[7:])==0
    assert torch.count_nonzero(actor.net[-1].bias.grad[7:])==0
    model.actor_optimizers[0].step()
    torch.testing.assert_close(weights,actor.net[-1].weight[7:],atol=0,rtol=0)
    torch.testing.assert_close(biases,actor.net[-1].bias[7:],atol=0,rtol=0)
    assert not torch.equal(original_std,actor(obs)[1])


def test_loaded_bc_means_match_and_unsupervised_distribution_is_different(tmp_path):
    source=learner(17)
    path=tmp_path/'bc.pt'
    source.save(path,dict(training_stage='behavior_cloning',actor_input_transform='folded_into_first_linear',
       demonstration_manifest_sha256='d'*64,seed=17,best_epoch=2))
    target=learner()
    warm_start_from_bc(target,path)
    observations=[np.zeros(66),np.ones(66)]
    for before,after in zip(source.act(observations,True),target.act(observations,True)):
        np.testing.assert_array_equal(before,after)
    torch.manual_seed(42)
    sampled=target.act(observations,False)
    assert any(not np.allclose(a,b) for a,b in zip(sampled,target.act(observations,True)))


def test_fixed_drift_subsets_are_rng_neutral_and_frozen_parameters_checked(tmp_path):
    model=learner()
    rng=np.random.default_rng(5)
    arrays={f'{key}{i}':rng.normal(size=(10,66 if key=='o' else 7)).astype(np.float32)
            for key in ['o','a'] for i in range(2)}
    class Retained:
        def __len__(self): return 10
    retained=Retained();retained.arrays=arrays
    demos=SimpleNamespace(replay=retained,validation=[(torch.randn(8,66),torch.zeros(8,7))]*2,
                          metadata=dict(test_used=False))
    settings=dict(subset_seed=42,subset_size=4)
    state=torch.get_rng_state().clone()
    reference=deepcopy(model.actors)
    audit=BaselineAudit(model,reference,demos,settings,tmp_path)
    assert torch.equal(state,torch.get_rng_state())
    assert all(v==0 for k,v in audit.drift(model).items() if 'bc_drift' in k)
    model.update(batch(),update_actor=False)
    audit.assert_frozen(model,0)
    audit.before_update(model,1,True)
    model.update(batch())
    assert any(v>0 for k,v in audit.drift(model).items() if 'bc_drift' in k)
    rows=[json.loads(line) for line in (tmp_path/'optimizer_events.jsonl').read_text().splitlines()]
    assert rows[0]['actor_sha256']==rows[1]['actor_sha256']
    with pytest.raises(RuntimeError,match='freeze'):
        audit.assert_frozen(model,2)


def test_critic_only_schedule_preserves_actor_alpha_sha_and_updates_targets():
    model=learner()
    model.config['critic_only_updates']=2
    initial=actor_sha(model)
    alpha=model.log_alpha.detach().clone()
    target=deepcopy(model.target_q1.state_dict())
    data=batch()
    model.update(data)
    model.update(data)
    assert actor_sha(model)==initial and torch.equal(alpha,model.log_alpha.detach())
    assert model.actor_optimizer_updates==[0,0] and model.alpha_optimizer_updates==0
    assert any(not torch.equal(v,model.target_q1.state_dict()[k]) for k,v in target.items())
    model.update(data)
    assert model.actor_optimizer_updates==[1,1] and model.alpha_optimizer_updates==1
    assert actor_sha(model)!=initial


def test_optional_warm_start_std_override_preserves_mean_and_is_disabled_by_default(tmp_path):
    source=learner(17)
    path=tmp_path/'bc.pt'
    source.save(path,dict(training_stage='behavior_cloning',actor_input_transform='folded_into_first_linear',
       demonstration_manifest_sha256='d'*64,seed=17,best_epoch=2))
    target=learner()
    info=warm_start_from_bc(target,path)
    assert info['log_std_override'] is None
    before=deepcopy(target.state_dict())
    obs=[np.arange(66,dtype=np.float32)/66,np.ones(66,dtype=np.float32)]
    means=target.act(obs,True)
    state=torch.get_rng_state().clone()
    info=warm_start_from_bc(target,path,initial_log_std=-3)
    assert torch.equal(state,torch.get_rng_state())
    assert info['log_std_override']==-3
    for original,current in zip(means,target.act(obs,True)):
        np.testing.assert_array_equal(original,current)
    for i,actor in enumerate(target.actors):
        assert torch.equal(actor(torch.tensor(obs[i]))[1],torch.full((7,),-3.))
    for k,v in target.state_dict().items():
        if not k.startswith('actors.'):
            torch.testing.assert_close(before[k],v,atol=0,rtol=0)
    assert target.optimizer_counts()['critic_optimizer_updates']==0
    with pytest.raises(ValueError):
        warm_start_from_bc(target,path,initial_log_std=float('nan'))


def test_treatments_are_single_factor_against_explicit_short_control():
    base=load_config(PROJECT_ROOT/'configs/experiment/stabilize_control.yaml')
    nominal=load_config(PROJECT_ROOT/'configs/experiment/bc_pilot.yaml')
    assert 'fine_tune' not in base['algo']
    assert base['algo']['warmup_steps']==0 and nominal['algo']['warmup_steps']==10000
    a=load_config(PROJECT_ROOT/'configs/experiment/stabilize_a_critic_only.yaml')
    b=load_config(PROJECT_ROOT/'configs/experiment/stabilize_b_initial_std.yaml')
    c=load_config(PROJECT_ROOT/'configs/experiment/stabilize_c_actor_lr.yaml')
    assert {k for k in a['algo'] if a['algo'][k]!=base['algo'].get(k)}=={'name','critic_only_updates'}
    assert a['algo']['critic_only_updates']==1000
    assert b['algo']==base['algo'] and b['bc_init_log_std']==-3
    assert {k for k in c['algo'] if c['algo'][k]!=base['algo'].get(k)}=={'name','actor_lr'}
    assert c['algo']['actor_lr']==3e-5 and c['algo']['critic_lr']==3e-4
    for cfg in [a,b,c]:
        assert cfg['audit']==base['audit']
        assert cfg['env_config']==base['env_config']
        assert cfg['bc_checkpoint']==base['bc_checkpoint']
