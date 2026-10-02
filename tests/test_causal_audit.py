"""Causal observer invariants, real dataset alignment and analytic optimizer checks."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from rel301m.algorithms.masac import MASAC
from rel301m.algorithms.replay_buffer import ReplayBuffer
from rel301m.envs.robosuite_factory import PROJECT_ROOT
from rel301m.imitation.demo_finetune import DemonstrationReplay
from rel301m.imitation.demonstrations import load_dataset, validate_trajectory
from rel301m.training.causal_tools import (peek_indices, composition, tree_hash, adam_expected_delta,
                                          discounted_returns, rng_state, restore_rng)
from rel301m.training.causal_trace import CausalTrace


def tiny_model():
    config = dict(hidden_dims=[16,16],actor_lr=3e-4,critic_lr=3e-4,alpha_lr=3e-4,
                  initial_alpha=.02,gamma=.99,tau=.005,bootstrap_time_limits=True,
                  fine_tune=dict(lambda_bc=.5,actor_init_log_std=-3,freeze_log_std_updates=3000))
    model = MASAC([4,4],9,[(np.full(2,-1.),np.full(2,1.))]*2,config)
    model.initialize_log_std(-3)
    model.configure_log_std()
    return model


def synthetic_demos(tmp_path,model):
    rng=np.random.default_rng(41)
    dataset=tmp_path/'demos';dataset.mkdir()
    metadata=dict(format_version=1,status='completed',env_config={'horizon':8},
                  observation_dims=[4,4],critic_state_dim=9,action_dims=[2,2],
                  action_specs=[(lo.tolist(),hi.tolist()) for lo,hi in model.action_specs],
                  successful_episodes=3,episodes=[])
    episodes=[]
    for index in range(3):
        data={}
        for key,width in [('o0',4),('o1',4),('s',9)]:
            sequence=rng.normal(size=(9,width)).astype(np.float32)
            data[key],data[f'next_{key}']=sequence[:-1],sequence[1:]
        for key in ['a0','a1']: data[key]=rng.uniform(-.6,.6,(8,2)).astype(np.float32)
        data['r']=rng.uniform(0,1,(8,1)).astype(np.float32)
        data['done']=np.zeros((8,1),np.float32);data['done'][-1]=1
        data['timeout']=data['done'].copy()
        data.update(success=np.ones(9,bool),grasps=np.ones((9,2),bool),qpos=np.zeros((9,3)),qvel=np.zeros((9,3)))
        file=dataset/f'episode_{index}.npz';np.savez_compressed(file,**data)
        metadata['episodes'].append(dict(episode=index,file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        episodes.append(data)
    manifest=dataset/'manifest.json';manifest.write_text(json.dumps(metadata))
    checkpoint=tmp_path/'bc.pt'
    model.save(checkpoint,dict(training_stage='behavior_cloning',env_config=metadata['env_config'],
        demonstration_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        episode_split=dict(train=[0],validation=[1],test=[2])))
    return dataset,checkpoint,episodes


def test_sampling_peek_does_not_consume_rng():
    replay=ReplayBuffer(10,[2,2],3,[1,1],seed=23)
    for i in range(10): replay.add([i]*2,[i]*2,[i]*3,[0],[0],i,[i+1]*2,[i+1]*2,[i+1]*3,False)
    old=deepcopy(replay.rng.bit_generator.state)
    indices=peek_indices(replay,4)
    assert replay.rng.bit_generator.state==old
    sample=replay.sample(4)
    assert np.array_equal(sample['r'].numpy().reshape(-1),indices)


def test_effective_demo_ratio_includes_prefilled_main_and_duplicate_identity():
    sources=[('demo',0,1),('demo',1,2),('demo',0,1),('online',0,3)]
    report=composition(sources,direct_count=2)
    assert report['effective_demo_ratio']==.75  # direct fraction is only 2/4
    assert report['effective_demo_samples']==3 and report['effective_online_samples']==1
    assert report['duplicate_rate']==.25
    assert report['unique_episode_distribution']=={'demo:0':2,'demo:1':1,'online:0':1}


@pytest.mark.parametrize('previous_steps',[0,3])
def test_analytic_adam_displacement_matches_actual_with_fresh_and_existing_moments(previous_steps):
    parameter=torch.nn.Parameter(torch.tensor([.6,-.2,0.,3.],dtype=torch.float32))
    optimizer=torch.optim.Adam([parameter],lr=3e-4)
    for step in range(previous_steps):
        parameter.grad=torch.tensor([.1+step,-.4,.02,0.])
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    before=parameter.detach().clone()
    gradient=torch.tensor([17.,-.38,.011,0.])
    if not previous_steps: assert not optimizer.state
    expected=adam_expected_delta(before,gradient,optimizer.state.get(parameter,{}),optimizer.param_groups[0])
    parameter.grad=gradient;optimizer.step()
    torch.testing.assert_close(parameter.detach()-before,expected,rtol=0,atol=2e-7)


def test_observer_is_rng_neutral_and_bitwise_equal_to_uninstrumented_update(tmp_path):
    torch.manual_seed(5)
    original=tiny_model();observed=tiny_model();observed.load_state_dict(original.state_dict())
    dataset,checkpoint,episodes=synthetic_demos(tmp_path,observed)
    indices=np.random.default_rng(0).integers(8,size=4)
    batch={key:torch.from_numpy(value[indices]) for key,value in episodes[0].items() if key in
           ['o0','o1','s','a0','a1','r','next_o0','next_o1','next_s','done','timeout']}
    start=rng_state()
    original.update(batch,demo_batch=batch)
    rng_after=rng_state()
    restore_rng(start)
    with CausalTrace(dataset,checkpoint,tmp_path/'trace',subset_size=4) as trace:
        replay=ReplayBuffer(100,[4,4],9,[2,2])
        demos=DemonstrationReplay(dataset,checkpoint,observed,{'horizon':8},0)
        demos.prefill(replay)
        sampled=demos.replay.sample(4)
        observed.update(sampled,demo_batch=sampled)
    assert tree_hash(rng_state())==tree_hash(rng_after)
    for key,tensor in original.state_dict().items(): assert torch.equal(tensor,observed.state_dict()[key]),key
    assert tree_hash(original.actor_optimizers[0].state_dict())==tree_hash(observed.actor_optimizers[0].state_dict())
    assert all(row['optimizer_state_empty_before'] for row in trace.records)
    assert all(row['critic_grad_clean'] and row['teammate_grad_clean'] for row in trace.records)
    assert all(row['combined_gradient_error_l2']<1e-5 for row in trace.records)


def test_fixed_log_std_preserves_mean_gradient_at_initialized_std_head():
    model=tiny_model();actor=model.actors[0]
    observations=torch.randn(8,4)
    start=rng_state()
    action,_=actor.sample(observations)
    fixed=torch.autograd.grad(action.square().mean(),list(actor.parameters()))
    restore_rng(start);actor.fixed_log_std=None
    action,_=actor.sample(observations)
    variable=torch.autograd.grad(action.square().mean(),list(actor.parameters()))
    for a,b in zip(fixed[:-2],variable[:-2]): torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(fixed[-2:],variable[-2:]): torch.testing.assert_close(a[:2],b[:2],rtol=0,atol=0)
    assert all(torch.count_nonzero(a[2:])==0 for a in fixed[-2:])
    assert any(torch.count_nonzero(a[2:])>0 for a in variable[-2:])


def test_discounted_returns_preserve_episode_boundary():
    np.testing.assert_allclose(discounted_returns([1,2,3],.5),[2.75,3.5,3])
    np.testing.assert_allclose(discounted_returns([4],.5),[4])


def real_inputs():
    dataset=PROJECT_ROOT/'data/demonstrations/two_arm_lift_scripted'
    checkpoint=PROJECT_ROOT/'experiments/phase3/bc_seed42/best.pt'
    if not dataset.exists() or not checkpoint.exists(): pytest.skip('Local historical demo/BC artifacts required')
    manifest,episodes=load_dataset(dataset)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    return manifest,episodes,payload


def test_real_train_transition_alignment_replay_fields_and_corruption_detection():
    manifest,episodes,payload=real_inputs()
    rng=np.random.default_rng(20261003)
    model=MASAC(payload['obs_dims'],payload['state_dim'],payload['action_specs'],payload['config'])
    demos=DemonstrationReplay(PROJECT_ROOT/'data/demonstrations/two_arm_lift_scripted',
          PROJECT_ROOT/'experiments/phase3/bc_seed42/best.pt',model,manifest['env_config'],0)
    offset=0
    for index in payload['metadata']['episode_split']['train']:
        episode=episodes[index];validate_trajectory(episode,manifest)
        for row in [0,len(episode['r'])-1,*rng.choice(len(episode['r'])-1,2,replace=False)]:
            for key,array in demos.replay.arrays.items():
                np.testing.assert_array_equal(array[offset+row],episode[key][row])
        assert episode['done'][-1,0]==episode['timeout'][-1,0]==1
        assert not episode['done'][:-1].any()
        offset+=len(episode['r'])
    bad=deepcopy(episodes[payload['metadata']['episode_split']['train'][0]])
    bad['next_s'][5,0]+=1
    with pytest.raises(ValueError,match='alignment'): validate_trajectory(bad,manifest)


def test_real_folded_inputs_match_reconstructed_normalized_actor():
    manifest,episodes,payload=real_inputs()
    assert manifest['actor_observations']=='raw_phase2_float32'
    assert payload['metadata']['actor_input_transform']=='folded_into_first_linear'
    model,_=MASAC.load(PROJECT_ROOT/'experiments/phase3/bc_seed42/best.pt',load_optimizers=False)
    for i,actor in enumerate(model.actors):
        unfolded=deepcopy(actor)
        normalization=payload['metadata']['observation_normalization'][i]
        mean=torch.tensor(normalization['mean']);std=torch.tensor(normalization['std'])
        with torch.no_grad():
            unfolded.net[0].bias.add_(actor.net[0].weight@mean)
            unfolded.net[0].weight.mul_(std)
        observations=torch.from_numpy(np.concatenate([episodes[e][f'o{i}'] for e in
                                     payload['metadata']['episode_split']['train']])[:256])
        with torch.no_grad():
            raw=actor.sample(observations,deterministic=True)[0]
            normalized=unfolded.sample((observations-mean)/std,deterministic=True)[0]
        torch.testing.assert_close(raw,normalized,rtol=0,atol=2e-4)


def test_real_teacher_actions_labels_and_live_observations_use_same_timestep():
    from rel301m.envs.robosuite_factory import make_two_arm_lift
    from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper
    from rel301m.imitation.scripted_teacher import ScriptedLiftTeacher
    manifest,episodes,payload=real_inputs()
    selected=sorted(payload['metadata']['episode_split']['train'])[:2]
    by_attempt={manifest['episodes'][index]['attempt']:index for index in selected}
    env=MultiAgentWrapper(make_two_arm_lift(seed=manifest['seed']))
    try:
        for attempt in range(max(by_attempt)+1):
            observations=env.reset();teacher=ScriptedLiftTeacher(env)
            for t in range(env.horizon):
                actions=teacher.act()
                if attempt in by_attempt:
                    saved=episodes[by_attempt[attempt]]
                    for i in range(2):
                        np.testing.assert_allclose(observations[f'agent_{i}']['actor_obs'],saved[f'o{i}'][t],rtol=0,atol=2e-6)
                        np.testing.assert_allclose(actions[i],saved[f'a{i}'][t],rtol=0,atol=2e-6)
                    np.testing.assert_allclose(env.critic_state,saved['s'][t],rtol=0,atol=2e-6)
                observations,reward,done,_=env.step(*actions)
                if attempt in by_attempt:
                    for i in range(2): np.testing.assert_allclose(observations[f'agent_{i}']['actor_obs'],saved[f'next_o{i}'][t],rtol=0,atol=2e-6)
                    np.testing.assert_allclose(env.critic_state,saved['next_s'][t],rtol=0,atol=2e-6)
                    np.testing.assert_allclose(reward,saved['r'][t,0],rtol=0,atol=2e-6)
                    assert bool(done)==bool(saved['done'][t,0])==bool(saved['timeout'][t,0])
    finally:
        env.close()


def test_restored_diagnostic_matches_native_learner_and_freeze_controls(tmp_path):
    from rel301m.training.causal_forks import diagnostic_update
    torch.manual_seed(24)
    native=tiny_model();diagnostic=tiny_model();diagnostic.load_state_dict(native.state_dict())
    _,_,episodes=synthetic_demos(tmp_path,native)
    batch={key:torch.from_numpy(value[:4]) for key,value in episodes[0].items() if key in
           ['o0','o1','s','a0','a1','r','next_o0','next_o1','next_s','done','timeout']}
    for _ in range(3):
        start=rng_state();native.update(batch,demo_batch=batch)
        expected_rng=rng_state();restore_rng(start)
        diagnostic_update(diagnostic,batch,batch,'restored')
        assert tree_hash(rng_state())==tree_hash(expected_rng)
        for key,tensor in native.state_dict().items(): assert torch.equal(tensor,diagnostic.state_dict()[key]),key
    for variant in ['A_fixed_critic','B_fixed_actor']:
        model=tiny_model();before=deepcopy(model.state_dict())
        diagnostic_update(model,batch,batch,variant)
        frozen_prefixes=['q1.','q2.','target_q1.','target_q2.'] if variant=='A_fixed_critic' else ['actors.','log_alpha']
        for name,old in before.items():
            if any(name.startswith(prefix) for prefix in frozen_prefixes): assert torch.equal(old,model.state_dict()[name])


def test_remove_Q_fork_changes_only_actor_gradient_component(tmp_path):
    from rel301m.training.causal_forks import diagnostic_update
    torch.manual_seed(9);model=tiny_model()
    _,_,episodes=synthetic_demos(tmp_path,model)
    batch={key:torch.from_numpy(value[:4]) for key,value in episodes[0].items() if key in
           ['o0','o1','s','a0','a1','r','next_o0','next_o1','next_s','done','timeout']}
    before=tree_hash(model.q1.state_dict())
    result=diagnostic_update(model,batch,batch,'C_no_Q_actor_gradient')
    assert tree_hash(model.q1.state_dict())!=before
    assert model.optimizer_counts()['actor_optimizer_updates_0']==1
    assert result['components'][0]['Q']['gradient_l2']>0
    assert model.config['fine_tune']['lambda_bc']==.5 and model.config['actor_lr']==3e-4


@pytest.mark.parametrize('variant',['E_no_entropy','F_BC_only'])
def test_additional_entropy_and_BC_only_diagnostics_keep_settings_and_counts(tmp_path,variant):
    from rel301m.training.causal_forks import diagnostic_update
    model=tiny_model();before=deepcopy(model.config)
    _,_,episodes=synthetic_demos(tmp_path,model)
    batch={key:torch.from_numpy(value[:4]) for key,value in episodes[0].items() if key in
           ['o0','o1','s','a0','a1','r','next_o0','next_o1','next_s','done','timeout']}
    diagnostic_update(model,batch,batch,variant)
    assert model.config==before
    assert model.optimizer_counts()==dict(critic_optimizer_updates=1,actor_optimizer_updates_0=1,
        actor_optimizer_updates_1=1,actor_optimizer_steps_total=2,alpha_optimizer_updates=1)


def test_fixed_std_entropy_bound_and_mean_pressure_oppose_saturated_expert_action():
    import math
    from rel301m.algorithms.networks import GaussianActor
    actor=GaussianActor(66,np.full(7,-1.),np.full(7,1.),hidden_dims=[16,16])
    with torch.no_grad():
        for parameter in actor.parameters(): parameter.zero_()
        actor.net[-1].bias[:7].fill_(3.)
    actor.fixed_log_std=-3
    observations=torch.zeros(1024,66)
    _,log_prob=actor.sample(observations)
    entropy_gradient=torch.autograd.grad(.02*log_prob.mean(),actor.net[-1].bias)[0]
    mean,_=actor(observations)
    bc_gradient=torch.autograd.grad(.5*(mean.tanh()-1).square().mean(),actor.net[-1].bias)[0]
    # Gaussian entropy is an upper bound: tanh's log Jacobian is nonpositive for unit scale.
    entropy_upper_bound=7*(.5*math.log(2*math.pi*math.e)-3)
    assert entropy_upper_bound < -7
    np.testing.assert_allclose(entropy_upper_bound,-11.06743026756729,rtol=0,atol=1e-10)
    assert torch.all(entropy_gradient[:7]>.039)
    assert torch.all(bc_gradient[:7]<0)
    assert torch.count_nonzero(entropy_gradient[7:])==torch.count_nonzero(bc_gradient[7:])==0


def test_no_Q_actor_gradient_is_invariant_to_changing_critic_landscape(tmp_path):
    from rel301m.training.causal_forks import diagnostic_update
    torch.manual_seed(72)
    first=tiny_model();second=tiny_model();second.load_state_dict(first.state_dict())
    with torch.no_grad():
        for critic in [second.q1,second.q2]:
            for parameter in critic.parameters(): parameter.mul_(7)
    _,_,episodes=synthetic_demos(tmp_path,first)
    batch={key:torch.from_numpy(value[:4]) for key,value in episodes[0].items() if key in
           ['o0','o1','s','a0','a1','r','next_o0','next_o1','next_s','done','timeout']}
    start=rng_state()
    diagnostic_update(first,batch,batch,'C_no_Q_actor_gradient')
    restore_rng(start)
    diagnostic_update(second,batch,batch,'C_no_Q_actor_gradient')
    assert tree_hash(first.q1.state_dict())!=tree_hash(second.q1.state_dict())
    for name,parameter in first.actors.state_dict().items():
        assert torch.equal(parameter,second.actors.state_dict()[name]),name
    assert tree_hash(first.actor_optimizers[0].state_dict())==tree_hash(second.actor_optimizers[0].state_dict())
