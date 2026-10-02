"""Replay the unchanged failing run and perform bounded, read-only causal forks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn import functional as F

from rel301m.utils.console import install_panda_warning_filter, TrainingProgress
install_panda_warning_filter()
from rel301m.algorithms.masac import MASAC
from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper
from rel301m.envs.robosuite_factory import PROJECT_ROOT, make_two_arm_lift
from rel301m.evaluation.evaluate import evaluate_policy, write_episodes
from .train import load_config, train
from .causal_trace import CausalTrace
from .causal_tools import (tree_hash, rng_state, restore_rng, cpu_tree, policy_probe, bounded_mean)
from .causal_forks import diagnostic_update, q_action_metrics
from .causal_measurements import fixed_training_states, critic_audit


VARIANTS=['restored','A_fixed_critic','B_fixed_actor','C_no_Q_actor_gradient','D_no_BC','snapshot','E_no_entropy','F_BC_only']


def compare_checkpoints(actual, expected):
    a=torch.load(actual,map_location='cpu',weights_only=True)
    b=torch.load(expected,map_location='cpu',weights_only=True)
    mismatches=[name for name,tensor in a['model'].items() if name not in b['model'] or not torch.equal(tensor,b['model'][name])]
    return dict(bitwise_equal_all_model_tensors=not mismatches,mismatches=mismatches,
                actual=str(actual),reference=str(expected))


def capture(config_path,root,historical):
    config=load_config(config_path)
    config.update(total_steps=2000,name='root_cause_reproduction')
    dataset=PROJECT_ROOT/config['algo']['fine_tune']['dataset']
    checkpoint=PROJECT_ROOT/config['bc_checkpoint']
    with CausalTrace(dataset,checkpoint,root/'trace') as observer:
        summary=train(config,root/'reproduction')
    checks={str(step):compare_checkpoints(root/'reproduction'/f'checkpoint_{step:07d}.pt',
                                        historical/f'checkpoint_{step:07d}.pt') for step in [1000,2000]}
    checks['1100']=compare_checkpoints(root/'trace/after_100.pt',PROJECT_ROOT/'experiments/phase3/stabilize_repro_3k/checkpoint_0001100.pt') if (PROJECT_ROOT/'experiments/phase3/stabilize_repro_3k/checkpoint_0001100.pt').exists() else dict(reference_unavailable=True)
    if not all(checks[str(step)]['bitwise_equal_all_model_tensors'] for step in [1000,2000]):
        raise AssertionError('Instrumentation changed the reproduced learner tensors')
    if observer.executed_actor_updates!=100 or len(observer.records)!=200:
        raise AssertionError('Did not capture precisely the first 100 updates of both actors')
    (root/'capture_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('CAPTURE PASS: first 100 updates traced; historical 1k/2k tensors bitwise identical',flush=True)
    return summary


def analyze(config_path,root,variants=None):
    config=load_config(config_path)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(config['torch_threads'])
    # Build probe metadata without installing any tracing patches or changing training data.
    trace=object.__new__(CausalTrace)
    trace.output=root/'trace'
    from rel301m.imitation.demonstrations import load_dataset
    trace.manifest,trace.episodes=load_dataset(PROJECT_ROOT/config['algo']['fine_tune']['dataset'])
    bc=torch.load(PROJECT_ROOT/config['bc_checkpoint'],map_location='cpu',weights_only=True)
    trace.split=bc['metadata']['episode_split']
    pre,_=MASAC.load(trace.output/'before_first_update.pt',device=device)
    trace.model=None;trace.subset_size=256
    # Match saved probe selections exactly; no replay samplers or global RNG used here.
    indices=json.loads((trace.output/'probe_indices.json').read_text())
    probes={}
    for split in ['train','validation']:
        probes[split]=[]
        import numpy as np
        for i in range(2):
            probes[split].append(tuple(torch.as_tensor(np.concatenate([trace.episodes[e][f'{key}{i}']
                for e in trace.split[split]])[indices[split]],device=device) for key in ['o','a']))
    reference=deepcopy(pre.actors);reference.requires_grad_(False)
    fixed,metadata=fixed_training_states(trace,device,pre.config['gamma'])
    bc_joint=torch.cat([bounded_mean(actor,fixed[f'o{i}']) for i,actor in enumerate(reference)],dim=-1)
    original100,_=MASAC.load(trace.output/'after_100.pt',device=device,load_optimizers=False)
    collapse_joint=torch.cat([bounded_mean(actor,fixed[f'o{i}']) for i,actor in enumerate(original100.actors)],dim=-1)
    eval_env=MultiAgentWrapper(make_two_arm_lift(PROJECT_ROOT/config['env_config'],seed=config['eval_seed']))
    eval_rng=deepcopy(eval_env._env.rng.bit_generator.state)
    previous=json.loads((root/'results.json').read_text()) if (root/'results.json').exists() else {}
    results=previous.get('forks',{})
    for result in results.values():
        result.setdefault('source_git_sha',previous.get('source_git_sha'))
    source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    try:
        for variant in (variants or VARIANTS):
            directory=root/variant
            directory.mkdir(parents=True,exist_ok=False)
            model,_=MASAC.load(trace.output/'before_first_update.pt',device=device)
            progress=TrainingProgress(100,config['seed'])
            before_hash=tree_hash(model.state_dict())
            before_q=tree_hash(dict(q1=model.q1.state_dict(),q2=model.q2.state_dict()))
            before_actor=tree_hash(model.actors.state_dict())
            records=[];evaluations={}
            for number in range(1,101):
                item=torch.load(trace.output/f'update_{number:03d}.pt',map_location='cpu',weights_only=True)
                batch={key:value.to(device) for key,value in item['batch'].items()}
                demo={key:value.to(device) for key,value in item['demo_batch'].items()}
                restore_rng(item['rng_before'])
                if number==1 or number in [2,11]: progress.set_phase(variant)
                update=diagnostic_update(model,batch,demo,variant)
                row=dict(actor_update_window=number,**policy_probe(model,reference,probes),
                         optimizer_counts=model.optimizer_counts(),gradients=update['components'])
                values,gradients=q_action_metrics(model,fixed['s'],bc_joint)
                row['Q_at_fixed_BC_mean']=float(values[2].mean())
                offset=0
                for i,width in enumerate(model.action_dims):
                    grad=gradients[2][:,offset:offset+width]
                    direction=collapse_joint[:,offset:offset+width]-bc_joint[:,offset:offset+width]
                    row[f'dQ_da_at_BC_norm_{i}']=float(grad.norm(dim=-1).mean())
                    row[f'dQ_alignment_with_recorded_collapse_{i}']=float(F.cosine_similarity(grad,direction,dim=-1).mean())
                    offset+=width
                records.append(row)
                progress.update(number)
                if number in [1,10,100]:
                    model.save(directory/f'after_{number:03d}.pt',dict(step=1000,diagnostic_only=True,variant=variant))
                    progress.set_phase('paired deterministic evaluation',episodes=config['eval_episodes'])
                    devices=list(range(torch.cuda.device_count())) if device=='cuda' else []
                    with torch.random.fork_rng(devices=devices):
                        episodes,summary=evaluate_policy(eval_env,model,config['eval_episodes'],config['eval_seed'],
                            initial_rng_state=eval_rng,episode_callback=progress.episode_completed)
                    write_episodes(directory/f'evaluation_{number:03d}.csv',episodes)
                    evaluations[str(number)]=summary
                    print(f'FORK {variant} update={number} ever={summary["success_rate"]} final={summary["final_success_rate"]}',flush=True)
            progress.finish('completed')
            results[variant]=dict(source_git_sha=source_sha,initial_model_sha256=before_hash,initial_Q_sha256=before_q,
                final_Q_sha256=tree_hash(dict(q1=model.q1.state_dict(),q2=model.q2.state_dict())),
                initial_actor_sha256=before_actor,final_actor_sha256=tree_hash(model.actors.state_dict()),
                evaluations=evaluations,final_drift={key:value for key,value in records[-1].items() if 'drift' in key or 'expert_mse' in key},
                optimizer_counts=model.optimizer_counts())
            if variant=='restored':
                check=compare_checkpoints(directory/'after_100.pt',trace.output/'after_100.pt')
                results[variant]['bitwise_restored_control']=check
                if not check['bitwise_equal_all_model_tensors']: raise AssertionError('Restored fork differs from captured learner')
            if variant=='A_fixed_critic' and before_q!=results[variant]['final_Q_sha256']:
                raise AssertionError('Fixed critic changed')
            if variant=='B_fixed_actor' and before_actor!=results[variant]['final_actor_sha256']:
                raise AssertionError('Fixed actor changed')
            (directory/'updates.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
            (directory/'summary.json').write_text(json.dumps(results[variant],indent=2)+'\n')
            (root/'fork_results.json').write_text(json.dumps(results,indent=2)+'\n')
        print('Starting fixed-critic action-ranking and return-to-go audit',flush=True)
        critic=json.loads((root/'critic_results.json').read_text()) if (root/'critic_results.json').exists() else critic_audit(trace,trace.output/'before_first_update.pt',PROJECT_ROOT/config['bc_checkpoint'],device)
        (root/'critic_results.json').write_text(json.dumps(critic,indent=2)+'\n')
    finally:
        eval_env.close()
    rows=[json.loads(line) for line in (trace.output/'actor_updates.jsonl').read_text().splitlines()]
    consolidated=dict(source_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        command=sys.argv,capture_source_git_sha=json.loads((root/'reproduction/metadata.json').read_text())['git_sha'],config=str(config_path),historical_capture=json.loads((root/'capture_checks.json').read_text()),
        first_update_checkpoint=str(trace.output/'before_first_update.pt'),first_100_exact_batches=str(trace.output),
        optimizer_validation=dict(max_delta_error=max(row['adam_expected_delta_max_error'] for row in rows),
            first_actor_states_empty=all(row['optimizer_state_empty_before'] for row in rows if row['actor_update']==1),
            actor_lrs=sorted(set(row['optimizer_lr'] for row in rows)),
            all_critic_and_teammate_grads_clean=all(row['critic_grad_clean'] and row['teammate_grad_clean'] for row in rows)),
        replay_composition=json.loads((trace.output/'replay_composition.json').read_text()),
        forks=results,critic=critic,
        conclusion='ROOT CAUSE NOT YET ESTABLISHED',
        conclusion_note='Review interventions, both-agent drift and paired evaluation before assigning a causal mechanism.')
    (root/'results.json').write_text(json.dumps(consolidated,indent=2,allow_nan=False)+'\n')
    print('ANALYSIS COMPLETE: review results before claiming root cause',flush=True)
    return consolidated


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/experiment/stabilize_repro.yaml'))
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--historical-dir',type=Path,default=Path('experiments/phase3/bc_finetune_10k_seed0'))
    parser.add_argument('--phase',choices=['capture','analyze','all'],default='all')
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,help='Run only these new diagnostic forks; existing directories are never overwritten')
    args=parser.parse_args()
    config=(PROJECT_ROOT/args.config).resolve()
    root=(PROJECT_ROOT/args.output_root).resolve()
    if args.phase in ['capture','all']:
        root.mkdir(parents=True,exist_ok=False)
        capture(config,root,(PROJECT_ROOT/args.historical_dir).resolve())
    if args.phase in ['analyze','all']:
        analyze(config,root,args.variants)


if __name__=='__main__':
    main()
