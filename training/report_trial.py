"""Audit and plot the completed bounded trial; never launch simulation/training."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch

from training.train import assert_same_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    from training.paths import REFERENCE_RUN
    parser.add_argument('--run',type=Path,default=REFERENCE_RUN)
    args=parser.parse_args()
    root=args.run
    summary=json.loads((root/'summary.json').read_text())
    rows=[json.loads(x) for x in (root/'train/progress.jsonl').read_text().splitlines()]
    episodes=[json.loads(x) for x in (root/'train/episodes.jsonl').read_text().splitlines()]
    assert len(rows)==100 and [r['iteration'] for r in rows]==list(range(1,101))
    n=summary['plan']['selected_num_envs']
    penalties=('body_ang_vel','angular_momentum','dof_pos_limits','action_rate_l2',
               'foot_clearance','foot_swing_height','foot_slip','self_collisions','head_pose_bias')
    for i,r in enumerate(rows,1):
        assert r['env_steps']==i*n*24 and r['vector_steps']==i*24
        assert r['optimization']['grad_norm_after']<=.50001
        assert r['normalization']['checkpoint_counts']=={'actor':12288,'critic':12288}
        for name in penalties:
            assert r['episodes']['rollout_reward_terms'][name]['mean_step_contribution']<=1e-9
    assert all('nan_state' not in e['end_terms'] for e in episodes)
    source=torch.load(summary['plan']['initial_checkpoint'],map_location='cpu',weights_only=True)
    checkpoint_reports=[]
    for iteration in (25,50,75,100):
        checkpoint=torch.load(root/f'train/iteration_{iteration:04d}.pt',map_location='cpu',weights_only=True)
        assert_same_state(checkpoint['normalizer'],source['normalizer'])
        assert checkpoint['training_state']['common_step_counter']==iteration*24
        assert checkpoint['metrics']['update']['optimizer_after']['step']==iteration*12
        checkpoint_reports.append({'iteration':iteration,'optimizer_step':iteration*12,
                                   'normalizer_exactly_unchanged':True})
    def window(lo,hi):
        selected=[e for e in episodes if lo<=e['iteration']<=hi]
        records=rows[lo-1:hi]
        return {'completed_episodes':len(selected),
                'mean_episode_return':statistics.mean(e['return'] for e in selected),
                'mean_episode_duration_s':statistics.mean(e['duration_s'] for e in selected),
                'fall_fraction':sum('fell_over' in e['end_terms'] for e in selected)/len(selected),
                'mean_step_reward':statistics.mean(r['mean_step_reward'] for r in records),
                'weighted_reward_terms':{term:statistics.mean(
                    r['episodes']['rollout_reward_terms'][term]['mean_step_contribution'] for r in records)
                    for term in rows[0]['episodes']['rollout_reward_terms']}}
    summary['audit']={'completed_episodes':len(episodes),'penalty_signs_verified':True,
                      'nan_state_terminations':0,'normalizer_checkpoints':checkpoint_reports,
                      'first_10_iterations':window(1,10),'last_10_iterations':window(91,100),
                      'maximum_approx_kl':max(r['optimization']['approx_kl'] for r in rows),
                      'maximum_action_abs':max(r['optimization']['action_max_abs'] for r in rows),
                      'maximum_normalized_abs':{group:max(r['normalization']['sampling_inputs'][group]
                             ['normalized_max_abs'] for r in rows) for group in ('actor','critic')},
                      'maximum_fraction_normalized_abs_gt5':{group:max(r['normalization']['sampling_inputs'][group]
                             ['fraction_abs_gt_5'] for r in rows) for group in ('actor','critic')},
                      'minimum_free_vram_bytes':min(r['performance']['cuda_device_free_bytes'] for r in rows),
                      'training_seconds_excluding_io_and_eval':sum(
                             r['performance']['iteration_seconds_excluding_io'] for r in rows),
                      'probe_transitions':288,'benchmark_transitions':5*24*(64+128+256),
                      'total_transitions_including_probes':100*24*n+288+5*24*(64+128+256)}
    for iteration in (50,100):
        text=(root/f'evaluation/play_{iteration:04d}.log').read_text()
        assert '[play] PASS:' in text and "'steps': 128" in text
    summary['audit']['viewer_replays_passed']=[50,100]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    x=[r['iteration'] for r in rows]
    axes[0,0].plot(x,[r['mean_step_reward'] for r in rows])
    axes[0,0].set_title('Training reward per step')
    axes[0,1].plot(x,[r['episodes']['mean_duration_s'] for r in rows])
    axes[0,1].set_title('Completed episode duration (s)')
    for name in ('track_linear_velocity','track_angular_velocity','upright'):
        axes[0,2].plot(x,[r['episodes']['rollout_reward_terms'][name]['mean_step_contribution'] for r in rows],label=name)
    axes[0,2].set_title('Weighted task reward per step')
    axes[0,2].legend(fontsize=7)
    axes[1,0].plot(x,[r['optimization']['approx_kl'] for r in rows])
    axes[1,0].set_title('PPO approximate KL')
    evaluations=summary['evaluations']
    initial=evaluations[0]['aggregates']['initial']
    for axis,metric,title in ((axes[1,1],'mean_recorded_return','Recorded return (10 s eval cap)'),
                               (axes[1,2],'mean_recorded_duration_s','Recorded duration (10 s cap)')):
        axis.plot([0]+[e['iteration'] for e in evaluations],
                  [initial[metric]]+[e['aggregates']['trained'][metric] for e in evaluations],marker='o')
        axis.axhline(initial[metric],color='gray',linestyle='--',label='initial policy')
        axis.set_title(title+'; 4 seeds')
        axis.legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel('Iteration')
        axis.grid(alpha=.2)
    fig.suptitle(f'Bounded PPO trial: {n} envs x 24 steps x 100 iterations; frozen normalization')
    fig.savefig(root/'training_curves.png',dpi=160)
    plt.close(fig)
    summary['curves']='training_curves.png'
    ranking=sorted(summary['evaluations'],key=lambda e:e['aggregates']['trained']['mean_recorded_return'],reverse=True)
    summary['highest_recorded_eval_return_iteration']=ranking[0]['iteration']
    summary['evaluation_caveats']=['Only four fixed seeds; not a reliable-walking certification.',
                                  '10 second cap is an evaluation cutoff, not a completion or success.',
                                  'All four stage checkpoints retained; final iteration is not automatically best.']
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary['audit'],indent=2))


if __name__=='__main__':
    main()
