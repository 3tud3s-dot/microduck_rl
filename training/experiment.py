"""One bounded local trial: resume probe, scale benchmark, 100 PPO rounds and evals."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import shutil
import sys

import torch

from training import train
from training.paths import INITIAL_CHECKPOINT


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_progress(directory):
    return [json.loads(line) for line in (directory/'progress.jsonl').read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=INITIAL_CHECKPOINT)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error('--checkpoint must exist')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    (root/'evaluation').mkdir()
    initial = root/'initial.pt'
    shutil.copyfile(args.checkpoint, initial)
    original_hash = hashlib.sha256(initial.read_bytes()).hexdigest()
    original = torch.load(initial, map_location='cpu', weights_only=True)
    plan = {'gpu': torch.cuda.get_device_name(0), 'training_iterations': 100, 'steps': 24,
            'normalization': 'frozen calibrated statistics', 'learning_rate': 1e-4,
            'epochs': 3, 'minibatches_per_epoch': 4, 'evaluation_iterations': [25,50,75,100],
            'evaluation_seeds': [100,101,102,103], 'evaluation_max_steps': 500,
            'scale_benchmarks': {'environments':[64,128,256], 'iterations_each':5},
            'resume_probe': '4 envs x 2 iterations + restored 1 iteration',
            'initial_checkpoint': str(initial), 'initial_sha256': original_hash,
            'maximum_training_transitions': 100*24*256,
            'formal_long_training': False}
    write_json(root/'plan.json',plan)

    def execute(directory, n, iterations, checkpoint=initial, resume=False, steps=24,
                save_every=1, callback=None):
        argv = ['--checkpoint',str(checkpoint),'--output-dir',str(directory),
                '--num-envs',str(n),'--steps',str(steps),'--iterations',str(iterations),
                '--epochs','3','--minibatch-size',str(n*steps//4),
                '--normalization','frozen','--save-every',str(save_every)]
        if resume:
            argv.append('--resume')
        else:
            argv += ['--lr','0.0001']
        train.main(argv,on_iteration=callback)
        gc.collect()
        torch.cuda.empty_cache()

    execute(root/'checks/resume_before',4,2)
    before_path=root/'checks/resume_before/iteration_0002.pt'
    execute(root/'checks/resume_after',4,1,checkpoint=before_path,resume=True)
    before=torch.load(before_path,map_location='cpu',weights_only=True)
    after=torch.load(root/'checks/resume_after/iteration_0003.pt',map_location='cpu',weights_only=True)
    for name in ('actor','critic'):
        train.assert_same_state(before[name+'_state_dict'],after['sampling_policy'][name+'_state_dict'])
    train.assert_same_state(before['normalizer'],after['normalizer'])
    assert after['metrics']['env_steps']==288 and after['metrics']['vector_steps']==72
    assert after['metrics']['update']['optimizer_before']['step']==24
    assert after['metrics']['update']['optimizer_after']['step']==36
    assert after['training_state']['common_step_counter']==72
    resume_validation={'passed':True,'model_and_stats_restored_exactly':True,
               'optimizer_state_checked_before_collection':True,'adam_steps':[0,24,36],
               'iterations':[2,3],'vector_steps':[48,72],'env_steps':[192,288],
               'common_step_counter':72,'simulator_reset_on_resume':True,
               'partial_episodes_discarded':before['training_state']['unfinished_episodes_discarded_on_resume'],
               'rng_restore_and_identical_next_update_test':'training.tests.unit.test_checkpoint'}
    write_json(root/'validation.json',{'resume_probe':resume_validation})
    print('[experiment] Resume probe PASS',flush=True)
    del before,after

    benchmarks=[]
    for n in (64,128,256):
        directory=root/f'checks/benchmark_{n}'
        execute(directory,n,5,save_every=5)
        rows=read_progress(directory)
        final=torch.load(directory/'iteration_0005.pt',map_location='cpu',weights_only=True)
        train.assert_same_state(final['normalizer'],original['normalizer'])
        entry={'num_envs':n,'median_samples_per_second':statistics.median(
                   row['performance']['samples_per_second_excluding_io'] for row in rows[1:]),
               'median_iteration_seconds':statistics.median(
                   row['performance']['iteration_seconds_excluding_io'] for row in rows[1:]),
               'minimum_free_vram_bytes':min(row['performance']['cuda_device_free_bytes'] for row in rows),
               'maximum_approx_kl':max(row['optimization']['approx_kl'] for row in rows),
               'normalizer_unchanged':True}
        benchmarks.append(entry)
        write_json(root/'validation.json',{'resume_probe':resume_validation,'scale_benchmarks':benchmarks})
        print('[experiment] Benchmark '+json.dumps(entry),flush=True)
        del final
    # Prefer the smaller scale if it is within 10% of the fastest measured one.
    feasible=[entry for entry in benchmarks if entry['minimum_free_vram_bytes']>=2*1024**3]
    if not feasible:
        raise RuntimeError('no tested scale retained 2 GiB free VRAM')
    fastest=max(entry['median_samples_per_second'] for entry in feasible)
    selected=min(entry['num_envs'] for entry in feasible
                 if entry['median_samples_per_second']>=.9*fastest)
    plan.update({'selected_num_envs':selected,'training_transitions':selected*24*100,
                 'selection':'smallest tested scale within 10% of fastest, >=2 GiB free VRAM'})
    write_json(root/'plan.json',plan)
    print('[experiment] Selected '+str(selected)+' environments; fixed 100-round budget',flush=True)

    evaluations=[]
    def external(module, arguments, log):
        with log.open('w') as stream:
            subprocess.run([sys.executable,'-u','-m',module,*arguments],stdout=stream,
                           stderr=subprocess.STDOUT,check=True)

    def periodic(record,*unused):
        iteration=record['iteration']
        if iteration % 25:
            return
        checkpoint=root/f'train/iteration_{iteration:04d}.pt'
        evaluation=root/f'evaluation/iteration_{iteration:04d}'
        print(f'[experiment] Evaluating iteration {iteration} on seeds 100..103',flush=True)
        external('training.evaluate',['--initial',str(initial),'--trained',str(checkpoint),
                 '--seed','100','--episodes','4','--max-steps','500','--output-dir',str(evaluation)],
                 root/f'evaluation/iteration_{iteration:04d}.log')
        (root/f'evaluation/iteration_{iteration:04d}.log').rename(evaluation/'evaluation.log')
        report=json.loads((evaluation/'summary.json').read_text())
        evaluations.append({'iteration':iteration,'aggregates':report['aggregates'],
                            'episodes':report['episodes'],'paired_comparisons':report['paired_comparisons']})
        write_json(root/'evaluation/summary.json',evaluations)
        print('[experiment] Eval '+json.dumps(evaluations[-1]['aggregates']),flush=True)
        if iteration in (50,100):
            print(f'[experiment] Native GPU viewer replay: iteration {iteration}, 128 steps',flush=True)
            external('training.play',['--checkpoint',str(checkpoint),'--steps','128'],
                     root/f'evaluation/play_{iteration:04d}.log')

    execute(root/'train',selected,100,save_every=25,callback=periodic)
    final_path=root/'train/iteration_0100.pt'
    final=torch.load(final_path,map_location='cpu',weights_only=True)
    train.assert_same_state(final['normalizer'],original['normalizer'])
    assert hashlib.sha256(initial.read_bytes()).hexdigest()==original_hash
    rows=read_progress(root/'train')
    assert len(rows)==100 and final['iteration']==100
    assert final['metrics']['update']['optimizer_after']['step']==1200
    # Reuse the original normalized ONNX path, validated against actual raw inputs.
    external('training.export',['--checkpoint',str(final_path),'--output-dir',str(root/'onnx')],
             root/'logs/export.log')
    (root/'logs/export.log').rename(root/'onnx/export.log')
    write_json(root/'summary.json',{'status':'completed','plan':plan,'benchmarks':benchmarks,
               'resume_validation':'validation.json#resume_probe','evaluations':evaluations,
               'optimizer_final_step':1200,'normalizer_unchanged':True,'old_initial_unchanged':True,
               'first_10_mean_step_reward':statistics.mean(r['mean_step_reward'] for r in rows[:10]),
               'last_10_mean_step_reward':statistics.mean(r['mean_step_reward'] for r in rows[-10:]),
               'final_checkpoint':str(final_path),'onnx_validation':'onnx/validation.json',
               'claim':'Trial completed; judge walking from evaluation and playback, not training loss.'})
    print('[experiment] All bounded stages completed: '+str(root),flush=True)


if __name__=='__main__':
    main()
