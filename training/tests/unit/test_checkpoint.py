import copy
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from training.checkpoint import capture_training_state, restore_training_state
from training.ppo import optimizer_status
from training.tests.unit import test_train as fixtures
from training.train import assert_same_state, cpu_snapshot, run_training


class CheckpointTests(unittest.TestCase):
    def test_disk_restore_keeps_rng_moments_counters_and_next_update(self):
        import tempfile
        from pathlib import Path
        actor, critic, norm, optimizer = fixtures.TrainTests().make_components()
        records = run_training(fixtures.LoopEnv(), actor, critic, norm, optimizer,
                               iterations=1, steps=4, epochs=2, minibatch_size=3)
        env = SimpleNamespace(num_envs=1, common_step_counter=4)
        source = cpu_snapshot({"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(),
                              "normalizer": norm.checkpoint(), "optimizer_state_dict": optimizer.state_dict(),
                              "training_state": capture_training_state(records[-1], env)})
        expected = (torch.rand(4), random.random(), np.random.rand(4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            torch.save(source,path)
            source = torch.load(path, weights_only=True)
        a, c, n, o = fixtures.TrainTests().make_components()
        a.load_state_dict(source['actor_state_dict'])
        c.load_state_dict(source['critic_state_dict'])
        n.load_state_dict(source['normalizer']['state_dict'])
        o.load_state_dict(copy.deepcopy(source['optimizer_state_dict']))
        restored_env = fixtures.LoopEnv()
        counters = restore_training_state(source, restored_env,a,c,n,o,normalization_mode='frozen')
        torch.testing.assert_close(torch.rand(4),expected[0],rtol=0,atol=0)
        self.assertEqual(random.random(),expected[1])
        np.testing.assert_array_equal(np.random.rand(4),expected[2])
        self.assertEqual(restored_env.common_step_counter,4)
        self.assertEqual(counters,{'iteration':1,'env_steps':4,'vector_steps':4})
        assert_same_state(o.state_dict(),source['optimizer_state_dict'])
        # Both fresh simulators reset explicitly; identical RNG must give the same
        # next rollout and update. This does not assert physical-state resume.
        restore_training_state(source,restored_env,a,c,n,o,normalization_mode='frozen')
        first = run_training(restored_env,a,c,n,o,iterations=1,steps=4,epochs=2,minibatch_size=3,
                             start_iteration=1,start_env_steps=4,start_vector_steps=4)
        restore_training_state(source,env,actor,critic,norm,optimizer,normalization_mode='frozen')
        run_training(fixtures.LoopEnv(),actor,critic,norm,optimizer,iterations=1,steps=4,epochs=2,minibatch_size=3)
        assert_same_state(a.state_dict(),actor.state_dict())
        assert_same_state(c.state_dict(),critic.state_dict())
        assert_same_state(o.state_dict(),optimizer.state_dict())
        self.assertEqual(first[0]['iteration'],2)
        self.assertEqual(first[0]['env_steps'],8)
        self.assertEqual(optimizer_status(o)['step'],8)

    def test_invalid_resume_is_rejected(self):
        a,c,n,o = fixtures.TrainTests().make_components()
        with self.assertRaisesRegex(ValueError,'version-1'):
            restore_training_state({},SimpleNamespace(num_envs=1),a,c,n,o,normalization_mode='frozen')
