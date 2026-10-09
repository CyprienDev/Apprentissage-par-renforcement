import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from rl_arena.arena_api import (
    Action, Observation, SelfState, VisualDetection, ObjectKind, InventoryItem,
    EpisodeResult, EpisodeEndReason,
)
from student_agent import MyAgent, MODEL_VERSION, state_representation


def observation(vision=(), inventory=(), tick=0):
    return Observation(tick, SelfState(1, .8, 1, 1, 70, 0, 0, 0, (), inventory, None), vision, (), (), ())


class StudentTests(unittest.TestCase):
    def test_q_learning_update_uses_maximum_next_action_value(self):
        agent = MyAgent(load=False, adaptive=True)
        agent.q_table = {'before': {'move': 2.}, 'after': {'rest': 3., 'move': 7., 'unavailable': 100.}}
        agent.learn('before', 'move', 1., 'after', ['rest', 'move'])
        self.assertAlmostEqual(agent.q_table['before']['move'], 2. + .15 * (1. + .98 * 7. - 2.))
        self.assertEqual(agent.updates, 1)

    def test_terminal_transition_does_not_bootstrap(self):
        agent = MyAgent(load=False, adaptive=True)
        agent.q_table = {'after': {'move': 1000.}}
        agent.learn('before', 'rest', -10., 'after', ['move'], terminal=True)
        self.assertAlmostEqual(agent.q_table['before']['rest'], -1.5)

    def test_frozen_unknown_state_does_not_create_table_rows(self):
        agent = MyAgent(load=False)
        for tick in range(10):
            agent.act(observation(tick=tick))
        agent.on_episode_end(EpisodeResult(EpisodeEndReason.ELIMINATED, 10))
        agent.learn('s', 'a', 10., terminal=True)
        self.assertEqual(agent.q_table, {})
        self.assertEqual(agent.updates, 0)

    def test_greedy_action_is_not_overridden_by_need_or_enemy_rules(self):
        agent = MyAgent(load=False)
        obs = observation((VisualDetection(7, ObjectKind.AGENT, 1., 0., .2),))
        obs = replace(obs, self_state=replace(obs.self_state, energy=.01, hydration=.01))
        actions = agent.candidates(obs)
        state = state_representation(obs)
        for wanted, action in actions.items():
            agent.reset()
            agent.q_table = {state: {wanted: 100.}}
            self.assertEqual(agent.act(obs), action)

    def test_references_and_slots_are_not_part_of_learned_keys(self):
        agent = MyAgent(load=False)
        first = observation((VisualDetection(123, ObjectKind.FOOD, .2, .3, .2),),
                            (InventoryItem(1, ObjectKind.DRINK, .3),))
        second = observation((VisualDetection(999, ObjectKind.FOOD, .2, .3, .2),),
                             (InventoryItem(6, ObjectKind.DRINK, .3),))
        self.assertEqual(state_representation(first), state_representation(second))
        self.assertEqual(set(agent.candidates(first)), set(agent.candidates(second)))
        self.assertNotEqual(agent.candidates(first)['pickup:FOOD:0'], agent.candidates(second)['pickup:FOOD:0'])

    def test_save_load_preserves_table(self):
        agent = MyAgent(load=False, adaptive=True)
        agent.learn('s', 'rest', 5., terminal=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            agent.save(path)
            restored = MyAgent(path)
            self.assertEqual(restored.q_table, agent.q_table)
            self.assertFalse(restored.adaptive)
            self.assertEqual(json.loads(path.read_text())['version'], MODEL_VERSION)

    def test_old_network_and_hybrid_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            for version in (3, 5):
                path.write_text(json.dumps({'version': version, 'weights': []}))
                with self.assertRaises(ValueError):
                    MyAgent(path)

    def test_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            path.write_text(json.dumps({'version': MODEL_VERSION, 'q_table': {'s': {'a': float('nan')}}}))
            with self.assertRaises(ValueError):
                MyAgent(path)

    def test_reset_retains_table_and_frozen_mode(self):
        agent = MyAgent(load=False)
        agent.q_table = {'s': {'move': 1.}}
        agent.reset()
        self.assertEqual(agent.q_table, {'s': {'move': 1.}})
        self.assertFalse(agent.adaptive)

    def test_same_tick_is_not_learned_twice(self):
        agent = MyAgent(load=False, adaptive=True)
        agent.act(observation())
        agent.act(observation(tick=1))
        updates = agent.updates
        result = agent.act(observation(tick=1))
        self.assertEqual(agent.updates, updates)
        self.assertEqual(result, agent.last_action)

    def test_warmup_can_learn_then_freeze(self):
        agent = MyAgent(load=False)
        agent.set_training_enabled(True)
        for tick in range(8):
            agent.act(observation(tick=tick))
        self.assertGreater(agent.updates, 0)
        agent.set_training_enabled(False)
        before = deepcopy(agent.q_table)
        agent.reset()
        for tick in range(8):
            agent.act(observation(tick=tick))
        agent.on_episode_end(EpisodeResult(EpisodeEndReason.ELIMINATED, 8))
        self.assertEqual(agent.q_table, before)
        self.assertEqual(agent.epsilon, 0.)

    def test_terminal_callback_is_idempotent(self):
        agent = MyAgent(load=False, adaptive=True)
        agent.act(observation())
        result = EpisodeResult(EpisodeEndReason.ELIMINATED, 1)
        agent.on_episode_end(result)
        before = deepcopy(agent.q_table)
        agent.on_episode_end(result)
        self.assertEqual(agent.q_table, before)
        self.assertEqual(agent.episodes, 1)

    def test_submission_contains_only_agent_and_small_table(self):
        from build_submission import build
        with ZipFile(build()) as archive:
            self.assertEqual(set(archive.namelist()), {'student_agent.py', 'artifacts/policy.json'})
            self.assertLessEqual(archive.getinfo('artifacts/policy.json').file_size, 10_000_000)

    def test_oversized_data_is_rejected(self):
        import build_submission
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'student_agent.py').write_text('')
            (root / 'artifacts').mkdir()
            with (root / 'artifacts' / 'policy.json').open('wb') as handle:
                handle.truncate(10_000_001)
            with patch.object(build_submission, '__file__', str(root / 'build_submission.py')):
                with self.assertRaises(ValueError):
                    build_submission.build()


if __name__ == '__main__':
    unittest.main()
