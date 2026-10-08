import tempfile
import unittest
from math import cos, sin
from pathlib import Path
from rl_arena.arena_api import Action, Observation, SelfState, VisualDetection, ObjectKind
from student_agent import MyAgent


def observation(vision=(), inventory=(), tick=0):
    return Observation(tick, SelfState(1, .8, 1, 1, 70, 0, 0, 0, (), inventory, None), vision, (), (), ())


class StudentTests(unittest.TestCase):
    def test_terminal_does_not_bootstrap(self):
        agent = MyAgent(load=False)
        state = (0,)*9
        agent.q[state] = [10000.]*5
        agent.learn(state, 0, -20, state, True, alpha=1)
        self.assertEqual(agent.q[state][0], -20)

    def test_save_load_and_reset_preserve_learning(self):
        agent = MyAgent(load=False)
        state = (0,)*9
        agent.learn(state, 0, 1, state, False)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.json'
            agent.save(path)
            restored = MyAgent(path)
            restored.reset()
            self.assertEqual(agent.q, restored.q)

    def test_unknown_properties_and_all_strategies(self):
        agent = MyAgent(load=False)
        obs = observation((VisualDetection(1, ObjectKind.UNKNOWN, 2, 0, .2),))
        for strategy in range(5):
            self.assertIsInstance(agent.action_for(obs, strategy), Action)
        self.assertEqual(agent.q, {})

    def test_no_stale_reference(self):
        agent = MyAgent(load=False)
        first = observation((VisualDetection(123, ObjectKind.FOOD, .5, 0, .2),))
        self.assertEqual(agent.act(first).interaction.target_ref, 123)
        self.assertIsNone(agent.act(observation(tick=1)).interaction)

    def test_rest_overridden_near_enemy(self):
        agent = MyAgent(load=False)
        obs = observation((VisualDetection(7, ObjectKind.AGENT, 2, 0, .2),))
        self.assertFalse(agent.action_for(obs, 3).rest)

    def test_flee_moves_away_for_every_bearing(self):
        for bearing in (-1., 0., 1.):
            agent = MyAgent(load=False)
            obs = observation((VisualDetection(7, ObjectKind.AGENT, 2, bearing, .2),))
            action = agent.action_for(obs, 2)
            self.assertIsNone(action.rotation)
            self.assertLess(action.movement.forward*cos(bearing) + action.movement.right*sin(bearing), 0)

    def test_learns_from_next_observation_without_external_reward(self):
        agent = MyAgent(load=False,seed=4)
        agent.act(observation(tick=0))
        self.assertEqual(agent.updates,0)
        agent.act(observation(tick=1))
        self.assertGreaterEqual(agent.updates,5)
        self.assertEqual(agent.online_transitions,1)
        self.assertTrue(agent.q)

    def test_frozen_agent_does_not_change_q(self):
        agent = MyAgent(load=False,adaptive=False)
        agent.act(observation(tick=0))
        agent.act(observation(tick=1))
        self.assertEqual(agent.q,{})
        self.assertEqual(agent.online_transitions,0)

    def test_duplicate_tick_is_idempotent(self):
        agent = MyAgent(load=False)
        first = agent.act(observation(tick=0))
        self.assertEqual(agent.act(observation(tick=0)),first)
        agent.act(observation(tick=1))
        updates = agent.updates
        agent.act(observation(tick=1))
        self.assertEqual(agent.updates,updates)

    def test_resource_memory_and_replay_reset_but_q_persists(self):
        agent = MyAgent(load=False)
        agent.act(observation((VisualDetection(7,ObjectKind.DRINK,4.,.1,.2),),tick=0))
        agent.act(observation(tick=1))
        self.assertTrue(agent.resources)
        self.assertTrue(agent.replay)
        learned = dict(agent.q)
        agent.reset()
        self.assertEqual(agent.resources,[])
        self.assertEqual(len(agent.replay),0)
        self.assertEqual(agent.q,learned)

    def test_terminal_notification_only_updates_once(self):
        from rl_arena.arena_api import EpisodeResult,EpisodeEndReason
        agent = MyAgent(load=False)
        agent.act(observation(tick=0))
        result = EpisodeResult(EpisodeEndReason.ELIMINATED,1)
        agent.on_episode_end(result)
        updates = agent.updates
        agent.on_episode_end(result)
        self.assertEqual(agent.updates,updates)
        self.assertEqual(agent.episodes,1)

    def test_learned_outlier_cannot_suppress_visible_resources(self):
        agent = MyAgent(load=False,epsilon=0)
        obs = observation((VisualDetection(7,ObjectKind.DRINK,3.,0.,.2),))
        state = agent._state(obs)
        agent.q[state] = [1000000.,-1000000.,0.,0.,0.]
        self.assertEqual(agent.choose(obs),1)

    def test_identifies_energy_cost_within_eight_observations(self):
        from dataclasses import replace
        agent = MyAgent(load=False,epsilon=0)
        for tick in range(9):
            obs = observation(tick=tick)
            obs = replace(obs,self_state=replace(obs.self_state,energy=.8-.01*tick,speed=1.7))
            agent.act(obs)
        self.assertEqual(agent.dynamics_samples,8)
        self.assertGreater(agent._travel_speed(obs),1.7)
        agent.reset()
        self.assertEqual(agent.dynamics_samples,0)

    def test_starvation_does_not_trigger_futile_flight(self):
        from dataclasses import replace
        from rl_arena.arena_api import Event,EventKind
        agent = MyAgent(load=False)
        obs = observation()
        obs = replace(obs,self_state=replace(obs.self_state,hydration=.1),
                      events=(Event(EventKind.DAMAGE_RECEIVED,.003),))
        agent.act(obs)
        self.assertEqual(agent.threat_until,-1)
        obs = replace(obs,tick=1,events=(Event(EventKind.DAMAGE_RECEIVED,.18),))
        agent.act(obs)
        self.assertGreater(agent.threat_until,1)

    def test_follows_new_open_door_to_explore_occluded_resources(self):
        agent = MyAgent(load=False)
        obs = observation((VisualDetection(33,ObjectKind.DOOR,2.,0.,1.,{'open':True}),))
        action = agent.act(obs)
        self.assertGreater(action.movement.forward,0)

    def test_transformed_scenario_is_reproducible(self):
        from experiments import make_environment
        first = make_environment(1234,10,'stress')
        second = make_environment(1234,10,'stress')
        self.assertEqual(first._last_observations,second._last_observations)

    def test_saturated_resource_is_not_a_target(self):
        from rl_arena.arena_api import InventoryItem
        inventory = tuple(InventoryItem(i,ObjectKind.FOOD,.3) for i in range(2))
        agent = MyAgent(load=False,epsilon=0)
        obs = observation((VisualDetection(7,ObjectKind.FOOD,2.,0.,.2),),inventory)
        agent.act(obs)
        self.assertNotEqual(agent.choose(obs),1)
        self.assertEqual(agent._state(obs)[6],0)

    def test_full_inventory_makes_room_for_water_without_recollecting_ammo(self):
        from dataclasses import replace
        from rl_arena.arena_api import InventoryItem,Drop,Pickup
        inventory = (InventoryItem(0,ObjectKind.AMMUNITION,.2),) + tuple(
            InventoryItem(i,ObjectKind.FOOD,.3) for i in range(1,7))
        agent = MyAgent(load=False)
        obs = observation((VisualDetection(7,ObjectKind.DRINK,.6,0.,.2),),inventory)
        obs = replace(obs,self_state=replace(obs.self_state,hydration=.3))
        self.assertIsInstance(agent.action_for(obs,1).interaction,Drop)
        next_obs = observation((VisualDetection(8,ObjectKind.AMMUNITION,.3,0.,.2),
                               VisualDetection(9,ObjectKind.DRINK,.6,.1,.2)),inventory[1:],tick=1)
        action = agent.action_for(next_obs,1)
        self.assertIsInstance(action.interaction,Pickup)
        self.assertEqual(action.interaction.target_ref,9)

    def test_bandage_is_kept_for_bleeding_not_wasted_on_low_health(self):
        from dataclasses import replace
        from rl_arena.arena_api import InventoryItem,Injury,InjuryKind,Ingest
        agent = MyAgent(load=False)
        obs = observation(inventory=(InventoryItem(0,ObjectKind.MEDICINE,.2,{'medical_type':'bandage'}),))
        obs = replace(obs,self_state=replace(obs.self_state,health=.6))
        self.assertNotIsInstance(agent.action_for(obs,0).interaction,Ingest)
        obs = replace(obs,self_state=replace(obs.self_state,injuries=(Injury(InjuryKind.WOUND,.3,.2),)))
        self.assertIsInstance(agent.action_for(obs,0).interaction,Ingest)

    def test_bootstrap_excludes_impossible_actions(self):
        agent = MyAgent(load=False)
        state = (0,)*9
        agent.q[state] = [1.,100000.,100000.,100000.,2.]
        agent.learn(state,0,0.,state,False,alpha=1.,gamma=.9)
        self.assertAlmostEqual(agent.q[state][0],1.8)

    def test_recovers_instead_of_repeated_short_rest(self):
        from dataclasses import replace
        agent = MyAgent(load=False)
        for tick,energy in enumerate((.15,.3,.5)):
            obs = observation(tick=tick)
            obs = replace(obs,self_state=replace(obs.self_state,energy=energy))
            self.assertTrue(agent.action_for(obs,0).rest)
        obs = observation(tick=3)
        self.assertFalse(agent.action_for(obs,0).rest)

    def test_cold_submission_never_loads_adjacent_checkpoint(self):
        from build_submission import build
        from zipfile import ZipFile
        from rl_arena.environment import TrainingEnvironment
        from rl_arena.config import EnvironmentConfig
        archive = build(cold=True)
        with tempfile.TemporaryDirectory() as d:
            with ZipFile(archive) as z:
                source = z.read('student_agent.py').decode('utf-8')
                self.assertNotIn('artifacts/policy.json',z.namelist())
            directory = Path(d)
            (directory/'artifacts').mkdir()
            (directory/'artifacts'/'policy.json').write_text('INVALID JSON',encoding='utf-8')
            namespace = {'__file__':str(directory/'student_agent.py')}
            exec(compile(source,'student_agent.py','exec'),namespace)
            agent = namespace['MyAgent']()
            self.assertEqual(agent.q,{})
            env = TrainingEnvironment(EnvironmentConfig(max_ticks=10))
            env.register_agent('student',agent)
            env.add_default_bots()
            env.reset(seed=3)
            for _ in range(10):
                env.step({})
            self.assertGreater(agent.online_transitions,0)
