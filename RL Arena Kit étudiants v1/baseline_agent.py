"""Agent de survie : Q-learning sur des stratégies, observations publiques seules."""
from __future__ import annotations
import json
from pathlib import Path
from random import Random
from math import pi, cos, sin
from rl_arena.arena_api import (Action, Movement, Rotation, ObjectKind, EventKind,
                                SoundKind, Ingest, Pickup, Manipulate)

MODEL_PATH = Path(__file__).with_name('artifacts') / 'policy.json'
STRATEGIES = ('explore', 'forage', 'flee', 'rest', 'scan')
BLOCKERS = {ObjectKind.WALL, ObjectKind.OBSTACLE, ObjectKind.MOVING_OBSTACLE,
            ObjectKind.HAZARD, ObjectKind.WATER}
RESOURCES = {ObjectKind.FOOD, ObjectKind.DRINK, ObjectKind.MEDICINE}


def state_representation(obs):
    s = obs.self_state
    enemy = min((d.distance for d in obs.vision if d.kind is ObjectKind.AGENT), default=99)
    return (int(s.health < .5), int(s.energy < .25), int(s.energy < .6),
            int(s.hydration < .5), int(s.satiety < .5),
            0 if enemy > 7 else 1 if enemy > 3 else 2,
            int(any(d.kind in RESOURCES for d in obs.vision)),
            int(bool(obs.touch)), int(any(e.kind is EventKind.DAMAGE_RECEIVED for e in obs.events)))


class MyAgent:
    def __init__(self, model_path=MODEL_PATH, seed=0, load=True):
        self.rng = Random(seed)
        self.q = {}
        self.episodes = 0
        if load and Path(model_path).exists():
            data = json.loads(Path(model_path).read_text(encoding='utf-8'))
            if data.get('version') != 1 or data.get('strategies') != list(STRATEGIES):
                raise ValueError('Modèle incompatible')
            self.q = {tuple(map(int, k.split(','))): v for k, v in data['q'].items()}
            if any(len(v) != len(STRATEGIES) for v in self.q.values()):
                raise ValueError('Dimensions de Q invalides')
            self.episodes = data['episodes']
        self.reset()

    def reset(self):
        self.threat_until = -1
        self.turn_sign = self.rng.choice((-1, 1))

    def on_episode_end(self, result):
        pass

    def values(self, state):
        # Prior de survie pour les observations jamais rencontrées.
        _, low, medium, thirsty, hungry, threat, resource, contact, damage = state
        return self.q.get(state, [0., 2. if resource else -1.,
                                 4. if threat or damage else -2.,
                                 3. if low else 1. if medium else -1., -0.5])

    def choose(self, obs, epsilon=0):
        state = state_representation(obs)
        if self.rng.random() < epsilon:
            return self.rng.randrange(len(STRATEGIES))
        values = self.values(state)
        return max(range(len(values)), key=lambda i: values[i])

    def act(self, observation):
        return self.action_for(observation, self.choose(observation))

    def action_for(self, obs, strategy):
        s = obs.self_state
        for item in s.inventory:
            if ((item.kind is ObjectKind.DRINK and s.hydration < .65) or
                (item.kind is ObjectKind.FOOD and s.satiety < .65) or
                (item.kind is ObjectKind.MEDICINE and s.health < .8)):
                return Action(interaction=Ingest(item.slot))
        enemies = [d for d in obs.vision if d.kind in {ObjectKind.AGENT, ObjectKind.PROJECTILE}]
        if enemies or any(e.kind is EventKind.DAMAGE_RECEIVED for e in obs.events):
            self.threat_until = obs.tick + 8
        threatened = obs.tick < self.threat_until
        # Ne jamais immobiliser volontairement l'agent à portée d'un adversaire.
        if enemies and min(d.distance for d in enemies) < 3:
            strategy = 2
        close = [d for d in obs.vision if d.kind in BLOCKERS and d.distance < 1.7 and abs(d.bearing) < 1.0]
        contact = [d for d in obs.touch if d.kind in BLOCKERS or d.kind is ObjectKind.DOOR]
        doors = [d for d in (*obs.vision, *obs.touch) if d.kind is ObjectKind.DOOR]
        for d in doors:
            if (getattr(d, 'distance', 0) < 1.1 and not d.properties.get('open', False)):
                return Action(interaction=Manipulate(d.ref))
        if close or contact:
            d = min(close, key=lambda x: x.distance) if close else contact[0]
            turn = -1 if d.bearing >= 0 else 1
            return Action(Movement(-.3, turn, 1.3), Rotation(turn * 2.4))
        useful = [d for d in obs.vision if d.kind in RESOURCES]
        if len(s.inventory) < 6 and not threatened:
            nearby = [d for d in useful if d.distance < .9]
            if nearby:
                return Action(interaction=Pickup(min(nearby, key=lambda d:d.distance).ref))
        if strategy == 3 and not threatened:
            return Action(rest=True)
        if s.energy < .08 and not threatened:
            return Action(rest=True)
        if strategy == 2:
            if enemies:
                d = min(enemies, key=lambda d:d.distance)
                return Action(Movement(-cos(d.bearing), -sin(d.bearing), 2.7))
            sounds = [x for x in obs.hearing if x.kind in {SoundKind.GUNSHOT, SoundKind.AGENT}]
            if sounds:
                sound = max(sounds, key=lambda x:x.intensity)
                return Action(Movement(-cos(sound.bearing), -sin(sound.bearing), 2))
        if strategy == 1 and useful:
            def priority(d):
                need = {ObjectKind.DRINK: 1-s.hydration, ObjectKind.FOOD: 1-s.satiety,
                        ObjectKind.MEDICINE: 1-s.health}[d.kind]
                return d.distance / (.2 + need)
            d = min(useful, key=priority)
            # Rotation précède déplacement dans le moteur : avancer après orientation.
            speed = min(1.6, max(.25, (d.distance - .65) * 2))
            return Action(Movement(1, 0, speed), Rotation(max(-pi,min(pi,d.bearing * 2))))
        if strategy == 4:
            return Action(rotation=Rotation(self.turn_sign * 1.5))
        if obs.tick % 25 == 0:
            self.turn_sign *= -1
        return Action(Movement(1, 0, 1.4), Rotation(self.turn_sign * .35))

    def learn(self, state, action, reward, next_state, terminal, alpha=.12, gamma=.98):
        values = self.q.setdefault(state, list(self.values(state)))
        target = reward if terminal else reward + gamma * max(self.values(next_state))
        values[action] += alpha * (target - values[action])

    def save(self, path=MODEL_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {'version': 1, 'strategies': list(STRATEGIES), 'episodes': self.episodes,
                'q': {','.join(map(str,k)):v for k,v in self.q.items()}}
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
        temporary.replace(path)
