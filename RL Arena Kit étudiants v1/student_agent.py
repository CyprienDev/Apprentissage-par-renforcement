from __future__ import annotations
from collections import deque
from math import atan2, cos, sin, pi, hypot, sqrt, tanh
import json
from pathlib import Path
from random import Random
from rl_arena.arena_api import (Action, Movement, Rotation, ObjectKind, EventKind,
    SoundKind, Ingest, Pickup, Manipulate, Equip, Use, Shoot, Drop, EpisodeEndReason)

MODEL_PATH = Path(__file__).with_name('artifacts') / 'policy.json'
STRATEGIES = ('explore', 'forage', 'flee', 'rest', 'scan')
BLOCKERS = {ObjectKind.WALL, ObjectKind.OBSTACLE, ObjectKind.MOVING_OBSTACLE,
            ObjectKind.HAZARD}
RESOURCES = {ObjectKind.FOOD, ObjectKind.DRINK, ObjectKind.MEDICINE}


def wrap(angle):
    return (angle + pi) % (2*pi) - pi


def dangerous_damage(obs):
    depleted = min(obs.self_state.hydration,obs.self_state.satiety) < .2
    bleeding = any(i.bleeding > 0 for i in obs.self_state.injuries)
    return any(e.kind is EventKind.DAMAGE_RECEIVED and
               (e.value is None or e.value > .01 or not (depleted or bleeding))
               for e in obs.events)


def visible_threats(obs):
    return [d for d in obs.vision if
            (d.kind is ObjectKind.AGENT and d.distance < 6) or
            (d.kind is ObjectKind.PROJECTILE and d.distance < 10)]


def state_representation(obs):
    s = obs.self_state
    enemy = min((d.distance for d in obs.vision if d.kind in
                 {ObjectKind.AGENT, ObjectKind.PROJECTILE}), default=99)
    return (int(s.health < .5), int(s.energy < .18), int(s.energy < .4),
            int(s.hydration < .65), int(s.satiety < .65),
            0 if enemy > 6 else 1 if enemy > 3 else 2,
            int(any(d.kind in RESOURCES for d in obs.vision)),
            int(bool(obs.touch)), int(dangerous_damage(obs)))


class MyAgent:
    def __init__(self, model_path=MODEL_PATH, seed=0, load=True, adaptive=True,
                 replay_steps=4, epsilon=.025):
        self.rng = Random(seed)
        self.learning_rng = Random(seed+991)
        self.adaptive = adaptive
        self.replay_steps = replay_steps
        self.epsilon = epsilon
        self.q, self.episodes, self.updates = {}, 0, 0
        if load and Path(model_path).exists():
            data = json.loads(Path(model_path).read_text(encoding='utf-8'))
            if data.get('version') in {1,2}:

                data = {'version': 3, 'strategies': list(STRATEGIES), 'q': {}, 'episodes': 0}
            if data.get('version') != 3 or data.get('strategies') != list(STRATEGIES):
                raise ValueError('Modèle incompatible')
            self.q = {tuple(map(int, k.split(','))): list(v) for k,v in data['q'].items()}
            if any(len(k) != 9 or len(v) != 5 or any(not isinstance(x, (float,int))
                   or not (-1e9 < x < 1e9) for x in v) for k,v in self.q.items()):
                raise ValueError('Dimensions ou valeurs Q invalides')
            self.episodes = data.get('episodes', 0)
        self.reset()

    def reset(self):

        self.position = [0., 0.]
        self.visits = {}
        self.resources = []
        self.replay = deque(maxlen=256)
        self.previous = None
        self.last_action = None
        self.last_tick = None
        self.last_strategy = None
        self.threat_until = -1
        self.threat_heading = None
        self.turn_sign = self.rng.choice((-1, 1))
        self.heading = self.rng.uniform(-pi, pi)
        self.stuck_ticks = 0
        self.detour_until = -1
        self.door_cooldowns = {}
        self.visited_doors = []
        self.enter_until = -1
        self.entry_heading = None
        self.online_transitions = 0
        self.last_reward = 0.
        self.energy_cost = .003
        self.rest_gain = .025
        self.dynamics_samples = 0
        self.ignored_loot_until = {}
        self.recovering = False
        self._ended = False

    def prior(self, state):
        _, low, medium, thirsty, hungry, threat, resource, contact, damage = state
        return [1., 2.5 if resource else .8,
            4. if threat or damage else -3., 3. if low else 1.5 if medium else -4., .7]

    def values(self, state):
        return self.q.get(state,self.prior(state))

    def _state(self, obs):
        state = list(state_representation(obs))
        state[6] = int(any(self._want_resource(r[0],obs.self_state) for r in self.resources)
                       or any(self._wanted_detection(d,obs.self_state) for d in obs.vision))
        if obs.tick < self.threat_until:
            state[5] = max(1,state[5])
        return tuple(state)

    def _want_resource(self, kind, s):
        if kind not in RESOURCES:
            return False
        count = sum(i.kind is kind for i in s.inventory)
        limit = 3 if kind is ObjectKind.DRINK else 2 if kind is ObjectKind.FOOD else 1
        if count >= limit or len(s.inventory) >= 7:
            return False


        return kind is not ObjectKind.MEDICINE or s.health < .85 or bool(s.injuries)

    def _wanted_detection(self, d, s):
        if not self._want_resource(d.kind,s):
            return False
        bandage = d.properties.get('medical_type',d.properties.get('label')) == 'bandage'
        return not bandage or any(i.bleeding > .05 for i in s.injuries)

    def choose(self, obs, epsilon=None):
        state = self._state(obs)
        s = obs.self_state
        available = [0, 4]
        wanted_memory = any(self._want_resource(r[0],s) for r in self.resources)
        if state[6]:
            available.append(1)
        if obs.tick < self.threat_until or state[5] or state[8]:
            available.append(2)
        if s.energy < .4 and obs.tick >= self.threat_until:
            available.append(3)

        if (s.hydration < .6 or s.satiety < .6) and wanted_memory and not state[5]:
            return 1
        epsilon = self.epsilon if epsilon is None else epsilon
        if self.rng.random() < epsilon and not state[5]:
            return self.rng.choice(available)
        values = self.values(state)
        prior = self.prior(state)
        best = max(values[i] for i in available)


        return max(available,key=lambda i:prior[i]+.5*tanh((values[i]-best)/2))

    def learn(self, state, action, reward, next_state, terminal, alpha=.2, gamma=.97):
        values = self.q.setdefault(state, list(self.values(state)))
        if terminal:
            target = reward
        else:
            eligible = [0,4]
            if next_state[6]:
                eligible.append(1)
            if next_state[5] or next_state[8]:
                eligible.append(2)
            if next_state[1] or next_state[2]:
                eligible.append(3)
            future = self.values(next_state)
            target = reward + gamma * max(future[i] for i in eligible)
        values[action] += alpha * (target - values[action])
        self.updates += 1

    def _feedback(self, obs, novelty):
        if self.previous is None:
            return
        old, state, strategy = self.previous
        s, before = obs.self_state, old.self_state
        collisions = sum(e.kind is EventKind.COLLISION for e in obs.events)
        if self.adaptive and self.last_action is not None:
            movement = self.last_action.movement
            if movement and s.speed > .5 and not collisions and before.energy > .2:
                cost = max(0.,before.energy-s.energy)/s.speed**2
                if 0 < cost < .1:
                    self.energy_cost = .7*self.energy_cost + .3*cost
                    self.dynamics_samples += 1
            if self.last_action.rest and s.energy < .99:
                gain = s.energy-before.energy
                if gain > 0:
                    self.rest_gain = .7*self.rest_gain + .3*gain
        reward = (.025 + 20 * max(0, s.hydration-before.hydration)
                  + 12 * max(0, s.satiety-before.satiety)
                  + 8 * (s.health-before.health)
                  + .25 * (s.energy-before.energy)
                  + .08 * novelty - .3 * min(2, collisions))
        reward += .3 * sum(e.kind is EventKind.ITEM_PICKED_UP for e in obs.events)
        if self.stuck_ticks > 3:
            reward -= .15
        self.last_reward = reward
        if self.adaptive:
            transition = (state, strategy, reward, self._state(obs), False)
            self.learn(*transition)
            self.replay.append(transition)
            for _ in range(self.replay_steps):
                self.learn(*self.learning_rng.choice(self.replay), alpha=.08)
            self.online_transitions += 1

    def _travel_speed(self, obs):
        if not self.adaptive or self.dynamics_samples < 8:
            return 1.7


        speed = max(1.2,min(2.7,sqrt(self.rest_gain/max(.0001,self.energy_cost))))
        if any(d.kind in BLOCKERS and d.distance < 2 for d in obs.vision):
            speed = min(speed,1.5)
        return speed

    def _observe(self, obs):
        if self.last_tick is not None and obs.tick < self.last_tick:
            self.reset()
        if self.previous is not None and self.last_action and self.last_action.movement:
            movement = self.last_action.movement
            collision = any(e.kind is EventKind.COLLISION for e in obs.events)
            travelled = obs.self_state.speed * .5 * (0.15 if collision else 1.)
            angle = obs.self_state.orientation + atan2(movement.right, movement.forward)
            self.position[0] += cos(angle) * travelled
            self.position[1] += sin(angle) * travelled
            self.stuck_ticks = self.stuck_ticks+1 if collision or travelled < .08 else 0
        cell = (round(self.position[0]/2), round(self.position[1]/2))
        novelty = cell not in self.visits
        self.visits[cell] = self.visits.get(cell, 0) + 1
        if len(self.visits) > 2048:
            self.visits.pop(next(iter(self.visits)))
        s = obs.self_state
        self.resources = [r for r in self.resources if obs.tick-r[3] < 120 and
                          hypot(r[1]-self.position[0], r[2]-self.position[1]) > .65]
        for d in obs.vision:
            if d.kind not in RESOURCES:
                continue
            angle = s.orientation + d.bearing
            x = self.position[0] + (d.distance+.2)*cos(angle)
            y = self.position[1] + (d.distance+.2)*sin(angle)
            self.resources = [r for r in self.resources if r[0] != d.kind or hypot(r[1]-x,r[2]-y) > 1.5]
            self.resources.append((d.kind,x,y,obs.tick))
        self.resources = self.resources[-32:]
        self._feedback(obs, novelty)
        enemies = visible_threats(obs)
        if enemies:
            d = min(enemies, key=lambda d:d.distance)
            self.threat_until = obs.tick+10
            self.threat_heading = wrap(s.orientation + d.bearing + pi)
        elif dangerous_damage(obs):
            self.threat_until = obs.tick+8
            sounds = sorted(obs.hearing, key=lambda x:x.intensity, reverse=True)
            if sounds:
                self.threat_heading = wrap(s.orientation+sounds[0].bearing+pi)
        if self.stuck_ticks >= 3:
            if obs.tick >= self.detour_until:
                self.turn_sign *= -1
            self.detour_until = obs.tick+12
        self.last_tick = obs.tick

    def act(self, observation):

        if self.last_tick == observation.tick and self.last_action is not None:
            return self.last_action
        self._observe(observation)
        strategy = self.choose(observation)
        action = self.action_for(observation, strategy)
        self.last_action = action
        self.previous = (observation, self._state(observation), self.last_strategy)
        return action

    def _resource_score(self, kind, s):
        if kind is ObjectKind.DRINK:
            return .35 + 2*(1-s.hydration)
        if kind is ObjectKind.FOOD:
            return .2 + 1.5*(1-s.satiety)
        return .1 + 1.5*(1-s.health) + sum(i.bleeding for i in s.injuries)

    def _navigate(self, obs, desired, speed, fleeing=False, entering=False):


        blockers = [d for d in obs.vision if d.kind in BLOCKERS or
                    (d.kind is ObjectKind.DOOR and not entering and not d.properties.get('open', False))]
        blockers.extend(d for d in obs.touch if d.kind in BLOCKERS)
        candidates = [wrap(desired + offset) for offset in (0,.35,-.35,.7,-.7,1.1,-1.1,1.6,-1.6,2.2,-2.2,pi)]
        def score(angle):
            value = 2.5*cos(angle-desired)
            for d in blockers:
                distance = getattr(d,'distance',0.)
                if distance > 3:
                    continue
                separation = abs(wrap(angle-d.bearing))


                angular = 0 if d.kind is ObjectKind.WALL else min(.3,getattr(d,'angular_size',.6)/2)
                margin = angular + atan2(.6,max(.2,distance))
                if separation < margin:
                    value -= (8 if distance < 1.2 else 3) * (1-separation/max(.01,margin))
            direction = obs.self_state.orientation+angle
            cell = (round((self.position[0]+2*cos(direction))/2),
                    round((self.position[1]+2*sin(direction))/2))
            value -= min(2., .02*self.visits.get(cell,0))
            if obs.tick < self.detour_until:
                value += .8*self.turn_sign*sin(angle)
            return value
        angle = max(candidates, key=score)
        if fleeing:
            return Action(movement=Movement(cos(angle),sin(angle),speed))
        rotation = max(-pi,min(pi,2*angle))
        relative = angle-rotation*.5
        return Action(Movement(cos(relative),sin(relative),speed),Rotation(rotation))

    def action_for(self, obs, strategy):
        s = obs.self_state
        self.last_strategy = strategy
        for item in s.inventory:
            bleeding = any(i.bleeding > .05 for i in s.injuries)
            bandage = item.properties.get('medical_type',item.properties.get('label')) == 'bandage'
            if ((item.kind is ObjectKind.DRINK and s.hydration < .6) or
                (item.kind is ObjectKind.FOOD and s.satiety < .65) or
                (item.kind is ObjectKind.MEDICINE and (bleeding or s.health < .85 and not bandage))):
                return Action(interaction=Ingest(item.slot))
        enemies = visible_threats(obs)
        threatened = bool(enemies) or obs.tick < self.threat_until
        if threatened:
            self.last_strategy = strategy = 2
        else:
            if s.energy < .2:
                self.recovering = True
            elif s.energy >= .75:
                self.recovering = False
        if self.recovering and not threatened:
            self.last_strategy = strategy = 3
        urgent = {ObjectKind.DRINK:s.hydration < .6,ObjectKind.FOOD:s.satiety < .55,
                  ObjectKind.MEDICINE:s.health < .5}
        if len(s.inventory) >= 7 and not threatened and any(
                urgent.get(d.kind,False) and d.distance < 1.5 for d in obs.vision):
            dispensable = [i for i in s.inventory if i.kind in {ObjectKind.AMMUNITION,
                ObjectKind.RAW_MATERIAL,ObjectKind.WEAPON_MODIFIER} or
                (i.kind is ObjectKind.MEDICINE and s.health > .85 and not s.injuries)]
            if dispensable:
                item = dispensable[0]
                self.ignored_loot_until[item.kind] = obs.tick+100
                return Action(interaction=Drop(item.slot))

        for d in sorted(obs.vision, key=lambda d:d.distance):
            if d.distance > .9 or len(s.inventory) >= 7:
                continue
            if obs.tick < self.ignored_loot_until.get(d.kind,-1):
                continue
            counts = sum(i.kind is d.kind for i in s.inventory)
            needed = self._want_resource(d.kind,s)
            needed |= d.kind is ObjectKind.MEDICINE and counts == 0
            needed |= d.kind is ObjectKind.WEAPON and counts == 0
            needed |= d.kind is ObjectKind.AMMUNITION and counts < 2 and any(i.kind is ObjectKind.WEAPON for i in s.inventory)
            if needed and (not threatened or d.kind is ObjectKind.MEDICINE and s.health < .4):
                return Action(interaction=Pickup(d.ref))
        doors = [d for d in (*obs.vision,*obs.touch) if d.kind is ObjectKind.DOOR and
                 getattr(d,'distance',0) < 1.2 and not d.properties.get('open',False)]
        if doors and not threatened:
            d = doors[0]
            key = round(wrap(s.orientation+d.bearing),1)
            if obs.tick >= self.door_cooldowns.get(key,-1):
                self.door_cooldowns[key] = obs.tick+6
                self.entry_heading = wrap(s.orientation+d.bearing)
                self.enter_until = obs.tick+6
                self.visited_doors.append((self.position[0]+getattr(d,'distance',0)*cos(self.entry_heading),
                                           self.position[1]+getattr(d,'distance',0)*sin(self.entry_heading),obs.tick))
                return Action(interaction=Manipulate(d.ref))
        if strategy == 3 and not threatened and (s.energy < .4 or self.recovering):
            return Action(rest=True)
        if strategy == 2 and threatened:
            if enemies:
                d = min(enemies,key=lambda d:d.distance)
                desired = wrap(d.bearing+pi)
            else:
                desired = wrap((self.threat_heading or self.heading)-s.orientation)
            action = self._navigate(obs,desired,2.7 if s.energy > .2 else 1.7,True)

            equipped = next((i for i in s.inventory if i.slot == s.equipped_slot),None)
            agents = [d for d in enemies if d.kind is ObjectKind.AGENT]
            if equipped and agents:
                target = min(agents,key=lambda d:d.distance)
                label = equipped.properties.get('weapon_type',equipped.properties.get('label'))
                interaction = None
                if label == 'rolling_pin' and target.distance < 1.05:
                    interaction = Use(equipped.slot,target.ref)
                elif label == 'potato_launcher' and target.distance < 7 and any(i.kind is ObjectKind.AMMUNITION for i in s.inventory):
                    interaction = Shoot(target.bearing)
                if interaction:
                    return Action(action.movement,interaction=interaction)
            return action
        weapon = next((i for i in s.inventory if i.kind is ObjectKind.WEAPON),None)
        if weapon and s.equipped_slot != weapon.slot:
            return Action(interaction=Equip(weapon.slot))
        if obs.tick < self.enter_until and self.entry_heading is not None:
            return self._navigate(obs,wrap(self.entry_heading-s.orientation),.85,entering=True)
        if strategy == 1:
            targets = [(d.kind,d.distance,d.bearing) for d in obs.vision if self._wanted_detection(d,s)]
            if not targets:
                targets = [(kind,hypot(x-self.position[0],y-self.position[1]),
                            wrap(atan2(y-self.position[1],x-self.position[0])-s.orientation))
                           for kind,x,y,_ in self.resources if self._want_resource(kind,s)]
            if targets:
                target = min(targets,key=lambda t:t[1]/self._resource_score(t[0],s))
                return self._navigate(obs,target[2],min(self._travel_speed(obs),max(.35,(target[1]-.65)*2)))
        self.visited_doors = [p for p in self.visited_doors if obs.tick-p[2] < 400][-32:]


        passages = []
        for d in obs.vision:
            if d.kind is not ObjectKind.DOOR or d.distance > 7:
                continue
            angle = s.orientation+d.bearing
            x,y = self.position[0]+d.distance*cos(angle),self.position[1]+d.distance*sin(angle)
            if not any(hypot(x-p[0],y-p[1]) < 2 for p in self.visited_doors):
                passages.append(d)
        if passages:
            d = min(passages,key=lambda d:d.distance)
            if d.properties.get('open',False) and d.distance < 1.2:
                self.entry_heading = wrap(s.orientation+d.bearing)
                self.enter_until = obs.tick+6
                self.visited_doors.append((self.position[0]+d.distance*cos(self.entry_heading),
                                           self.position[1]+d.distance*sin(self.entry_heading),obs.tick))
            return self._navigate(obs,d.bearing,min(1.4,max(.5,d.distance)),entering=True)

        if obs.tick % 35 == 0 or self.stuck_ticks >= 3:
            self.heading = wrap(s.orientation+self.turn_sign*self.rng.uniform(.5,1.7))
        desired = wrap(self.heading-s.orientation)
        if strategy == 4:
            desired = wrap(desired+self.turn_sign*.65)
        return self._navigate(obs,desired,self._travel_speed(obs) if strategy != 4 else 1.2)

    def on_episode_end(self, result):
        if self._ended:
            return
        self._ended = True
        if self.adaptive and self.previous is not None:
            _,state,strategy = self.previous
            terminal = result.reason is EpisodeEndReason.ELIMINATED
            self.learn(state,strategy,-12. if terminal else .025,None,True)
        self.episodes += 1

    def save(self, path=MODEL_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        data = {'version':3,'strategies':list(STRATEGIES),'episodes':self.episodes,
                'q':{','.join(map(str,k)):v for k,v in self.q.items()}}
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
        temporary.replace(path)
