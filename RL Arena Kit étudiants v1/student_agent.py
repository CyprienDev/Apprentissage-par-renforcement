import json
import random
from math import cos, sin, pi, floor
from pathlib import Path

from rl_arena.arena_api import (
    Action, Movement, Rotation, Pickup, Drop, Equip, Use, Ingest,
    Manipulate, Shoot, ObjectKind, EpisodeEndReason, EventKind,
)

MODEL_PATH = Path(__file__).with_name('artifacts') / 'policy.json'
MODEL_VERSION = 'q_learning_v1'
ACTION_TYPES = ('idle', 'rest', 'move', 'rotate', 'pickup', 'manipulate',
                'shoot', 'drop', 'equip', 'use', 'ingest')


def level(value, thresholds):
    # Compte les seuils dépassés : une mesure continue devient une catégorie.
    return sum(value >= threshold for threshold in thresholds)


def item_name(item):
    # Distingue les deux armes et les soins sans utiliser le numéro du slot.
    subtype = item.properties.get('weapon_type') or item.properties.get('medical_type') or ''
    return f'{item.kind.name}:{subtype}'


def state_representation(observation):
    # Un état discret représente les besoins et la menace visible, pas une carte.
    state = observation.self_state
    enemies = [d.distance for d in observation.vision if d.kind is ObjectKind.AGENT]
    # 0 signifie aucun adversaire visible ; les autres nombres indiquent sa proximité.
    threat = 0 if not enemies else 1 + level(min(enemies), (1.5, 5.))
    inventory = {item.kind for item in state.inventory}
    # Chaque bit indique la présence d'une catégorie utile dans l'inventaire.
    kinds = (ObjectKind.FOOD, ObjectKind.DRINK, ObjectKind.MEDICINE,
             ObjectKind.WEAPON, ObjectKind.AMMUNITION)
    inventory_mask = sum(1 << index for index, kind in enumerate(kinds) if kind in inventory)
    equipped = next((item for item in state.inventory if item.slot == state.equipped_slot), None)
    weapon = item_name(equipped) if equipped is not None else 'none'
    collision = any(event.kind is EventKind.COLLISION for event in observation.events)
    # Des catégories grossières permettent de réutiliser un état dans plusieurs arènes.
    values = (level(state.health, (.5,)), level(state.energy, (.25, .6)),
              level(state.hydration, (.35,)), level(state.satiety, (.35,)),
              threat, inventory_mask, weapon, int(collision))
    return '|'.join(map(str, values))


class MyAgent:
    def __init__(self, model_path=MODEL_PATH, seed=0, load=True, adaptive=False,
                 epsilon=None, temperature=None):
        self.rng = random.Random(seed)  # Rend les tirages reproductibles.
        self.adaptive = bool(adaptive)  # False par défaut : pas d'apprentissage en tournoi.
        self.epsilon = (.2 if adaptive else 0.) if epsilon is None else float(epsilon)
        self.temperature = 0. if temperature is None else float(temperature)
        self.alpha = .15  # Part de la nouvelle estimation dans une mise à jour.
        self.gamma = .98  # Importance des récompenses futures.
        self.q_table = {}  # Structure : état -> action -> valeur Q apprise.
        self.episodes = 0
        self.updates = 0
        if load and Path(model_path).is_file():
            self._load(model_path)  # Charge une table Q ; les anciens réseaux sont incompatibles.
        if temperature is not None:
            self.temperature = float(temperature)
        if not 0 <= self.temperature < float('inf'):
            raise ValueError('La température doit être finie et positive ou nulle')
        self.reset()

    def reset(self):
        # Efface la mémoire d'une partie, mais conserve la table Q et le mode tournoi.
        self.previous = None
        self.last_tick = -1
        self.last_action = Action()
        self.last_reward = 0.
        self.online_transitions = 0
        self._ended = False
        self.position = [0., 0.]
        self.visits = {(0, 0): 1}
        self.exploration_reward = 0.

    def set_training_enabled(self, enabled, epsilon=.2):
        self.adaptive = bool(enabled)
        self.epsilon = float(epsilon) if enabled else 0.
        self.previous = None  # Évite de réutiliser une transition de la phase précédente.

    def candidates(self, observation):
        # Construire les possibilités n'impose aucun choix : choose consulte la table Q.
        actions = {'idle': Action(), 'rest': Action(rest=True)}
        for direction in range(8):
            angle = direction * pi / 4
            for speed in (1., 3.2):
                move = Movement(cos(angle), sin(angle), speed)
                actions[f'move:direction:{direction}:{speed}'] = Action(movement=move)
        for sign in (-1, 1):
            actions[f'rotate:{sign}'] = Action(rotation=Rotation(sign * pi / 2))
        for direction in range(8):
            actions[f'shoot:direction:{direction}'] = Action(interaction=Shoot(direction * pi / 4))

        # Une cible par catégorie : la plus proche. Le traitement est identique pour toutes.
        nearest = {}
        for detection in observation.vision:
            if detection.kind not in nearest or detection.distance < nearest[detection.kind].distance:
                nearest[detection.kind] = detection
        for kind, target in nearest.items():
            # Le numéro de référence reste dans l'Action, jamais dans la clé apprise.
            distance = level(target.distance, (.45, 1.5, 5.))
            label = f'{kind.name}:{distance}'
            for offset in (0, 1, 2, 3):
                angle = target.bearing + offset * pi / 2
                for speed in (1., 3.2):
                    move = Movement(cos(angle), sin(angle), speed)
                    actions[f'move:target:{label}:{offset}:{speed}'] = Action(movement=move)
            actions[f'pickup:{label}'] = Action(interaction=Pickup(target.ref))
            actions[f'manipulate:{label}'] = Action(interaction=Manipulate(target.ref))
            actions[f'shoot:target:{label}'] = Action(interaction=Shoot(target.bearing))

        # L'audition fournit aussi des directions possibles, même hors du champ de vision.
        if observation.hearing:
            sound = max(observation.hearing, key=lambda sound: sound.intensity)
            intensity = level(sound.intensity, (.2, .6))
            for offset in (0, 1, 2, 3):
                angle = sound.bearing + offset * pi / 2
                move = Movement(cos(angle), sin(angle), 3.2)
                actions[f'move:sound:{sound.kind.name}:{intensity}:{offset}'] = Action(movement=move)

        for item in observation.self_state.inventory:
            name = item_name(item)
            # setdefault garde un exemplaire si plusieurs objets ont le même rôle.
            for verb, interaction in (('drop', Drop(item.slot)), ('equip', Equip(item.slot)),
                                      ('use', Use(item.slot)), ('ingest', Ingest(item.slot))):
                actions.setdefault(f'{verb}:item:{name}', Action(interaction=interaction))
            for kind, target in nearest.items():
                distance = level(target.distance, (.45, 1.5, 5.))
                key = f'use:target:{name}:{kind.name}:{distance}'
                interaction = Use(item.slot, target.ref)
                actions.setdefault(key, Action(interaction=interaction))
                move = Movement(cos(target.bearing), sin(target.bearing), 3.2)
                actions.setdefault(key + ':move', Action(movement=move, interaction=interaction))
        return actions

    def choose(self, state, actions):
        keys = list(actions)
        if self.adaptive and self.rng.random() < self.epsilon:
            # Exploration équilibrée : tirer d'abord un type, puis une action de ce type.
            types = sorted({key.split(':')[0] for key in keys})
            selected = self.rng.choice(types)
            return self.rng.choice([key for key in keys if key.split(':')[0] == selected])
        values = self.q_table.get(state, {})  # La lecture n'ajoute rien en tournoi.
        best = max(values.get(key, 0.) for key in keys)
        if self.temperature > 0:
            from math import exp
            weights = [exp((values.get(key, 0.) - best) / self.temperature) for key in keys]
            return self.rng.choices(keys, weights=weights, k=1)[0]
        # Les actions inconnues valent zéro ; les égalités sont départagées au hasard.
        return self.rng.choice([key for key in keys if abs(values.get(key, 0.) - best) <= 1e-9])

    def learn(self, state, action, reward, next_state=None, next_actions=(), terminal=False):
        if not self.adaptive:
            return  # Protection appliquée même si un script appelle learn pendant le tournoi.
        row = self.q_table.setdefault(state, {})
        old_value = row.get(action, 0.)
        future = self.q_table.get(next_state, {})
        best_next = 0. if terminal else max((future.get(key, 0.) for key in next_actions), default=0.)
        target = reward + self.gamma * best_next
        # Formule du Q-learning : Q(s,a) += alpha * [r + gamma*max Q(s',a') - Q(s,a)].
        row[action] = old_value + self.alpha * (target - old_value)
        self.updates += 1
        self.online_transitions += 1

    def reward(self, previous, current):
        before, after = previous.self_state, current.self_state
        ticks = max(1, current.tick - previous.tick)
        reward = .01 * ticks  # La survie compte, sans écraser les gains de ressources.
        reward += 10 * (after.health - before.health)
        reward += 4 * (after.hydration - before.hydration)
        reward += 3 * (after.satiety - before.satiety)
        reward += .2 * (after.energy - before.energy)
        reward += .2 * (self.gamma ** ticks * len(after.inventory) - len(before.inventory))
        reward += self.exploration_reward  # Encourage à découvrir de nouvelles cellules estimées.
        reward -= .15 * sum(event.kind is EventKind.COLLISION for event in current.events)
        if self.last_action.interaction is not None:
            successes = {EventKind.ITEM_PICKED_UP, EventKind.ITEM_DROPPED, EventKind.ITEM_EQUIPPED,
                         EventKind.ITEM_USED, EventKind.SHOT_FIRED, EventKind.INTERACTION}
            if not any(event.kind in successes for event in current.events):
                reward -= .1  # Une tentative sans effet apporte une pénalité.
        return reward

    def _update_position(self, observation):
        self.exploration_reward = 0.
        if self.previous is None or self.last_action.movement is None:
            return
        if any(event.kind is EventKind.COLLISION for event in observation.events):
            return  # Une collision rend l'estimation du déplacement peu fiable.
        move = self.last_action.movement
        angle = observation.self_state.orientation
        elapsed = max(1, observation.tick - self.last_tick) * .5
        distance = elapsed * observation.self_state.speed
        self.position[0] += distance * (cos(angle) * move.forward - sin(angle) * move.right)
        self.position[1] += distance * (sin(angle) * move.forward + cos(angle) * move.right)
        cell = tuple(floor(value) for value in self.position)
        if cell not in self.visits:
            self.exploration_reward = .02
        if len(self.visits) >= 2048 and cell not in self.visits:
            self.visits.pop(next(iter(self.visits)))
        self.visits[cell] = self.visits.get(cell, 0) + 1

    def act(self, observation):
        if observation.tick == self.last_tick:
            return self.last_action  # Deux appels au même tour ne créent pas deux transitions.
        if observation.tick < self.last_tick:
            self.reset()
        self._update_position(observation)
        state = state_representation(observation)
        actions = self.candidates(observation)
        if self.previous is not None:
            old_observation, old_state, old_action = self.previous
            self.last_reward = self.reward(old_observation, observation)
            self.learn(old_state, old_action, self.last_reward, state, actions)
        key = self.choose(state, actions)
        self.last_action = actions[key]
        self.previous = (observation, state, key)
        self.last_tick = observation.tick
        return self.last_action

    def on_episode_end(self, result):
        if self._ended:
            return
        self._ended = True
        if self.previous is not None:
            observation, state, action = self.previous
            reward = -10. if result.reason is EpisodeEndReason.ELIMINATED else 0.
            reward -= .2 * len(observation.self_state.inventory)
            self.learn(state, action, reward, terminal=True)  # Pas de valeur future après la fin.
        self.episodes += 1

    def _load(self, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version') != MODEL_VERSION:
            raise ValueError('Modèle incompatible : une table Q-learning est nécessaire')
        table = data.get('q_table')
        if not isinstance(table, dict):
            raise ValueError('Table Q invalide')
        for state, row in table.items():
            if not isinstance(state, str) or not isinstance(row, dict):
                raise ValueError('État Q invalide')
            for action, value in row.items():
                if not isinstance(action, str) or not isinstance(value, (int, float)):
                    raise ValueError('Valeur Q invalide')
                if not -float('inf') < value < float('inf'):
                    raise ValueError('Valeur Q non finie')
        self.q_table = table
        self.episodes = int(data.get('episodes', 0))
        self.temperature = float(data.get('temperature', 0.))

    def save(self, path=MODEL_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {'version': MODEL_VERSION, 'algorithm': 'tabular_q_learning',
                'episodes': self.episodes, 'temperature': self.temperature, 'q_table': self.q_table}
        content = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        if len(content.encode('utf-8')) > 10_000_000:
            raise ValueError('La table dépasse la limite de données de 10 Mo')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(path)  # Sauvegarde explicite ; aucun fichier écrit automatiquement en tournoi.
