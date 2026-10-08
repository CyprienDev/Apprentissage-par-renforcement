import json
from collections import deque
from math import cos, sin, pi
from pathlib import Path

import numpy as np
from rl_arena.arena_api import (
    Action, Movement, Rotation, Pickup, Drop, Equip, Use, Ingest,
    Manipulate, Shoot, ObjectKind, EpisodeEndReason, EventKind,
)

MODEL_PATH = Path(__file__).with_name('artifacts') / 'policy.json'
KINDS = tuple(ObjectKind)
ACTION_TYPES = (
    'idle', 'move', 'rotate', 'rest', 'pickup', 'drop',
    'equip', 'use', 'ingest', 'manipulate', 'shoot',
)

STATE_FEATURE_COUNT = 13 + 4 * len(KINDS)
ACTION_FEATURE_COUNT = len(ACTION_TYPES) + 5 + len(KINDS) + 8
FEATURE_COUNT = STATE_FEATURE_COUNT + ACTION_FEATURE_COUNT
HIDDEN_NEURONS = 32
MEMORY_SIZE = 2048
BATCH_SIZE = 16
TRAIN_EVERY = 4
TARGET_SYNC_EVERY = 200
LEARNING_RATE = .0005


def state_representation(observation):
    agent_state = observation.self_state
    bleeding = sum(injury.bleeding for injury in agent_state.injuries)
    collision = any(event.kind is EventKind.COLLISION for event in observation.events)
    damage = 0.
    for event in observation.events:
        if event.kind is EventKind.DAMAGE_RECEIVED:
            damage += event.value or 0.
    sound_intensity = sum(sound.intensity for sound in observation.hearing)

    features = [
        agent_state.health,
        agent_state.energy,
        agent_state.hydration,
        agent_state.satiety,
        min(1., agent_state.carried_mass / 20),
        min(1., abs(agent_state.speed) / 3),
        len(agent_state.inventory) / 8,
        min(1., bleeding),
        float(bool(observation.touch)),
        float(collision),
        min(1., damage),
        min(1., sound_intensity),
        float(agent_state.equipped_slot is not None),
    ]

    for kind in KINDS:
        visible_objects = []
        for detection in observation.vision:
            if detection.kind is kind:
                visible_objects.append(detection)

        if visible_objects:
            nearest = min(visible_objects, key=lambda detection: detection.distance)
            proximity = 1 / (1 + max(0., nearest.distance))
            features.extend([
                proximity,
                proximity * cos(nearest.bearing),
                proximity * sin(nearest.bearing),
            ])
        else:
            features.extend([0., 0., 0.])

    for kind in KINDS:
        quantity = sum(item.kind is kind for item in agent_state.inventory)
        features.append(quantity / 8)

    return np.asarray(features, dtype=np.float32)


class MyAgent:
    def __init__(self, model_path=MODEL_PATH, seed=0, load=True, adaptive=False,
                 replay_steps=1, epsilon=None):
        self.rng = np.random.default_rng(seed)
        self.replay_rng = np.random.default_rng(seed + 991)
        self.adaptive = adaptive
        if epsilon is None:
            self.epsilon = .10 if adaptive else 0.
        else:
            self.epsilon = epsilon
        self.replay_steps = replay_steps
        self.gamma = .99
        self.episodes = 0
        self.updates = 0

        input_weights = self.rng.normal(0, .08, (FEATURE_COUNT, HIDDEN_NEURONS))
        input_bias = np.zeros(HIDDEN_NEURONS, dtype=np.float32)
        output_weights = self.rng.normal(0, .08, (HIDDEN_NEURONS, 1))
        output_bias = np.zeros(1, dtype=np.float32)
        self.weights = [
            input_weights.astype(np.float32), input_bias,
            output_weights.astype(np.float32), output_bias,
        ]

        if load and Path(model_path).exists():
            self._load(model_path)

        self.target = [weights.copy() for weights in self.weights]
        self.moments = [np.zeros_like(weights) for weights in self.weights]
        self.variances = [np.zeros_like(weights) for weights in self.weights]
        self.replay = deque(maxlen=MEMORY_SIZE)
        self.reset()

    def reset(self):
        self.previous = None
        self.last_tick = -1
        self.last_action = Action()
        self.online_transitions = 0
        self.last_reward = 0.
        self._ended = False

    def set_training_enabled(self, enabled, epsilon=.10):
        self.adaptive = bool(enabled)
        self.epsilon = epsilon if enabled else 0.
        self.previous = None
        if not enabled:
            self.replay.clear()

    def _load(self, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version') != 4 or data.get('actions') != list(ACTION_TYPES):
            raise ValueError('Checkpoint incompatible : entraîner le nouvel agent RL version 4')

        loaded_weights = []
        for values in data['weights']:
            loaded_weights.append(np.asarray(values, dtype=np.float32))
        if len(loaded_weights) != len(self.weights):
            raise ValueError('Poids de modèle invalides')
        for loaded, expected in zip(loaded_weights, self.weights):
            if loaded.shape != expected.shape or not np.isfinite(loaded).all():
                raise ValueError('Poids de modèle invalides')

        self.weights = loaded_weights
        self.episodes = int(data.get('episodes', 0))

    def candidates(self, observation):
        actions = [
            (Action(), 'idle', None),
            (Action(rest=True), 'rest', None),
        ]
        directions = np.arange(8) * pi / 4
        for angle in directions:
            for speed in (1., 2.5):
                movement = Movement(cos(angle), sin(angle), speed)
                actions.append((Action(movement=movement), 'move', None))
        for angular_speed in (-pi / 2, pi / 2):
            rotation = Rotation(angular_speed)
            actions.append((Action(rotation=rotation), 'rotate', None))

        for detection in observation.vision:
            pickup = Action(interaction=Pickup(detection.ref))
            manipulate = Action(interaction=Manipulate(detection.ref))
            actions.append((pickup, 'pickup', detection))
            actions.append((manipulate, 'manipulate', detection))

        for item in observation.self_state.inventory:
            interactions = [
                ('drop', Drop(item.slot)),
                ('equip', Equip(item.slot)),
                ('use', Use(item.slot)),
                ('ingest', Ingest(item.slot)),
            ]
            for name, interaction in interactions:
                actions.append((Action(interaction=interaction), name, item))
            for detection in observation.vision:
                interaction = Use(item.slot, detection.ref)
                actions.append((Action(interaction=interaction), 'use', detection))

        for bearing in directions:
            shoot = Shoot(float(bearing))
            actions.append((Action(interaction=shoot), 'shoot', None))
        return actions

    def features(self, observation, candidates):
        state = state_representation(observation)
        rows = []
        for action, name, target in candidates:
            action_type = [0.] * len(ACTION_TYPES)
            action_type[ACTION_TYPES.index(name)] = 1.

            movement_features = [0.] * 5
            if action.movement is not None:
                movement = action.movement
                movement_features[:3] = [
                    movement.forward, movement.right, movement.speed / 3,
                ]
            if action.rotation is not None:
                movement_features[3] = action.rotation.angular_speed / pi
            if isinstance(action.interaction, Shoot):
                bearing = action.interaction.bearing
                movement_features[:2] = [cos(bearing), sin(bearing)]
            movement_features[4] = float(action.rest)

            target_kind = [0.] * len(KINDS)
            target_properties = [0.] * 8
            if target is not None:
                target_kind[KINDS.index(target.kind)] = 1.
                if hasattr(target, 'distance'):
                    target_properties[:4] = [
                        1 / (1 + max(0., target.distance)),
                        cos(target.bearing),
                        sin(target.bearing),
                        min(1., target.angular_size / pi),
                    ]
                else:
                    target_properties[4] = min(1., target.mass / 10)
                target_properties[5] = float(target.properties.get('open', False))
                target_properties[6] = float(target.properties.get('medical_type') == 'bandage')
                target_properties[7] = float(target.properties.get('label') == 'potato_launcher')

            row = list(state) + action_type + movement_features + target_kind + target_properties
            rows.append(row)
        return np.asarray(rows, dtype=np.float32).reshape(-1, FEATURE_COUNT)

    def values(self, features, target=False):
        network = self.target if target else self.weights
        input_weights, input_bias, output_weights, output_bias = network
        hidden_layer = np.tanh(features @ input_weights + input_bias)
        q_values = hidden_layer @ output_weights + output_bias
        return q_values.ravel()

    def choose(self, features, epsilon=None):
        exploration = 0.
        if self.adaptive:
            exploration = self.epsilon if epsilon is None else epsilon
        if self.rng.random() < exploration:
            return int(self.rng.integers(len(features)))

        q_values = self.values(features)
        highest_value = q_values.max()
        best_actions = np.flatnonzero(np.isclose(q_values, highest_value, rtol=0, atol=1e-7))
        return int(self.rng.choice(best_actions))

    def reward(self, previous, current):
        before = previous.self_state
        after = current.self_state
        elapsed_ticks = max(1, current.tick - previous.tick)
        collisions = sum(event.kind is EventKind.COLLISION for event in current.events)

        survival_reward = .025 * elapsed_ticks
        health_reward = 10 * (after.health - before.health)
        hydration_reward = 3 * (after.hydration - before.hydration)
        food_reward = 2 * (after.satiety - before.satiety)
        energy_reward = .1 * (after.energy - before.energy)
        collision_penalty = .05 * collisions

        return (survival_reward + health_reward + hydration_reward
                + food_reward + energy_reward - collision_penalty)

    def remember(self, features, reward, next_features, terminal=False):
        if not self.adaptive:
            return
        experience = (features.copy(), float(reward), next_features.copy(), terminal)
        self.replay.append(experience)
        self.online_transitions += 1

        training_due = self.online_transitions % TRAIN_EVERY == 0
        if terminal or training_due:
            for _ in range(self.replay_steps):
                self.learn()

    def _training_targets(self, batch):
        expected_values = np.asarray([experience[1] for experience in batch], dtype=np.float32)
        next_states = []
        batch_positions = []
        for position, experience in enumerate(batch):
            _, _, next_features, terminal = experience
            if not terminal:
                next_states.append(next_features)
                batch_positions.append(position)

        if not next_states:
            return expected_values

        all_next_features = np.concatenate(next_states)
        online_values = self.values(all_next_features)
        target_values = self.values(all_next_features, target=True)
        offset = 0
        for position, next_features in zip(batch_positions, next_states):
            action_count = len(next_features)
            action_values = online_values[offset:offset + action_count]
            best_action = int(np.argmax(action_values))
            future_value = target_values[offset + best_action]
            expected_values[position] += self.gamma * future_value
            offset += action_count
        return expected_values

    def _gradients(self, inputs, expected_values):
        input_weights, input_bias, output_weights, output_bias = self.weights
        hidden_layer = np.tanh(inputs @ input_weights + input_bias)
        predictions = (hidden_layer @ output_weights + output_bias).ravel()

        errors = np.clip(predictions - expected_values, -1., 1.)
        output_gradient = (errors / len(inputs))[:, None]
        activation_derivative = 1 - hidden_layer * hidden_layer
        hidden_gradient = (output_gradient @ output_weights.T) * activation_derivative

        input_weight_gradient = inputs.T @ hidden_gradient
        input_bias_gradient = hidden_gradient.sum(axis=0)
        output_weight_gradient = hidden_layer.T @ output_gradient
        output_bias_gradient = output_gradient.sum(axis=0)
        return [
            input_weight_gradient, input_bias_gradient,
            output_weight_gradient, output_bias_gradient,
        ]

    def _update_weights(self, gradients):
        self.updates += 1
        for index, gradient in enumerate(gradients):
            self.moments[index] = .9 * self.moments[index] + .1 * gradient
            self.variances[index] = .999 * self.variances[index] + .001 * gradient * gradient
            corrected_moment = self.moments[index] / (1 - .9 ** self.updates)
            corrected_variance = self.variances[index] / (1 - .999 ** self.updates)
            adjustment = LEARNING_RATE * corrected_moment / (np.sqrt(corrected_variance) + 1e-8)
            self.weights[index] -= adjustment

    def learn(self, batch=None):
        if not self.adaptive:
            return
        if batch is None:
            sample_count = min(BATCH_SIZE, len(self.replay))
            if sample_count == 0:
                return
            indices = self.replay_rng.choice(len(self.replay), sample_count, replace=False)
            batch = [self.replay[int(index)] for index in indices]

        inputs = np.stack([experience[0] for experience in batch])
        expected_values = self._training_targets(batch)
        gradients = self._gradients(inputs, expected_values)
        self._update_weights(gradients)

        if self.updates % TARGET_SYNC_EVERY == 0:
            self.target = [weights.copy() for weights in self.weights]

    def act(self, observation):
        if observation.tick == self.last_tick:
            return self.last_action
        if observation.tick < self.last_tick:
            self.reset()

        possible_actions = self.candidates(observation)
        action_features = self.features(observation, possible_actions)
        if self.previous is not None:
            previous_observation, previous_features = self.previous
            self.last_reward = self.reward(previous_observation, observation)
            self.remember(previous_features, self.last_reward, action_features)

        chosen_index = self.choose(action_features)
        self.last_action = possible_actions[chosen_index][0]
        self.previous = (observation, action_features[chosen_index].copy())
        self.last_tick = observation.tick
        return self.last_action

    def on_episode_end(self, result):
        if self._ended:
            return
        self._ended = True
        if self.previous is not None:
            terminal_reward = -10. if result.reason is EpisodeEndReason.ELIMINATED else 0.
            no_next_actions = np.empty((0, FEATURE_COUNT), dtype=np.float32)
            self.remember(self.previous[1], terminal_reward, no_next_actions, terminal=True)
        self.episodes += 1

    def save(self, path=MODEL_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'version': 4,
            'actions': list(ACTION_TYPES),
            'episodes': self.episodes,
            'weights': [weights.tolist() for weights in self.weights],
        }
        temporary_path = path.with_suffix('.tmp')
        temporary_path.write_text(json.dumps(data, allow_nan=False), encoding='utf-8')
        temporary_path.replace(path)
