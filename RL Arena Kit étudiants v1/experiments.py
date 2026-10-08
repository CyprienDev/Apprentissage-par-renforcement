"""Scénarios d'entraînement et de test. Les détails du moteur restent ici."""
from random import Random
from rl_arena.config import EnvironmentConfig
from rl_arena.environment import TrainingEnvironment
from rl_arena.bots import HunterBot
from rl_arena.arena_api import ObjectKind
from rl_arena.geometry import Rect


def make_environment(seed, ticks, scenario='standard', agent=None):
    config = EnvironmentConfig(max_ticks=ticks)
    rng = Random(seed)
    if scenario in {'curriculum', 'stress'}:
        config.generation.fog_probability = .65
        config.generation.low_light_probability = .5
        config.generation.item_count = (20,30)
        config.physiology.hydration_decay_per_tick = rng.uniform(.0005,.0013)
        config.physiology.satiety_decay_per_tick = rng.uniform(.0003,.0008)
    env = TrainingEnvironment(config)
    env.register_agent('student',agent)
    env.add_default_bots()
    if scenario == 'stress':
        env.register_agent('hunter_extra',HunterBot(seed=seed+17))
    env.reset(seed=seed)
    if scenario in {'curriculum', 'stress'}:
        student = env.debug_state()['agents']['student']
        student.hydration = rng.uniform(.3,.85)
        student.satiety = rng.uniform(.4,.9)
        student.energy = rng.uniform(.25,1.)
        # Ressources dispersées dans des positions libres : pas de carte apprise
        # par coeur. Modifie uniquement l'expérience, jamais la politique.
        for obj in env.world.objects:
            if obj.kind not in {ObjectKind.FOOD,ObjectKind.DRINK,ObjectKind.MEDICINE}:
                continue
            for _ in range(80):
                x,y = rng.uniform(1,29),rng.uniform(1,29)
                if not env._circle_hits_solid(x,y,.8):
                    obj.x,obj.y = x,y
                    break
        if scenario == 'curriculum' and seed % 2 == 0:
            for door in env.world.doors:
                door.open = True
        turns = rng.randrange(4)
        shift_x,shift_y = rng.uniform(-10,10),rng.uniform(-10,10)
        def transform(x,y):
            for _ in range(turns):
                x,y = 30-y,x
            return (x+shift_x)%30,(y+shift_y)%30
        for entity in [*env.world.rects,*env.world.doors,*env.world.areas]:
            rect = entity.rect
            x,y = transform(rect.x,rect.y)
            entity.rect = Rect(x,y,rect.h if turns%2 else rect.w,rect.w if turns%2 else rect.h)
        for obj in env.world.objects:
            obj.x,obj.y = transform(obj.x,obj.y)
            if obj.kind is ObjectKind.MOVING_OBSTACLE:
                obj.properties['origin_x'],obj.properties['origin_y'] = transform(
                    float(obj.properties.get('origin_x',0)),float(obj.properties.get('origin_y',0)))
                if turns%2:
                    obj.properties['axis'] = 'y' if obj.properties.get('axis') == 'x' else 'x'
        # Réassigner des positions libres après transformation de la géométrie.
        positions = []
        for state in env.debug_state()['agents'].values():
            for _ in range(500):
                x,y = rng.uniform(0,30),rng.uniform(0,30)
                if not env._circle_hits_solid(x,y,.8) and all((x-p[0])**2+(y-p[1])**2 > 9 for p in positions):
                    state.x,state.y = x,y
                    positions.append((x,y))
                    break
        env._last_observations = env._observe_all()
    return env
