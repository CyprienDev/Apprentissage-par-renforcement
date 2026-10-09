"""Entraînement Q-learning hors tournoi et sélection sur une validation figée."""
import argparse
import csv
from copy import deepcopy
from pathlib import Path
from statistics import mean

from experiments import make_environment
from student_agent import MyAgent, MODEL_PATH


def episode(agent, seed, ticks=2400, scenario='standard'):
    env = make_environment(seed, ticks, scenario, agent)
    while True:
        result = env.step({})
        if result.terminated['student'] or result.truncated:
            return result.info['student']


def validation(agent, ticks, count):
    scores = []
    for index in range(count):
        clone = MyAgent(load=False, seed=50000 + index, adaptive=False)
        clone.q_table = deepcopy(agent.q_table)
        clone.temperature = agent.temperature
        scenario = 'standard' if index % 2 else 'stress'
        scores.append(episode(clone, 50000 + index, ticks, scenario).survival_ticks)
    return mean(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=60)
    parser.add_argument('--ticks', type=int, default=2400)
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    parser.add_argument('--validation-episodes', type=int, default=6)
    parser.add_argument('--validate-every', type=int, default=10)
    parser.add_argument('--defense', action='store_true')
    args = parser.parse_args()
    if min(args.episodes, args.ticks, args.validation_episodes, args.validate_every) < 1:
        parser.error('Les nombres doivent être positifs')
    if args.defense:
        from train_defense import train
        output = args.model if args.model != MODEL_PATH else MODEL_PATH.parent / 'defense' / 'policy.json'
        train(args.episodes, args.ticks, args.seed, MODEL_PATH, output)
        return
    agent = MyAgent(args.model, load=args.resume, seed=args.seed, adaptive=True)
    best = validation(agent, args.ticks, args.validation_episodes)
    agent.save(args.model)
    print(f'Validation initiale : {best:.1f}', flush=True)
    with args.model.with_name('training_q_learning.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['episode', 'seed', 'survival', 'pickups', 'updates', 'states', 'validation'])
        for index in range(args.episodes):
            agent.epsilon = max(.05, .5 * (1 - index / args.episodes))
            scenario = 'standard' if index % 3 == 0 else 'curriculum'
            info = episode(agent, args.seed + index, args.ticks, scenario)
            score = ''
            if (index + 1) % args.validate_every == 0 or index + 1 == args.episodes:
                score = validation(agent, args.ticks, args.validation_episodes)
                if score >= best:
                    best = score
                    agent.save(args.model)
            writer.writerow([agent.episodes, args.seed + index, info.survival_ticks, info.pickups,
                             agent.updates, len(agent.q_table), score])
            handle.flush()
            print(f'episode={index + 1} survival={info.survival_ticks} pickups={info.pickups} '
                  f'states={len(agent.q_table)} validation={score}', flush=True)
    agent.save(args.model.with_name('last_policy_q_learning.json'))
    print(f'Modèle retenu : {args.model}; validation={best:.1f}', flush=True)


if __name__ == '__main__':
    main()
