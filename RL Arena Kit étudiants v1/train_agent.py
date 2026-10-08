"""Curriculum, adaptation en ligne et validation sans fuite de données."""
import argparse
import csv
from pathlib import Path
from statistics import mean
from experiments import make_environment
from student_agent import MyAgent, MODEL_PATH


def episode(agent, seed, ticks=2400, scenario='standard'):
    env = make_environment(seed,ticks,scenario,agent)
    while True:
        result = env.step({})
        if result.terminated['student'] or result.truncated:
            return result.info['student']


def validation(agent, ticks, count):
    scores = []
    for n in range(count):
        clone = MyAgent(load=False,seed=50000+n,adaptive=True)
        clone.q = {k:list(v) for k,v in agent.q.items()}
        scores.append(episode(clone,50000+n,ticks,'standard' if n%2 else 'stress').survival_ticks)
    return mean(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=60)
    parser.add_argument('--ticks',type=int,default=2400)
    parser.add_argument('--seed',type=int,default=3000)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--model',type=Path,default=MODEL_PATH)
    parser.add_argument('--validation-episodes',type=int,default=6)
    parser.add_argument('--validate-every',type=int,default=10)
    args = parser.parse_args()
    if min(args.episodes,args.ticks,args.validation_episodes,args.validate_every) < 1:
        parser.error('les nombres doivent être positifs')
    agent = MyAgent(model_path=args.model,load=args.resume,seed=args.seed)
    best = validation(agent,args.ticks,args.validation_episodes)
    agent.save(args.model)
    log_path = args.model.with_name('training_v2.csv')
    with log_path.open('a' if args.resume else 'w',newline='',encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(['episode','seed','scenario','epsilon','survival_ticks','pickups','damage_received','online_transitions','validation'])
        for n in range(args.episodes):
            agent.epsilon = max(.025,.18*(1-n/args.episodes))
            scenario = 'standard' if n%3 == 0 else 'curriculum'
            info = episode(agent,args.seed+n,args.ticks,scenario)
            score = ''
            if (n+1)%args.validate_every == 0 or n+1 == args.episodes:
                score = validation(agent,args.ticks,args.validation_episodes)
                if score >= best:
                    best = score
                    agent.save(args.model)
            writer.writerow([agent.episodes,args.seed+n,scenario,agent.epsilon,info.survival_ticks,
                             info.pickups,info.damage_received,agent.online_transitions,score])
            handle.flush()
            print(f'episode={agent.episodes} scenario={scenario} survival={info.survival_ticks} pickups={info.pickups} updates={agent.online_transitions} validation={score}',flush=True)
    agent.save(args.model.with_name('last_policy_v2.json'))
    print(f'Modèle sélectionné : {args.model}; validation={best:.1f}')


if __name__ == '__main__':
    main()
