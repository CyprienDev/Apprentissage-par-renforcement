"""Ablations sur des graines inédites, comparaison appariée et coût de décision."""
import argparse
import csv
import json
import importlib.util
from pathlib import Path
from dataclasses import asdict
from statistics import mean,median
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from rl_arena.bots import RandomBot,HunterBot
from experiments import make_environment
from student_agent import MyAgent,MODEL_PATH
from baseline_agent import MyAgent as BaselineAgent
from previous_agent_v2 import MyAgent as PreviousAgent


def snapshot_agent(snapshot, seed):
    spec = importlib.util.spec_from_file_location('old_frozen_agent',snapshot/'student_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent(snapshot/'policy.json',seed=seed,adaptive=False)


def old_frozen_agent(seed):
    return snapshot_agent(MODEL_PATH.parent/'v4'/'snapshot',seed)


class TimedAgent:
    def __init__(self,agent):
        self.agent = agent
        self.times = []
    def reset(self):
        self.agent.reset()
    def act(self,obs):
        start = perf_counter()
        action = self.agent.act(obs)
        self.times.append(perf_counter()-start)
        return action
    def on_episode_end(self,result):
        self.agent.on_episode_end(result)


def evaluate_one(task):
    name,seed,ticks,scenario,model = task[:5]
    temperature = task[5] if len(task) > 5 else None
    factories = {
        'adaptive':lambda:MyAgent(model,seed=seed,adaptive=True),
        'frozen':lambda:MyAgent(model,seed=seed,adaptive=False,epsilon=0,temperature=temperature),
        'cold_adaptive':lambda:MyAgent(seed=seed,load=False,adaptive=True),
        'cold_frozen':lambda:MyAgent(seed=seed,load=False,adaptive=False,epsilon=0),
        'v1':lambda:BaselineAgent(model_path=MODEL_PATH.parent/'v1'/'policy.json',seed=seed),
        'v2':lambda:PreviousAgent(model_path=MODEL_PATH.parent/'v2'/'snapshot'/'policy.json',seed=seed),
        'v2_cold':lambda:PreviousAgent(load=False,seed=seed),
        'random':lambda:RandomBot(seed=seed),
        'hunter':lambda:HunterBot(seed=seed)}
    factories['old_frozen'] = lambda:old_frozen_agent(seed)
    factories['phase1_frozen'] = lambda:snapshot_agent(MODEL_PATH.parent/'v5'/'phase1',seed)
    factories['v5_frozen'] = lambda:snapshot_agent(MODEL_PATH.parent/'v5'/'snapshot',seed)
    factories['dqn_frozen'] = lambda:snapshot_agent(MODEL_PATH.parent/'double_dqn'/'snapshot',seed)
    agent = TimedAgent(factories[name]())
    env = make_environment(seed,ticks,scenario,agent)
    while True:
        result = env.step({})
        if result.terminated['student'] or result.truncated:
            break
    times = sorted(agent.times)
    return dict(policy=name,seed=seed,scenario=scenario,**asdict(result.info['student']),
                reached_limit=not result.terminated['student'],
                online_transitions=getattr(agent.agent,'online_transitions',0),
                act_mean_ms=1000*mean(times),act_p95_ms=1000*times[min(len(times)-1,int(.95*len(times)))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=12)
    parser.add_argument('--seed',type=int,default=100000)
    parser.add_argument('--ticks',type=int,default=3000)
    parser.add_argument('--scenario',choices=['standard','stress'],default='standard')
    parser.add_argument('--output',type=Path,default=MODEL_PATH.parent/'q_learning'/'evaluation')
    parser.add_argument('--model',type=Path,default=MODEL_PATH)
    parser.add_argument('--temperature',type=float,default=None)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--policies',nargs='+',choices=['adaptive','frozen','cold_adaptive','cold_frozen','v1','v2','v2_cold','old_frozen','phase1_frozen','v5_frozen','dqn_frozen','random','hunter'],
                        default=['frozen','dqn_frozen','random'])
    args = parser.parse_args()
    if min(args.episodes,args.ticks,args.workers) < 1:
        parser.error('les nombres doivent être positifs')
    tasks = [(name,seed,args.ticks,args.scenario,str(args.model),args.temperature) for name in args.policies
             for seed in range(args.seed,args.seed+args.episodes)]
    rows = []
    args.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output/'evaluation.csv').open('w',newline='',encoding='utf-8') as f:
        writer = None
        for row in pool.map(evaluate_one,tasks):
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(f,fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(f"{row['policy']} seed={row['seed']} survival={row['survival_ticks']} pickups={row['pickups']}",flush=True)
    summaries = {}
    for name in args.policies:
        group = [r for r in rows if r['policy']==name]
        scores = [r['survival_ticks'] for r in group]
        summaries[name] = dict(mean_ticks=mean(scores),median_ticks=median(scores),min_ticks=min(scores),max_ticks=max(scores),
            mean_pickups=mean(r['pickups'] for r in group),reached_limit=sum(r['reached_limit'] for r in group),
            mean_kills=mean(r['kills'] for r in group),mean_damage_inflicted=mean(r['damage_inflicted'] for r in group),
            act_mean_ms=mean(r['act_mean_ms'] for r in group))
        print(name,summaries[name],flush=True)
    paired = {}
    if 'adaptive' in args.policies:
        adaptive = {r['seed']:r['survival_ticks'] for r in rows if r['policy']=='adaptive'}
        for name in args.policies:
            if name == 'adaptive':
                continue
            diffs = [adaptive[r['seed']]-r['survival_ticks'] for r in rows if r['policy']==name]
            paired[name] = dict(mean_gain_ticks=mean(diffs),wins=sum(d>0 for d in diffs),ties=sum(d==0 for d in diffs),losses=sum(d<0 for d in diffs))
    (args.output/'evaluation.json').write_text(json.dumps(dict(seed=args.seed,episodes=args.episodes,
        ticks=args.ticks,scenario=args.scenario,temperature=args.temperature,
        results=summaries,paired_vs_adaptive=paired),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
