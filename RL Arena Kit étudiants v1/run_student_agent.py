"""Lance votre agent contre les deux bots de référence.

Usage :
    python run_student_agent.py          # mode headless
    python run_student_agent.py --gui    # visualisation Arcade
    python run_student_agent.py --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_arena.environment import TrainingEnvironment
from student_agent import MyAgent


def build_env(gui: bool, seed: int, cold: bool = False, frozen: bool = True,
              agent=None) -> TrainingEnvironment:
    env = TrainingEnvironment(render=gui)
    env.register_agent("student", agent if agent is not None else MyAgent(
        seed=seed, load=not cold, adaptive=not frozen, epsilon=0 if frozen else .10))
    env.add_default_bots()
    env.reset(seed=seed)
    return env


def run_headless(env: TrainingEnvironment) -> None:
    result = None
    while True:
        result = env.step({})
        if result.truncated or result.terminated.get("student", False):
            break

    assert result is not None
    info = result.info["student"]
    print("Episode terminé")
    print(f"  survie : {info.survival_ticks} ticks ({info.survival_ticks * 0.5:.1f} s simulées)")
    print(f"  reward cumulé : {info.cumulative_reward:.2f}")
    print(f"  kills : {info.kills}")
    print(f"  pickups : {info.pickups}")
    print(f"  dégâts reçus : {info.damage_received:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="ouvrir l'interface Arcade")
    parser.add_argument("--seed", type=int, default=42, help="seed de l'épisode")
    parser.add_argument("--cold", action="store_true", help="démarrer avec une table Q vide")
    parser.add_argument("--frozen", action="store_true", help="évaluer sans exploration ni mise à jour")
    parser.add_argument("--warmup", action="store_true", help="activer l'apprentissage hors tournoi")
    parser.add_argument("--save-model", type=Path, help="enregistrer la table Q après la chauffe")
    parser.add_argument("--model", type=Path, help="table Q à charger à la place du modèle habituel")
    args = parser.parse_args()
    if args.warmup and args.frozen:
        parser.error('--warmup et --frozen sont incompatibles')
    if args.save_model and not args.warmup:
        parser.error('--save-model nécessite --warmup')

    model_options = {'model_path': args.model} if args.model is not None else {}
    agent = MyAgent(seed=args.seed, load=not args.cold, adaptive=args.warmup, **model_options)
    env = build_env(args.gui, args.seed, args.cold, not args.warmup, agent)
    if args.gui:
        from rl_arena.render_arcade import run_arcade
        run_arcade(env)
    else:
        run_headless(env)
    if args.save_model:
        agent.set_training_enabled(False)
        agent.save(args.save_model)
        print(f'Modèle après chauffe : {args.save_model}')


if __name__ == "__main__":
    main()
