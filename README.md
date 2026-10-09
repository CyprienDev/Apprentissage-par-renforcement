# Agent Q-learning pour l'arène

Le projet actif se trouve dans **RL Arena Kit étudiants v1**.
`student_agent.py` utilise désormais du **Q-learning tabulaire** : il mémorise
les valeurs des actions dans une table, sans réseau de neurones et sans
règle imposant repos, fuite, collecte ou attaque.

Depuis le dossier du kit :

```sh
.venv/bin/python run_student_agent.py --gui --seed 42
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python check_student_agent.py
.venv/bin/python build_submission.py
```

`MyAgent()` charge la table de `artifacts/policy.json` et la garde figée pendant
le tournoi. Une chauffe doit être activée explicitement et désactivée avant
les combats. `reset()` n'efface pas la table et ne réactive pas l'entraînement.

Le guide pour les trois membres du groupe est `LIRE_AGENT.md`. Le protocole,
les résultats et les limites sont dans `RAPPORT.md`. La remise est expliquée
dans `SOUMISSION.md`.

Les sources et le modèle Double DQN précédents sont archivés dans
`artifacts/double_dqn/snapshot/`. Leur JSON est incompatible avec le nouvel
agent et leurs anciennes performances ne décrivent pas le Q-learning.

Les dossiers `legacy/`, `data/` et les tests à la racine appartiennent à l'ancien
projet de labyrinthe : lancer les tests actifs depuis le dossier du kit.
