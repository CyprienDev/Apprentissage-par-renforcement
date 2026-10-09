# Remettre l'agent Q-learning

La remise comprend uniquement `student_agent.py` et `artifacts/policy.json`.
Le JSON contient une table Q avec la version `q_learning_v1`, pas les anciens
poids Double DQN. Le programme vérifie que les données ne dépassent pas 10 Mo.

Depuis le dossier du kit :

```sh
.venv/bin/python build_submission.py
```

Cela produit `artifacts/submission.zip`. Le professeur doit fournir Python et
`rl_arena.arena_api`. NumPy est utilisé par certains outils de test du kit,
mais le nouvel agent lui-même utilise seulement la bibliothèque standard et
le contrat public.

Le professeur instancie `MyAgent()`, puis appelle `reset()`, `act(observation)`
et `on_episode_end(result)`. Par défaut, la table est figée : ni les décisions,
ni la fin de partie, ni un reset ne provoquent d'apprentissage.

L'entraînement est autorisé avant le tournoi. Si une chauffe autorise des
mises à jour sur la même instance, appeler `set_training_enabled(True)` au
début, puis `set_training_enabled(False)` avant les combats. L'API ne signale
pas cette frontière automatiquement.

Pour lancer votre agent :

```sh
.venv/bin/python run_student_agent.py --gui --seed 42
```

Pour repartir sans données, utiliser `--cold`. Une table vide n'est pas un
agent entraîné et ne saura pas automatiquement survivre.
