# Agent de survie adaptatif — apprentissage par renforcement

Le projet actif est dans **RL Arena Kit étudiants v1**. `student_agent.py` contient la version 3 : apprentissage pendant chaque match, cibles filtrées selon l'inventaire, gestion des bandages, récupération d'énergie prolongée, mémoire des ressources, exploration des portes et défense pendant la fuite. Le contrat public est respecté ; la décision ne consulte pas le moteur d'entraînement.

## Lancer

Depuis le dossier du kit :

```bash
source .venv/bin/activate
python check_student_agent.py
python run_student_agent.py --seed 42
python run_student_agent.py --gui --seed 42
python -m unittest discover -s tests -v
```

L'environnement du kit a été configuré dans PyCharm. Sur une autre machine, créer `.venv`, l'activer et installer `requirements.txt`. Le fonctionnement sans fenêtre nécessite uniquement Python et le kit. Les versions installées localement sont dans `requirements-lock.txt`.

## Entraîner et comparer

```bash
python train_agent.py --episodes 60 --ticks 3000
python train_agent.py --resume --episodes 60 --ticks 3000 --seed 8000
python evaluate_agent.py --episodes 12 --ticks 3000
python evaluate_agent.py --scenario stress --episodes 8 --ticks 3000 --seed 130000 --output artifacts/v2/stress
```

L'agent apprend automatiquement dès qu'il reçoit l'observation qui suit sa première action. Il ne faut pas lancer le script d'entraînement dans l'arène finale. La table Q survit aux resets de la même instance ; la mémoire de la carte est réinitialisée. Le calibrage énergétique utilise huit mesures de déplacement valides. Le code est livré avec des priors de survie : une carte inconnue n'impose pas de repartir sans aucun comportement appris.

`train_agent.py` mélange des expériences standard et un curriculum avec ressources dispersées, géométrie transformée et physiologie variable. La validation utilise des clones indépendants. `evaluate_agent.py` compare le modèle adaptatif, sa version figée, les versions sans préentraînement, l'ancien agent et les bots ; le CSV conserve chaque résultat, le JSON présente les comparaisons appariées et les temps de décision.

Les résultats actuels sont dans `artifacts/v3/standard` et `artifacts/v3/stress`, avec leur interprétation dans `RAPPORT.md`. V3 sans checkpoint améliore la survie moyenne en stress de 14,9 % face à V2 avec son modèle, mais perd 4,7 % en standard : le gain n'est pas uniforme.

Pour remettre un agent sans mémoire préentraînée, envoyer `artifacts/submission_cold.zip`. Il contient une copie autonome de `student_agent.py` avec `load=False` par défaut et aucun checkpoint. Pour une remise autorisant le modèle, utiliser `artifacts/submission.zip`. Générer les archives avec `python build_submission.py --cold` et `python build_submission.py`.

Les archives V1 sont dans `artifacts/v1` ; V2 est conservée dans `previous_agent_v2.py` et `artifacts/v2/snapshot`. Les variantes de développement sont séparées des tests finaux. Les anciennes tables V1/V2 ne sont pas réutilisées dans V3.

Les dossiers `legacy/`, `data/` et les tests à la racine sont une ancienne archive de labyrinthe. Ils ne sont pas utilisés pour l'arène ; leurs tests dépendent d'un module historique `main` absent. Lancer les tests actifs depuis le dossier du kit.
