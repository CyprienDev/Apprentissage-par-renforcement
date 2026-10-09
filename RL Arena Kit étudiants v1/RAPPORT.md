# Agent Q-learning tabulaire

## Méthode actuelle

Le projet est repassé de Double DQN au Q-learning tabulaire. Le fichier
`student_agent.py` ne contient plus de réseau, de gradients, d'Adam, de réseau
cible ou de replay buffer. La table Q remplace les poids du réseau.

Chaque observation publique devient un état discret : catégories de santé,
d'énergie, d'hydratation et de satiété, distance de l'adversaire visible le
plus proche, présence des catégories d'inventaire, objet équipé et collision.
L'agent ne connaît ni les coordonnées cachées ni la santé de son adversaire.

Les actions portent des clés stables indépendantes des références et des
slots. Pour chaque catégorie visible, la cible la plus proche fournit des
actions vers elle, à l'opposé, sur les côtés, et des interactions. La distance
est discrétisée dans la clé. Les déplacements fixes, rotations, repos, tirs
et interactions avec l'inventaire restent disponibles. L'audition fournit
également des possibilités de déplacement par rapport au son le plus intense.
Le choix final consulte uniquement les valeurs Q : aucun seuil de besoin
n'impose une action et aucune règle ne force la fuite ou l'attaque.

La mise à jour est :

```text
Q(s,a) ← Q(s,a) + 0,15 × [r + 0,98 × max Q(s',a') − Q(s,a)]
```

Le maximum utilise seulement les actions disponibles. Une fin de partie
supprime la valeur future. L'exploration epsilon-greedy est utilisée pendant
l'entraînement ; en tournoi, epsilon est désactivé et les égalités sont
simplement départagées au hasard. Une température fixe peut être testée
séparément, sans autoriser de mises à jour.

## Récompense observable

La récompense utilise +0,01 par tick, 10 fois la variation de santé, 4 fois
celle d'hydratation, 3 fois celle de satiété et 0,2 fois celle d'énergie.
Chaque collision enlève 0,15 ; une interaction sans événement de réussite
enlève 0,1. Une cellule locale estimée visitée pour la première fois apporte
0,02. L'inventaire utilise le potentiel `0,2 × (gamma^durée × après − avant)`,
avec correction terminale. La mort reçoit une pénalité terminale de 10.

Ces coefficients définissent le retour d'expérience, pas le comportement.
Les scripts de combat et de défense peuvent ajouter des récompenses hors
tournoi, décrites dans `COMBAT.md`.

## Entraînement et reproduction

```sh
.venv/bin/python train_curriculum.py --practice-episodes 1200 --arena-episodes 40 --seed 1140000 --model artifacts/q_learning/refined/policy.json
.venv/bin/python evaluate_agent.py --model artifacts/policy.json --policies frozen dqn_frozen random --episodes 6 --ticks 2400 --seed 1240000 --output artifacts/q_learning/standard
.venv/bin/python evaluate_agent.py --model artifacts/policy.json --policies frozen dqn_frozen random --episodes 6 --ticks 2400 --seed 1250000 --scenario stress --output artifacts/q_learning/stress
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python check_student_agent.py
.venv/bin/python build_submission.py
```

Les exercices de ressources utilisent des positions et des besoins variables,
puis les arènes alternent standard et curriculum. Les graines 50000–50003
servent à la sélection avec apprentissage désactivé. Les évaluations finales
utilisent d'autres graines et ne mettent jamais à jour la table.

## Limites

La représentation est une approximation : des situations différentes peuvent
partager le même état et la même clé d'action. Les orientations absolues ne
sont pas encodées ; les actions relatives à une cible ou un son permettent de
réutiliser certaines valeurs malgré cette perte d'information. Les états
inconnus n'héritent pas des valeurs de leurs voisins comme avec un réseau.
Le nombre de catégories est donc un compromis entre précision et couverture.

Le code est plus simple à expliquer, mais le Q-learning n'est pas forcément
plus performant que Double DQN. Les anciens résultats ne s'appliquent pas à
cette version. L'environnement local n'implémente pas les ressources qui
réapparaissent chaque tour annoncées pour l'arène finale. Aucun exploit du
moteur n'est intégré.

## Remise

Le JSON porte la version `q_learning_v1` et contient la table, le nombre
d'épisodes et la température. Il est incompatible avec les anciens fichiers
Double DQN, sauvegardés dans `artifacts/double_dqn/snapshot/`.
La remise reste un fichier Python et des données de moins de 10 Mo.
L'agent n'apprend pas pendant les combats, n'enregistre aucun fichier
spontanément et respecte `reset`, `act` et `on_episode_end`.

## Résultats de la version remise

Le checkpoint retenu vient de la campagne affinée : 1200 exercices de 96 tours,
graines 1140000–1141199, puis 40 arènes de 2400 tours maximum, graines
1150000–1150039. Il contient 1240 épisodes, 519 états et 13854 valeurs Q.
Son JSON pèse 652319 octets. La température vaut zéro.
La validation de sélection sur quatre arènes atteint 1467,5 tours.

La première campagne avait utilisé 2000 exercices et 60 arènes, avec un bonus
de survie de 0,05 et d'exploration de 0,1. Le nouveau bonus de survie est 0,01
et celui d'exploration 0,02, afin de réduire leur poids face aux ressources.
La première campagne atteint 1328,25 tours en validation ; ses résultats sont
conservés dans `artifacts/q_learning/initial_standard` et `initial_stress`.

Les mesures finales portent sur six graines inédites par scénario, sans mise
à jour pendant l'évaluation, avec un maximum de 2400 tours.

| Politique | Standard 1240000–1240005 | Stress 1250000–1250005 |
|---|---:|---:|
| Q-learning remis | 1748,17 | 816,17 |
| Double DQN précédent archivé | 1934,83 | 1150,83 |
| Bot aléatoire | 1852,67 | 982,17 |

Ces petits échantillons montrent une régression de la survie avec le retour
au Q-learning. La simplification du code ne constitue pas un gain de
performance. Le temps moyen de décision du Q-learning est d'environ 0,08 ms,
contre environ 2 ms pour le Double DQN dans ces exécutions.

La moyenne des ramassages standard vaut 42, mais elle est trompeuse : sur la
graine 1240002, l'agent ramasse 250 fois et dépose 250 fois le même objet.
Il ne consomme qu'une fois et meurt à 1750 tours. Ce cycle appris est une
limite de la représentation et de l'apprentissage ; aucun correctif de
comportement codé en dur n'a été ajouté pour cacher cet échec. Le diagnostic
est dans `artifacts/q_learning/loop_diagnostic.json`.

Un candidat de défense a aussi été entraîné sur 200 poursuites de 300 tours,
graines 1130000–1130199. Il survit en moyenne 87,42 tours et termine 2 des
12 poursuites de validation, mais sa validation en arène reste à 1316,5.
Un candidat de combat obtient 1316,75 en arène. Le modèle de survie affinée
est donc retenu selon le critère principal du projet. La comparaison est
conservée dans `artifacts/q_learning/model_selection.json`.

Le candidat de combat séparé a été entraîné sur 200 duels, graines
1160000–1160199, à partir du checkpoint intermédiaire de la campagne affinée
après 10 arènes. Il remporte 2 des 12 duels de validation et 1 des 24 duels
supplémentaires 1260000–1260023, contre 0 pour l'ancienne V5 sur ces 24 duels.
Il meurt dans 20 duels contre 18 pour V5 : cela ne démontre pas une meilleure
survie. Ce candidat n'est pas la table chargée par défaut pour la remise.
