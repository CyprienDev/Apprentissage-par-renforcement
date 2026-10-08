# Agent de survie

## Objectif et interface

L'objectif principal est la durée de survie. Chaque action concerne 0,5 seconde simulée. La politique ne consulte que `Observation` et les types de `rl_arena.arena_api`. Le moteur fourni reste inchangé. Les références d'objets sont utilisées uniquement au tick courant. Les propriétés optionnelles ne sont jamais supposées présentes.

## Représentation et apprentissage

L'état discret comporte neuf composantes : santé basse, énergie basse, énergie moyenne, besoin d'eau, besoin de nourriture, distance de menace en trois classes, présence de consommables, contact et dégâts récents. Il y a au plus 768 états binaires/ternaires (certaines combinaisons d'énergie sont impossibles). L'agent conserve temporairement une alerte de menace pendant huit ticks ; aucune référence d'objet n'est mémorisée.

Cinq stratégies sont sélectionnées : explorer, chercher des consommables, fuir, se reposer et scanner. Chaque stratégie transforme les perceptions en actions continues. Des règles prioritaires assurent les soins, l'ouverture des portes, l'évitement des obstacles et empêchent le repos à proximité d'un ennemi visible. Le Q-learning apprend donc une sélection de stratégies sous ces contraintes, plutôt que chaque combinaison de commandes physiques.

La mise à jour est `Q(s,a) += 0.12 * (r + 0.98 * max Q(s',.) - Q(s,a))`. En fin d'épisode, le terme futur est nul, y compris à la limite de simulation (objectif de durée finie). La récompense d'apprentissage est +0,1 par tick, moins quatre fois la perte de santé et moins 20 en cas d'élimination. Elle diffère volontairement de la récompense du kit, qui valorise aussi les éliminations et ramassages. Les journaux enregistrent la récompense du kit ; la sélection du modèle repose sur la survie.

L'exploration epsilon-greedy décroît de 0,5 à 0,05. Les valeurs initiales favorisent le repos lorsque l'énergie manque, les ressources lorsqu'elles sont visibles et la fuite lorsqu'une menace apparaît. Les états inconnus utilisent ce prior, sans modifier la table en évaluation.

## Protocole expérimental

Par défaut : 100 épisodes de 600 ticks, graines 1000 à 1099 ; validation sur quatre graines 50000 à 50003 toutes les dix séances ; test indépendant sur vingt graines à partir de 90000. Le modèle sélectionné maximise la survie moyenne de validation ; à égalité le modèle le plus récent est retenu. Le dernier modèle est enregistré séparément. `--resume` reprend le modèle sélectionné ; changer `--seed` permet de poursuivre sur de nouvelles arènes.

L'évaluation rejoue chaque graine pour chacune des quatre politiques avec les deux mêmes adversaires. Les CSV présentent les observations de survie, ramassages, dégâts et récompenses. Le JSON résume moyenne, médiane, minimum et maximum. Une survie de 600 ticks est censurée par la limite de temps : elle indique seulement une survie d'au moins 300 secondes. Elle ne mesure pas la survie sans limite et ne départage pas les agents qui atteignent tous cette limite.

## Limites et pistes

Le contrat permet la compétition finale, mais les résultats sur les cartes du kit ne garantissent pas une victoire sur une arène inconnue. Le modèle est tabulaire, la représentation compacte perd beaucoup d'information et la mémoire ne suit pas les ressources cachées. L'agent privilégie l'évitement et ne développe pas de combat armé. Les règles de navigation et leurs seuils restent sensibles à des changements de géométrie. La validation sur quatre graines est petite ; augmenter ce nombre et répéter l'apprentissage avec plusieurs graines donnerait une estimation plus solide. Des essais plus longs sont particulièrement nécessaires lorsque les scores atteignent la limite.

Le code historique de labyrinthe est conservé et n'appartient pas au livrable d'arène. Ses tests ne peuvent pas s'exécuter sans le module historique manquant ; les tests de l'arène se lancent depuis le dossier du kit.

## Résultats obtenus dans ce dépôt

Le modèle livré a été entraîné pendant 100 épisodes sur les graines 1000–1099, puis dix épisodes supplémentaires sur 2000–2009 après vérification de la direction de fuite. Le premier journal est `artifacts/initial_training.csv` ; la continuation est `artifacts/training.csv`. Le modèle compte 110 épisodes.

Sur vingt arènes de test (90000–90019), les quatre politiques atteignent la limite de 600 ticks : ces essais ne permettent pas de départager les politiques. Sur cinq arènes prolongées (95000–95004, limite de 2000 ticks), les résultats sont :

| Politique | Survie moyenne (ticks) | Secondes simulées |
|---|---:|---:|
| Agent appris | 1856,6 | 928,3 |
| Prior sans apprentissage | 1856,6 | 928,3 |
| RandomBot | 1800,0 | 900,0 |
| HunterBot | 1756,6 | 878,3 |

L'agent obtient une meilleure moyenne que les deux bots sur ce petit échantillon, mais **aucun gain dû à l'apprentissage n'est démontré par rapport au prior**. Il serait incorrect de présenter ces résultats comme une preuve de supériorité du Q-learning. La survie sur les essais prolongés dépend notamment de l'épuisement des ressources ; l'horizon d'entraînement de 600 ticks ne couvre pas bien cette phase. Une suite naturelle est l'apprentissage sur des épisodes prolongés et la validation sur davantage de graines.

Les douze tests du kit et de l'agent réussissent. Le vérificateur de contrat réussit. Les bibliothèques NumPy, scikit-learn et Arcade ont été installées dans le venv du kit ; une fenêtre Arcade a été créée, dessinée et fermée sans erreur.

Reproduire le test prolongé :

```bash
python evaluate_agent.py --episodes 5 --ticks 2000 --seed 95000 --output artifacts/long
```
