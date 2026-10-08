# Agent adaptatif de survie — version 2

## Ce que l'agent sait en arrivant

Le contrat d'observation/action reste connu ; la carte, les ressources accessibles et le comportement des adversaires restent inconnus. L'agent arrive avec un prior de survie et une table Q préentraînée, puis apprend pendant le match. Il n'a besoin ni du script d'entraînement, ni d'une récompense fournie par l'arène finale : les conséquences observées de ses actions suffisent à calculer son signal d'apprentissage.

L'agent ne consulte jamais l'état privé du moteur. `student_agent.py` utilise uniquement la bibliothèque standard et `rl_arena.arena_api`. Les références `ref` ne sont jamais mémorisées d'un tick à l'autre.

## Apprendre rapidement sans sacrifier la survie

À la réception de chaque nouvelle observation, l'agent évalue son action précédente : changement de santé, énergie, hydratation et satiété, collisions, ramassages et exploration de cellules nouvelles. La table Q est mise à jour immédiatement, puis quatre expériences récentes sont rejouées. Le tampon est limité à 256 transitions. Le taux d'apprentissage direct vaut 0,2 ; le rejeu utilise 0,08 ; le facteur d'actualisation vaut 0,97. Les mises à jour de fin d'épisode ne bootstrapent pas ; une élimination reçoit une pénalité terminale de -12.

La récompense observable par tick est :

```
0.025 + 20*gain_hydratation + 12*gain_satiete + 8*variation_sante
      + 0.25*variation_energie + 0.08*cellule_nouvelle
      - 0.3*min(2, collisions) + 0.3*ramassages
```

Un blocage prolongé ajoute une pénalité de 0,15. Cette récompense favorise la survie et l'accès aux ressources, sans prime aux éliminations. L'évaluation finale repose uniquement sur les durées de survie.

Les valeurs apprises ne commandent pas librement toutes les décisions : le score de sélection combine le prior et un ajustement borné `0.5*tanh((Q(a)-maxQ)/2)`. Les soins, la fuite face à une menace proche, le repos à énergie critique et les ressources vitales connues passent avant ce score. L'exploration aléatoire est limitée aux stratégies autorisées et désactivée lorsqu'un adversaire proche est visible. Cela réduit le risque qu'une expérience bruitée fasse abandonner une décision vitale.

L'agent identifie aussi son coût énergétique de déplacement et sa récupération au repos par moyennes mobiles. Après huit mesures de marche valides, il ajuste la vitesse pour maximiser la distance estimée par cycle marche/repos, selon un modèle de coût quadratique. Les vitesses restent bornées et sont réduites près des obstacles. Huit ticks correspondent à quatre secondes simulées, mais les collisions, le repos ou une énergie initiale basse peuvent retarder l'identification. Ce délai décrit le calibrage énergétique ; il ne promet pas un apprentissage complet de l'arène en quatre secondes.

## Mémoire et navigation

La position est une estimation locale par intégration de la vitesse observée et de la commande précédente ; ce n'est pas une coordonnée privilégiée du moteur. Elle sert à éviter de revisiter les mêmes cellules et à retrouver approximativement les consommables vus récemment. La mémoire des ressources expire après 120 ticks et garde au plus 32 entrées. Les erreurs de mouvement et les collisions peuvent faire dériver cette estimation.

Le navigateur compare douze directions selon l'objectif, les obstacles et les visites précédentes. Il conserve un sens de détour pendant un blocage pour éviter les oscillations. Les murs sont traités par leur point observable le plus proche : leur longueur apparente ne doit pas fermer artificiellement les passages. L'agent explore les portes découvertes afin de voir les ressources occultées, ouvre les portes proches et traverse l'entrée.

Les faibles dégâts compatibles avec la déshydratation ou les saignements ne déclenchent pas automatiquement une fuite sans fin. Les dégâts plus importants restent considérés comme dangereux. Les médicaments traitent également les blessures ; nourriture et boisson sont consommées selon les besoins. L'agent peut récupérer et équiper une arme et l'utiliser pendant une fuite si sa fonction est reconnue. Il ne poursuit pas un adversaire uniquement pour augmenter les éliminations.

## Entraînement et sélection

Le curriculum alterne les arènes standard et des expériences avec ressources dispersées, visibilité dégradée, physiologie différente et géométrie tournée/translatée. La validation stress ajoute un chasseur. Les modifications du monde appartiennent uniquement à `experiments.py` ; elles ne sont jamais fournies à l'agent comme information cachée.

Les clones de validation démarrent avec la table du candidat et s'adaptent durant leur match. Leur apprentissage est ensuite jeté, sans contaminer le candidat. Le meilleur checkpoint maximise la survie moyenne sur les graines de validation. Les données de test utilisent d'autres graines. Les essais pilotes et anciennes configurations sont conservés séparément et ne constituent pas des preuves de performance sur des arènes inédites.

## Reproduction

Depuis le dossier du kit et son venv activé :

```bash
python train_agent.py --episodes 32 --ticks 3000 --seed 6000 --validation-episodes 6 --validate-every 8
python evaluate_agent.py --episodes 12 --ticks 3000 --seed 120000 --output artifacts/v2/standard
python evaluate_agent.py --episodes 8 --ticks 3000 --seed 130000 --scenario stress --output artifacts/v2/stress
python -m unittest discover -s tests -v
python check_student_agent.py
python run_student_agent.py --gui --seed 42
```

`adaptive` et `frozen` utilisent le même modèle : le second désactive les mises à jour Q et l'identification énergétique. Cette ablation mesure les effets conjoints de ces deux adaptations, et non le Q-learning seul. `cold_adaptive` et `cold_frozen` partent sans table préentraînée. `v1` reprend l'ancien agent et son checkpoint. Les politiques Random/Hunter servent de références supplémentaires. Chaque politique rejoue les mêmes arènes ; les différences de survie sont calculées graine par graine. Les durées à 3000 ticks sont censurées par cette limite.

Les connaissances Q persistent lors de `reset()` sur la même instance, tandis que les souvenirs de carte, les expériences rejouées et les paramètres physiques estimés sont réinitialisés. Aucun fichier n'est écrit automatiquement pendant un match. Pour conserver les connaissances entre deux processus, le responsable d'exécution peut appeler `agent.save(path)` si le règlement permet cette sauvegarde.

## Limites

L'adaptation suppose que le contrat et les catégories d'objets restent les mêmes. Une arène différente peut invalider des seuils ou fournir moins d'expériences utiles. Une seule rencontre fatale ne permet pas toujours d'apprendre une défense à temps. La mémoire spatiale est approximative, la navigation reste locale et certaines ressources derrière des obstacles peuvent demeurer inaccessibles. Les résultats doivent être examinés sur plusieurs graines et les changements de carte les plus proches de la compétition. Aucune méthode ne garantit d'être la plus forte sur une arène inconnue.

## Résultats finaux de cette version

Le checkpoint livré compte 32 épisodes de la dernière campagne (graines 6000–6031, limite de 3000 ticks). La sélection utilise six graines de validation 50000–50005, dont trois scénarios stress. Le score moyen de validation du checkpoint sélectionné est 2092,17 ticks, contre 2073,33 pour le candidat initial. Ce faible gain de validation ne prouve pas une généralisation du préentraînement.

Les tests finaux utilisent douze graines standard 120000–120011 et huit graines stress 130000–130007, distinctes des graines de développement et de validation :

| Politique | Standard : survie moyenne, ticks | Stress : survie moyenne, ticks |
|---|---:|---:|
| V2 adaptative | 2499,92 | 1192,38 |
| V2 figée | 2367,92 | 1240,00 |
| V1 précédente | 1898,58 | 1331,00 |
| RandomBot | 1895,00 | 941,00 |
| HunterBot | 1752,75 | 669,25 |

Sur les douze arènes standard, V2 adaptative améliore la survie moyenne de **31,7 % par rapport à V1**, et de **5,6 % par rapport à V2 figée**. Elle bat V1 sur onze graines et fait égalité sur la douzième. Elle atteint la limite de 3000 ticks sur quatre graines ; ces quatre survies réelles pourraient être plus longues.

Sur les huit scénarios stress, V2 adaptative reste meilleure que RandomBot et HunterBot en moyenne, mais elle est **10,4 % sous V1** et **3,8 % sous V2 figée**. L'adaptation n'est donc pas uniformément bénéfique. Ce scénario cumule ressources initiales plus faibles, physiologie modifiée, positions de ressources différentes, visibilité réduite, géométrie transformée et adversaire supplémentaire.

Sur les arènes standard, les politiques adaptatives avec et sans table préentraînée donnent exactement les mêmes survies. Les versions figées avec et sans préentraînement donnent également les mêmes survies. Le gain du préentraînement Q n'est donc **pas démontré sur ce test**. L'amélioration globale de V2 tient aussi aux règles de navigation, aux portes et à la gestion des ressources ; l'ablation adaptive/frozen désactive simultanément le Q-learning en ligne et le calibrage énergétique, et ne permet pas d'attribuer leur différence au seul Q-learning.

La moyenne mesurée du temps de décision de V2 adaptative est d'environ **0,041 ms** en standard et **0,043 ms** en stress sur cette machine, pendant les évaluations exécutées en parallèle. C'est un coût observé localement, et non une garantie de temps sur la machine de compétition. Les 22 tests et le vérificateur de contrat passent.

Les CSV complets, comparaisons appariées et JSON sont dans `artifacts/v2/standard` et `artifacts/v2/stress`. Les empreintes des sources utilisées sont dans `artifacts/v2/source_sha256.txt`. Les anciennes évaluations et pilotes ne doivent pas être confondus avec ces résultats finaux.
