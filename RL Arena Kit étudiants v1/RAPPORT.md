# Agent adaptatif de survie — version 3

## Objectif et compatibilité

L'objectif est la durée de survie sur une arène inconnue respectant le contrat du kit. L'agent utilise uniquement les perceptions publiques et les types de `rl_arena.arena_api`. Les références d'objets ne sont utilisées comme cibles d'interaction que pendant leur tick de validité. Aucun accès au moteur d'entraînement n'est nécessaire en compétition.

L'agent peut démarrer avec une table Q vide et apprendre pendant son premier match. Cela ne supprime pas les règles programmées de soin, de navigation et de défense : départ sans checkpoint ne veut pas dire absence de stratégie. Le syllabus communiqué n'impose pas explicitement une remise sans modèle ; le format doit être confirmé auprès du professeur.

## Améliorations de V3

- Les cibles de recherche tiennent compte des ressources déjà transportées et de la capacité de l'inventaire. L'agent cesse de poursuivre une catégorie dont il possède déjà assez d'exemplaires.
- Un médicament connu comme bandage est réservé aux saignements ; il n'est plus consommé pour soigner simplement une perte de santé, car cela n'avait pas cet effet dans le kit.
- Quand l'inventaire empêche de ramasser une ressource vitale proche, l'agent peut abandonner une munition ou un objet moins utile. Une exclusion temporaire empêche de reprendre immédiatement l'objet abandonné.
- La récupération d'énergie utilise une hystérésis : elle commence sous 0,20 et continue jusqu'à 0,75, sauf menace. Cela évite les répétitions de très courts repos avec un déplacement constamment limité par la fatigue.
- Le coût énergétique est estimé à partir de la vitesse réellement observée, plutôt que de la vitesse demandée que le moteur peut plafonner.
- La cible de Q-learning exclut les stratégies incompatibles avec les caractéristiques du prochain état discret. Les transitions terminales ne bootstrapent jamais.

Les anciennes règles de protection restent présentes : soins prioritaires, fuite face à une menace proche, évitement des obstacles, ouverture et exploration des portes, mémoire temporaire des ressources, défense pendant la fuite. Les souvenirs de position restent approximatifs et peuvent dériver pendant les collisions.

## Apprentissage pendant le match

L'observation suivante sert à évaluer l'action précédente. La récompense observable combine survie (+0,025 par tick), gains d'hydratation et de satiété, variation de santé et d'énergie, nouvelles cellules explorées, collisions et ramassages. Une élimination reçoit -12 en fin de match. Une mise à jour directe Q est suivie de quatre rejeux de transitions récentes ; le tampon est limité à 256 expériences. Les taux d'apprentissage sont 0,20 et 0,08, avec un discount de 0,97.

Les neuf caractéristiques discrètes résument santé, énergie, besoins alimentaires, menace, disponibilité de ressources utiles, toucher et dégâts dangereux. La présence de ressources est filtrée selon les besoins et l'inventaire ; le souvenir d'une menace conserve une alerte temporaire. Les actions apprises sont cinq stratégies : exploration, recherche de ressources, fuite, repos et exploration prudente. Leur sélection combine un prior de survie et une correction Q bornée. Les décisions vitales peuvent être imposées par les règles de protection.

Le modèle énergétique est calibré après huit déplacements valides, puis mis à jour progressivement. Cela correspond à quatre secondes simulées si ces huit déplacements se suivent ; les collisions ou le repos peuvent retarder ce calibrage. Ce délai ne garantit pas que l'agent apprenne toute la carte aussi rapidement.

`reset()` réinitialise les souvenirs de carte, l'hystérésis de repos, le tampon de rejeu et les estimations physiques ; il conserve la table Q sur la même instance. Aucun fichier n'est écrit automatiquement pendant un match. `agent.save(path)` peut conserver les connaissances entre processus si le règlement l'autorise.

## Protocole expérimental

Les essais de développement utilisent les graines standard 150000–150007 et stress 140000–140007. Plusieurs variantes ont été comparées : gestion des cibles seule, récupération longue, puis estimation des consommations avec priorités alimentaires modifiées. Cette dernière variante a régressé en développement et a été écartée ; son test unitaire de calibrage alimentaire a été retiré avec le mécanisme. Les tests de protection et d'apprentissage conservés restent exécutés.

La version retenue est entraînée à nouveau sur seize épisodes de 3000 ticks, graines 9000–9015. Le curriculum disperse les ressources, tourne et translate la géométrie, modifie la physiologie et la visibilité. Six graines de validation 50000–50005 sélectionnent le checkpoint. Les clones de validation s'adaptent pendant leur match, puis leur apprentissage est jeté : il ne retourne pas dans le candidat.

Les tests finaux utilisent douze graines standard 160000–160011 et douze graines stress 170000–170011. Ils démarrent **sans checkpoint pour V3**, pour couvrir la remise envisagée par l'étudiant. V2 est comparée à la fois sans modèle (`v2_cold`) et avec son ancien checkpoint (`v2`). Chaque politique rejoue les mêmes arènes avec les mêmes adversaires. La limite est de 3000 ticks, soit 1500 secondes simulées ; une survie atteignant cette limite est censurée. Les petits échantillons ne garantissent pas une performance similaire sur toute arène inconnue.

Le stress cumule une physiologie variable, des ressources et des positions initiales différentes, une géométrie transformée, une visibilité souvent dégradée et un chasseur supplémentaire. Les modifications du monde appartiennent uniquement au dispositif d'expérience ; elles ne sont jamais fournies à la politique comme information cachée.

## Reproduire

Depuis le dossier du kit, avec son venv activé :

```bash
python -m unittest discover -s tests -v
python check_student_agent.py
python train_agent.py --episodes 16 --ticks 3000 --seed 9000 --validation-episodes 6 --validate-every 8 --model artifacts/v3/final/policy.json
python evaluate_agent.py --episodes 12 --ticks 3000 --seed 160000 --policies cold_adaptive v2_cold v2 --output artifacts/v3/standard
python evaluate_agent.py --episodes 12 --ticks 3000 --seed 170000 --scenario stress --policies cold_adaptive v2_cold v2 --output artifacts/v3/stress
python build_submission.py --cold
python build_submission.py
```

Les données par épisode et temps de décision sont dans les CSV ; les moyennes sont dans les JSON. Les empreintes des sources sont dans `artifacts/v3/source_sha256.txt`. Le rapport et le checkpoint V2 sont archivés dans `artifacts/v2/snapshot`, avec son code dans `previous_agent_v2.py`.

## Remise au professeur

`artifacts/submission_cold.zip` contient une copie autonome de `student_agent.py` qui instancie `MyAgent()` avec `load=False` par défaut, sans fichier de modèle. Ce comportement a été testé même avec un checkpoint invalide placé à côté de l'agent : aucun checkpoint n'est lu et l'agent apprend durant le match. Les règles programmées restent présentes.

`artifacts/submission.zip` contient aussi le checkpoint V3 sélectionné, pour une remise autorisant l'entraînement préalable. Les anciennes tables V1/V2 ne sont pas réutilisées par V3, car les caractéristiques et les règles ont changé. Sans checkpoint compatible, l'agent utilise ses priors puis apprend en ligne.

## Limites d'interprétation

Une amélioration de la survie de V3 ne doit pas être attribuée automatiquement au seul Q-learning : les règles de choix de ressources, d'inventaire et de repos changent aussi. Les évaluations principales portent sur l'agent complet sans checkpoint. Le préentraînement et le calibrage énergétique nécessitent des ablations supplémentaires pour mesurer leur contribution propre.

La navigation reste locale, la mémoire est approximative, les seuils programmés peuvent être moins adaptés à une autre physiologie et l'exploration peut encore être fatale. Les connaissances acquises après une élimination ne peuvent aider le même match. Le contrat public, plutôt qu'une connaissance de la carte, assure la portabilité ; il ne garantit pas une victoire.

## Résultats finaux de V3

Sur 24 arènes inédites, V3 démarre sans checkpoint et conserve ses adaptations en ligne :

| Politique | Standard, 12 graines : survie moyenne en ticks | Stress, 12 graines : survie moyenne en ticks |
|---|---:|---:|
| V3 sans checkpoint | 2119,08 | 1539,50 |
| V2 sans checkpoint | 2224,00 | 1284,42 |
| V2 avec son checkpoint | 2224,00 | 1339,42 |

V3 améliore la survie moyenne en stress de **19,9 % face à V2 sans checkpoint**, et de **14,9 % face à V2 avec checkpoint**. En standard, V3 perd **4,7 %** face à V2 : le résultat est un compromis de robustesse, pas une amélioration uniforme.

En donnant le même poids aux deux familles de scénarios, la moyenne V3 est de 1829,29 ticks, contre 1781,71 pour V2 avec checkpoint, soit **+2,7 %**. Ce regroupement donne un poids arbitraire de 50 % à chaque scénario ; la distribution de la compétition réelle est inconnue.

V3 garde certains avantages de moyenne avec une dispersion importante : en stress, elle peut encore mourir à 743 ticks, tandis que V2 survit au moins 761 ticks sur cet échantillon. Le gain moyen ne garantit donc pas une meilleure survie dans chaque match. Plusieurs résultats à 3000 ticks correspondent à une élimination exactement à la limite, et non à une survie censurée ; le CSV distingue ces cas avec `reached_limit`.

Les temps de décision moyens observés sont environ 0,054 ms en standard et 0,058 ms en stress, sur cette machine pendant des évaluations parallèles. Ils ne garantissent pas la même latence en compétition. Les **28 tests** passent, y compris la remise sans checkpoint, l'apprentissage en ligne, le filtrage des cibles, la libération de place dans l'inventaire, les bandages et le repos prolongé.

Le checkpoint optionnel livré provient de la campagne finale de seize épisodes ; la validation sélectionnée vaut 2036,33 ticks en moyenne. Les résultats principaux ci-dessus évaluent la version sans checkpoint et ne démontrent pas un gain spécifique du préentraînement Q. Le modèle est optionnel pour la remise sans mémoire.

Les anciennes données V2 et les résultats intermédiaires restent archivés. Les seules mesures de test de la version retenue sont `artifacts/v3/standard` et `artifacts/v3/stress`. Les améliorations observées concernent l'agent complet ; elles ne prouvent pas que le Q-learning seul explique les gains.
