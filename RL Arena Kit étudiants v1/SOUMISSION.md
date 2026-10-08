# Remettre l'agent au professeur

## Sans mémoire préentraînée

Envoyer `artifacts/submission_cold.zip`. Extraire `student_agent.py` dans un dossier dédié au groupe, à côté du kit final. Instancier `MyAgent()` : la table Q est vide et aucun checkpoint n'est chargé par défaut. L'agent apprend à partir des observations suivantes. Les règles de survie programmées restent présentes.

Le code de l'archive est autonome : aucun module de ce dépôt n'est requis en dehors de `rl_arena.arena_api`, fourni par le kit. Respecter les appels `reset()`, `act(observation)`, `on_episode_end(result)`.

## Avec un modèle préentraîné si le professeur l'autorise

Envoyer `artifacts/submission.zip`. Conserver la structure `student_agent.py` et `artifacts/policy.json`. Ce modèle est de version 3. L'agent continue à apprendre pendant le match.

Le règlement de remise n'est pas précisé dans le syllabus reçu : confirmer si les checkpoints et la conservation des connaissances entre plusieurs matchs sont autorisés. Aucun fichier n'est écrit automatiquement en compétition.

Les deux archives incluent le rapport et les résultats des essais finaux. Le dispositif d'entraînement et les bots de comparaison restent dans le dépôt de travail.
