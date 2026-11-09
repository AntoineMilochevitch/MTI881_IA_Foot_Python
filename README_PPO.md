# PPO pour le football Unity

Ce guide explique le modèle, les commandes utiles et les courbes à surveiller.
Pour l'installation et le détail du pont, voir le [README général](README.md) et le [guide Unity](README_UNITY.md). Les pistes de travail sont dans [TODO_PPO.md](TODO_PPO.md).

## 1. Qui fait quoi ?

**Unity fait jouer le match. Python entraîne les réseaux.**

- L'**acteur** reçoit les observations et choisit les actions.
- Le **critique** estime les récompenses futures actualisées : combien le joueur peut espérer obtenir à partir de cette situation. Ce n'est pas une probabilité de gagner.
- Unity utilise une copie de l'acteur. Le critique reste dans Python pour aider à apprendre.

La boucle se répète ainsi :

1. Python envoie l'acteur à Unity.
2. Unity joue et renvoie les transitions : observation, action, récompense, observation suivante et informations de fin d'épisode.
3. Python rassemble ces transitions dans un **rollout**, puis entraîne l'acteur et le critique.
4. Python sauvegarde et envoie la nouvelle version de l'acteur.

PPO est **model-free** : il n'apprend pas à simuler la physique. Il est **on-policy** : chaque mise à jour utilise les actions de la politique qui vient de jouer. C'est un **actor-critic** : il associe une politique et une estimation de valeur.

## 2. Ce que voit et fait le modèle

Le schéma actuel contient **46 observations** : informations sur le joueur, le ballon, les autres joueurs et le match. Unity les prépare et les normalise. L'ordre et le détail des observations sont décrits dans le README général ; Python lit le schéma envoyé par le pont.

L'acteur et le critique ont chacun deux couches cachées de **128 neurones**, avec des activations `Tanh`.

L'acteur produit les paramètres de sept commandes :

| Commande | Rôle |
| --- | --- |
| `move_x`, `move_z` | Déplacement horizontal |
| `shoot` | Déclencher une frappe |
| `shoot_power` | Régler sa puissance, pour une passe ou un tir |
| `shoot_curve` | Régler sa courbe |
| `tackle`, `jump` | Demander un tacle ou un saut |

### Comment fonctionne l'exploration ?

Pour les quatre commandes continues, l'acteur choisit une **moyenne**, puis tire une valeur dans une loi normale autour de cette moyenne.

- Petit écart-type : les actions restent proches de la moyenne.
- Grand écart-type : les actions sont plus dispersées.
- Les quatre écarts-types commencent à **0,5**, puis sont appris. Ils sont communs aux observations, alors que les moyennes dépendent de la situation.

Pour `shoot`, `tackle` et `jump`, il apprend une probabilité d'activation, puis tire 0 ou 1.

Une action peut donc sembler aléatoire même si la politique a appris quelque chose. Une baisse de l'écart-type signifie qu'elle devient plus concentrée, pas forcément meilleure. Il n'y a actuellement **pas de bonus d'entropie** dans la loss.

Unity borne les commandes avant de les appliquer. PPO conserve les **actions brutes tirées** pour calculer leurs probabilités : il ne faut pas les remplacer par les commandes bornées dans le buffer.

## 3. Comment PPO apprend

Le **buffer** stocke les transitions sur CPU. Python reconstruit les trajectoires séparément pour chaque joueur et chaque épisode, sans relier deux épisodes entre eux. Avec `--device cuda`, les calculs d'entraînement utilisent le GPU ; l'export vers Unity reste sur CPU.

### Avantage et critique

L'**avantage** répond à la question : « Cette action a-t-elle donné mieux ou moins bien que ce qu'on attendait ? »

- Avantage positif : on cherche à rendre cette action plus probable dans cette situation.
- Avantage négatif : on cherche à la rendre moins probable.

Le **GAE** estime ces avantages à partir des récompenses et des valeurs du critique. `gamma` pondère les récompenses futures ; `gae_lambda` règle le compromis entre des estimations courtes et longues.

Lors d'une vraie fin, comme un but, il n'y a plus de valeur future à ajouter. Lors d'une coupure temporelle, on utilise la valeur de la dernière observation pour estimer la suite, sans continuer dans l'épisode suivant.

### Loss, clipping et KL

| Élément | Explication simple |
| --- | --- |
| Loss acteur | Encourage les actions dont l'avantage est positif et décourage celles dont il est négatif. Sa valeur n'est pas un score de football. |
| Loss critique | Mesure l'écart entre la valeur prédite et la cible construite à partir des récompenses et du GAE. |
| Ratio PPO | Probabilité de l'action avec le nouvel acteur, divisée par sa probabilité avec l'ancien. À 1, elle n'a pas changé. |
| Clipping | Limite l'intérêt de modifier trop fortement ce ratio. Avec `0,2`, la zone de référence est `[0,8 ; 1,2]`. Ce n'est pas une limite stricte sur les poids. |
| KL | Mesure à quel point la politique a changé sur les observations du rollout. Si elle dépasse la cible, on arrête les époques restantes. |

Une **époque** est un passage sur le rollout, découpé en mini-lots. Plusieurs époques réutilisent ces mêmes données, puis un nouveau rollout est collecté.

Les réglages par défaut sont : rollout `2048`, mini-lot `128`, maximum `4` époques, learning rates acteur/critique `0,0003`, `gamma = 0,99`, `gae_lambda = 0,95`, clipping `0,2` et cible KL `0,01`. Les commandes plus bas utilisent un rollout de `512`.

La KL est vérifiée après chaque époque : un arrêt conserve les mises à jour déjà effectuées. Augmenter le nombre d'époques n'aide donc pas si cet arrêt intervient déjà tôt.

## 4. Récompenses et shaping : la différence

### Les récompenses d'Unity

Unity attribue les récompenses du jeu : but marqué ou encaissé, contact avec le ballon, progression du ballon vers le but adverse, temps écoulé et éventuellement distance au ballon. Leurs coefficients se règlent sur `FootAgent`.

Le réglage `Distance To Ball Per Second` applique une **pénalité liée à la distance actuelle**. Plus le joueur reste loin, plus il est pénalisé au fil du temps. Cette pénalité existe même s'il ne bouge pas.

### Le potentiel

Dans notre shaping Python, le potentiel est simplement :

```text
potentiel = −distance horizontale normalisée entre le joueur et le ballon
```

Par exemple, à une distance de `0,5`, le potentiel vaut `−0,5`. À une distance de `0,2`, il vaut `−0,2` : **être plus proche donne un potentiel plus élevé**.

Il s'agit de la distance **joueur–ballon**, pas de la distance ballon–but. La hauteur est ignorée pour que sauter ne modifie pas cette distance. Les valeurs sont normalisées par Unity, pas exprimées directement en mètres.

### Le shaping

Le potentiel décrit une situation. Le shaping compare la situation **avant et après une action** :

```text
shaping = coefficient × (gamma × potentiel_après − potentiel_avant)
récompense utilisée pour apprendre = récompense Unity + shaping
```

Avec `gamma = 1`, cela devient exactement :

```text
shaping = coefficient × (distance_avant − distance_après)
```

Donc : rapprochement → bonus ; éloignement → pénalité ; distance inchangée → zéro.

Avec les valeurs utilisées ici, `gamma = 0,99` et coefficient `0,1` :

| Distance avant → après | Shaping ajouté |
| --- | ---: |
| `0,5 → 0,4` : rapprochement | `+0,0104` |
| `0,5 → 0,5` : inchangée | `+0,0005` |
| `0,5 → 0,6` : éloignement | `−0,0094` |

**La différence avec une simple pénalité de distance :** arriver à `0,4` depuis `0,5` donne ici un bonus ; arriver au même endroit depuis `0,3` donne une pénalité. La distance finale seule ne permet pas de distinguer les deux.

On garde le même `gamma` que PPO pour respecter son actualisation des récompenses. Cela explique le petit terme positif même à distance inchangée : le shaping n'est pas exactement une mesure du rapprochement lorsque `gamma < 1`.

Cette construction fait se compenser les potentiels intermédiaires dans la somme actualisée. Avec les bonnes conditions de fin, elle peut guider l'apprentissage sans changer quelles politiques sont optimales pour la récompense de départ. Elle ne garantit pas que PPO apprendra plus vite.

À une vraie fin d'épisode, le potentiel suivant est mis à **zéro**. À un timeout ou une fin de rollout, on garde celui de la dernière observation, avant le repositionnement. Ces cas sont déjà gérés dans `ppo_shaping.py`.

### Quel réglage utiliser ?

- Le shaping est désactivé par défaut. `--ball-approach-shaping 0.1` l'active.
- Il est ajouté dans Python **avant le GAE**, sans modifier la loss PPO ni les récompenses envoyées par Unity.
- Le run `solo_random_pbrs_01` a été lancé avec **shaping `0,1` et pénalité Unity de distance `0,005`**. Les deux effets s'additionnent ; conserver ces réglages pour poursuivre la même expérience.
- Pour tester le shaping seul, mettre la pénalité Unity de distance à `0`, puis lancer une nouvelle expérience avec un nouveau nom. Python ne change pas automatiquement ce réglage Unity.

Le shaping mesure une variation de distance, quelle qu'en soit la cause : le ballon peut aussi se rapprocher du joueur. Les buts et les occasions de frappe restent nécessaires pour juger si le comportement devient utile.

## 5. Lancer, reprendre et évaluer

### Préparer Unity

Activer `includeNextObservations` sur `TrainingBridge`, et `recordExperience` sur le `FootAgent` qui apprend. Son comportement doit être `Foot`. Désactiver les autres contrôleurs d'entrée de ce joueur.

Choisir la scène et les adversaires **dans Unity** :

- **Solo aléatoire** : activer les apparitions aléatoires du `MatchManager` et désactiver le GameObject de l'adversaire.
- **Contre le bot** : adversaire avec `SimpleBotInput` actif et `FootAgent` désactivé.
- **Self-play** : deux joueurs d'équipes opposées, chacun avec `FootAgent` et le même comportement `Foot`. Ils partagent le même acteur.

Le nom du run Python ne sélectionne pas la scène ni les apparitions aléatoires. Voir le guide Unity pour leur configuration.

### Démarrer de zéro

Depuis le dépôt Python, avec les dépendances et le pont installés :

```powershell
.\.venv\Scripts\Activate.ps1
python -m football_rl.ppo_unity --device cuda --rollout-size 512 --ball-approach-shaping 0.1 --run-name solo_random_pbrs_02 --verbose
```

Choisir un **nom encore inutilisé**, puis lancer **Play** dans Unity. Sans `--resume`, les réseaux repartent de zéro. Remplacer `cuda` par `cpu` si nécessaire.

### Reprendre le run existant

```powershell
python -m football_rl.ppo_unity --device cuda --resume checkpoints/solo_random_pbrs_01/latest.pt --run-name solo_random_pbrs_01 --verbose
```

Le même `--run-name` conserve le dossier TensorBoard et le dossier de sauvegarde. Le fichier est bien **`latest.pt`**.

La reprise restaure les réseaux, les optimiseurs, les compteurs et les réglages d'apprentissage, dont le shaping. Ajouter un autre learning rate ou rollout dans cette commande ne remplace pas les valeurs sauvegardées. Le device et les options d'évaluation peuvent changer.

Un checkpoint est enregistré après chaque cycle réussi, avec une archive tous les dix cycles. La partie Unity et le rollout incomplet ne sont pas sauvegardés. Une déconnexion conserve les réseaux en mémoire tant que Python reste ouvert, mais abandonne la collecte incomplète.

### Évaluer sans apprendre

Configurer l'adversaire voulu dans Unity, puis :

```powershell
python -m football_rl.ppo_unity --device cuda --resume checkpoints/solo_random_pbrs_01/latest.pt --evaluate --evaluation-episodes 200 --run-name eval_solo_pbrs_01
```

Les poids restent fixes, mais **les actions restent tirées au hasard dans les distributions apprises**. L'évaluation compte uniquement les épisodes sportifs : but marqué, encaissé ou timeout. Les interruptions et `match_over` sont exclus de cette cible.

Pour comparer deux checkpoints, utiliser les mêmes conditions : bot, règles d'apparition, durée maximale et récompenses. Donner un nom distinct à chaque évaluation. Les retours affichés sont ceux d'Unity, sans shaping Python.

En self-play partagé, chaque but produit une victoire et une défaite parmi les deux joueurs. Les résultats globaux peuvent donc rester équilibrés même si le niveau change. **L'évaluation contre un bot fixe sert de référence.**

### Diagnostiquer le pont

```powershell
python -m football_rl.ppo_unity --collect-only --rollout-size 512 --verbose
```

Ce mode collecte sans optimiser et sans TensorBoard. Il vérifie notamment que Python et Unity calculent les mêmes log-probabilités. L'écart doit rester sous la tolérance par défaut `1e-4`.

Le rollout est un seuil : les lots Unity sont conservés entiers, donc une collecte peut dépasser `512` transitions.

## 6. Lire TensorBoard

```powershell
python -m tensorboard.main --logdir runs --port 6006
```

Ouvrir <http://localhost:6006>. Observer les tendances sur plusieurs milliers de transitions, pas un seul point.

| Mesure | Ce qu'elle permet de comprendre |
| --- | --- |
| `episodes/outcome_goal_scored`, `outcome_goal_conceded`, `outcome_timeout` | Résultats sportifs : surtout utiles face à un adversaire fixe. Vérifier aussi le nombre d'épisodes. |
| `episodes/return_mean` | Récompense totale moyenne reçue d'Unity, **sans shaping Python**. Comparer à réglages de récompense identiques. |
| `approach/distance_mean_normalized` | Distance joueur–ballon : une baisse indique qu'il en est plus proche en moyenne. |
| `approach/episodes_with_kick_opportunity_fraction` | Part des épisodes sportifs complets suivis où une occasion de frappe a existé. Une occasion n'est pas un contact réel. |
| `reward/environment_mean`, `reward/shaping_mean`, `reward/training_mean` | Récompense Unity, ajout du shaping et somme utilisée pour apprendre, **par transition**. |
| `loss/critic` | Qualité d'ajustement des valeurs aux cibles du rollout. Une petite loss ne prouve pas que le joueur est bon. |
| `ppo/approx_kl`, `ppo/epochs_done` | Ampleur du changement de politique et nombre d'époques réellement effectuées. Des arrêts KL fréquents invitent à examiner le learning rate. |
| `ppo/clip_fraction` | Part des ratios hors de la zone de clipping. Une valeur élevée indique des changements importants, pas de meilleures actions. |
| `exploration/std_move_x`, `exploration/std_move_z` | Dispersion des actions de déplacement. Leur baisse ne suffit pas à démontrer une progression. |

La loss acteur peut être négative : inutile de chercher à la faire tendre vers zéro. Les buts et les occasions de frappe donnent une mesure plus directe du progrès.

Pendant l'entraînement, les proportions d'issues TensorBoard portent sur les résumés reçus, qui peuvent inclure des interruptions. En évaluation, les statistiques sportives utilisent uniquement les épisodes sportifs et leur nombre comme axe horizontal.

## 7. Où se trouve le code ?

Tous ces fichiers sont dans `src/football_rl/` :

| Fichier | Rôle |
| --- | --- |
| `ppo_network.py` | Acteur, critique, tirage des actions et log-probabilités |
| `ppo_export.py` | Export de l'acteur et du décodeur vers Unity |
| `ppo_buffer.py` | Stockage et reconstruction des trajectoires |
| `ppo_return.py` | Avantages GAE et cibles du critique |
| `ppo_training.py` | Losses, mises à jour et arrêt KL |
| `ppo_trainer.py` | Organisation du cycle collecte → entraînement → export |
| `ppo_unity.py` | Commandes de lancement et serveur |
| `ppo_checkpoint.py` | Sauvegarde et reprise |
| `ppo_shaping.py` | Potentiel joueur–ballon et suivi des occasions de frappe |
| `ppo_logging.py` | Enregistrement TensorBoard |

Pour retrouver toutes les options sans allonger ce guide :

```powershell
python -m football_rl.ppo_unity --help
```
