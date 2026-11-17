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

Les nouveaux acteurs et critiques ont chacun deux couches cachées de **256 neurones**, avec des activations `Tanh`. Cette taille est définie dans le code, sans option de commande. Avec 46 observations : acteur `46 → 256 → 256 → 7`, critique `46 → 256 → 256 → 1`.

Les anciens checkpoints et snapshots restent en **128 × 128** : leur taille est retrouvée dans les poids avant chargement. Les nouveaux checkpoints enregistrent aussi leur architecture. Un acteur apprenant en 256 peut donc jouer contre une snapshot en 128. Reprendre un ancien checkpoint ne l'agrandit pas ; pour le nouveau réseau, commencer **sans `--resume`**.

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

Une action peut donc sembler aléatoire même si la politique a appris quelque chose. Une baisse de l'écart-type signifie qu'elle devient plus concentrée, pas forcément meilleure.

L'**entropie** mesure la diversité des choix. Pour les commandes continues, elle augmente avec l'écart-type. Pour un bouton, elle est maximale à une probabilité de `0,5` et faible près de `0` ou `1`.

Le bonus optionnel `--entropy-coef 0.001` encourage à conserver cette diversité :

```text
loss acteur = loss PPO − coefficient × entropie moyenne
```

Le coefficient vaut **0 par défaut**, y compris pour les anciens checkpoints. Le bonus ne change ni les récompenses ni les cibles du critique. Il peut freiner la baisse de l'exploration, mais ne force pas les écarts-types à augmenter et ne garantit pas de meilleurs comportements. La limite KL reste active.

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
| KL | Mesure à quel point la politique a changé sur les observations du rollout. Si elle dépasse la cible, on arrête les mises à jour de l'acteur ; le critique continue. |

Une **époque** est un passage sur le rollout, découpé en mini-lots. Plusieurs époques réutilisent ces mêmes données, puis un nouveau rollout est collecté.

Les réglages par défaut sont : rollout `2048`, mini-lot `128`, maximum `4` époques, learning rates acteur/critique `0,0003`, `gamma = 0,99`, `gae_lambda = 0,95`, clipping `0,2` et cible KL `0,01`. Les commandes plus bas utilisent un rollout de `512`.

La KL est mesurée sur tout le rollout après chaque mise à jour de l'acteur. Dès que la cible est dépassée, **l'acteur reste figé jusqu'au prochain rollout**, y compris son optimiseur Adam. Le dernier pas reste appliqué, donc la cible peut être légèrement dépassée.

**Le critique termine les quatre époques prévues** (ou la valeur de `--num-epochs`). Ses poids sont séparés de ceux de l'acteur : il peut continuer à apprendre sans changer la politique. Les cibles GAE et les avantages sont calculés une seule fois avant l'optimisation, puis restent fixes pendant tout le cycle.

Exemple : si la KL dépasse la cible au deuxième mini-lot de la première époque, l'acteur fait deux mises à jour au total. Le critique traite quand même tous les mini-lots des quatre époques. Les logs affichent ensuite `acteur=figé`.

`ppo/epochs_done` compte les époques où l'acteur a travaillé (même partiellement), et `ppo/optimizer_steps` ses pas Adam. `ppo/critic_epochs_done` et `ppo/critic_optimizer_steps` donnent les compteurs du critique. Les losses acteur affichées restent celles de sa dernière époque active, pas des zéros ajoutés après son arrêt.

La **variance expliquée** mesure si le critique distingue les situations aux retours élevés de celles aux retours faibles : `1 - Var(cible - prédiction) / Var(cible)`. Proche de 1, il explique bien leurs différences ; proche de 0, il ne fait pas mieux qu'une prédiction constante sur ce critère ; négative, ses erreurs varient davantage que les cibles. Elle est mesurée sur tout le rollout avant/après l'optimisation, avec les mêmes cibles. Ce n'est pas une mesure de généralisation ni un score de football. Elle ignore un biais constant : regarder aussi la MSE. Si les cibles sont presque constantes, la variance expliquée n'est pas définie et le point n'est pas écrit (`*_defined = 0`).

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
- **Snapshot fixe** : apprenant `Foot` avec `Record Experience` coché ; adversaire `FootSnapshot` avec cette case décochée. Les deux utilisent `FootAgent`, sans contrôleur humain ni bot.

Le nom du run Python ne sélectionne pas la scène ni les apparitions aléatoires. Voir le guide Unity pour leur configuration.

Avec plusieurs terrains, conserver `Clone Scene Arena` coché sur `ArenaReplicator` pour reproduire ces réglages partout. `Random Spawns → Mode = Around Ball` tire le ballon dans la zone jouable puis les joueurs autour : bleu à gauche du rouge, distances plus courtes près du but défendu. Les intervalles et angles sont réglables dans Unity ([détails](README_UNITY.md#plusieurs-terrains-et-départs-autour-du-ballon)).

Le rollout est **global**. Avec 32 apprenants, un seuil de 512 représente environ 16 décisions par joueur, contre 64 avec 2048, hors dépassement de lot et fins d'épisode. Un rollout plus grand donne davantage de contexte au GAE et de situations par mise à jour. Une reprise garde le volume sauvegardé sauf si `--rollout-size` est fourni.

Diagnostic contre une snapshot (aucune optimisation ni sauvegarde) :

```powershell
python -m football_rl.ppo_unity --device cuda --resume checkpoints/snapshots/adversaire_v1.pt --opponent-checkpoint checkpoints/snapshots/adversaire_v1.pt --opponent-behavior FootSnapshot --collect-only --run-name diagnostic_snapshot_32 --verbose
```

Avec 32 terrains en 1 contre 1, attendre deux comportements et 64 agents dans `hello`, puis des batchs `Foot` seulement. Une fois le diagnostic validé, retirer `--collect-only` et choisir un nouveau nom de run pour l'entraînement avec ces nouveaux départs.

### Mélange : 16 bots + 16 snapshots du modèle 128 à 12 M

L'acteur 256 continue d'apprendre sur les 32 joueurs bleus. Les rouges sont répartis entre **16 bots Unity et 16 copies figées du modèle 128**, sauvegardé à 12 162 503 transitions. Seul le bleu enregistre des expériences ; le modèle adverse n'est jamais mis à jour.

Hors Play, conserver un joueur actif dans chaque équipe et un composant `FootAgent` sur chacun (celui du rouge peut être désactivé avant Play). Dans **IA Foot > Entraînement parallèle > Configurer le mélange 16 bots + 16 snapshots**, le menu active `ArenaOpponentMix`, règle 32 terrains, `Clone Scene Arena`, `Bot Arena Count = 16` et `Old Snapshot Arena Count = 16`. Il ne reste donc aucun terrain pour `FootSnapshotRecent`. Conserver les apparitions aléatoires et les récompenses actuelles, puis sauvegarder la scène sans appliquer ces réglages au prefab partagé.

```powershell
python -m football_rl.ppo_unity --device cuda --resume checkpoints/solo_256_vers_bot_random/latest.pt --opponent FootSnapshotOld checkpoints/snapshots/reference_128_12162503.pt --run-name mixed_256_bot_128_12m
```

Attendre **2 comportements et 48 agents PPO** : 32 bleus `Foot` et 16 rouges `FootSnapshotOld`. Les 16 bots sont contrôlés localement. Python installe la snapshot avant l'apprenant. Les poids et optimiseurs de l'apprenant sont repris ; les hyperparamètres restent inchangés. Pour un diagnostic sans apprentissage, ajouter `--collect-only --verbose`.

Pour reprendre ensuite ce nouveau run, utiliser `--resume checkpoints/mixed_256_bot_128_12m/latest.pt`, avec le même nom de run et la même option `--opponent`. La snapshot de référence reste fixe.

Suivre séparément `opponents/bot/sport/*` et `opponents/snapshot_old/sport/*`. Une répartition 16/16 des terrains ne garantit pas autant d'épisodes terminés contre chaque adversaire. Pour comparer les progrès au bilan précédent, conserver des évaluations séparées contre le bot, avec les mêmes apparitions aléatoires.

### Mélange d'adversaires : 16 bots + 8 anciennes snapshots + 8 récentes

Un seul acteur et un seul critique apprennent à partir des **32 joueurs bleus `Foot`**. Les rouges restent fixes : 16 `SimpleBotInput`, 8 `FootSnapshotOld` et 8 `FootSnapshotRecent`. Les adversaires n'enregistrent aucune transition. C'est un mélange fixe, pas une ligue avec remplacement automatique des adversaires.

Dans Unity, arrêter Play puis utiliser **IA Foot > Entraînement parallèle > Configurer le mélange 16 bots + 8 anciennes + 8 récentes**. La scène source doit avoir un joueur actif dans chaque équipe, chacun avec un `FootAgent` configuré. Sauvegarder la scène. Le menu active `ArenaOpponentMix`, impose 32 terrains et conserve les réglages de récompenses et d'apparitions. À chaque Play, les contrôleurs et comportements sont affectés automatiquement avant leur inscription au pont.

Une copie figée du modèle à **9 263 696 transitions** est disponible dans `checkpoints/snapshots/mix_recent_9263696.pt`. Elle sert à la fois de point de départ pour l'apprenant et d'adversaire récent ; seule la copie apprenante évolue. L'ancien adversaire reste `adversaire_v1.pt`. Ces fichiers locaux sont ignorés par Git.

```powershell
python -m football_rl.ppo_unity --device cuda --resume checkpoints/snapshots/mix_recent_9263696.pt --opponent FootSnapshotOld checkpoints/snapshots/adversaire_v1.pt --opponent FootSnapshotRecent checkpoints/snapshots/mix_recent_9263696.pt --run-name mixed_opponents_01
```

Pour vérifier le pont avant d'entraîner, ajouter `--collect-only --verbose`. Attendre **3 comportements, 48 agents** dans le `hello` (les 16 bots ne sont pas des `FootAgent` actifs), les deux confirmations de snapshots, puis celle de l'apprenant et une collecte valide. Retirer `--collect-only` pour entraîner. Les anciennes options `--opponent-checkpoint` et `--opponent-behavior` restent utilisables pour une seule snapshot ; elles ne se combinent pas avec `--opponent`.

À la reprise, remplacer uniquement le chemin de `--resume` par `checkpoints/mixed_opponents_01/latest.pt`, conserver le nom du run et **répéter les deux mêmes options `--opponent`**. Ne pas remplacer la snapshot récente par le dernier checkpoint de l'apprenant : cela changerait l'adversaire de l'expérience. Les hyperparamètres, dont l'entropie `0,001`, sont conservés.

Dans TensorBoard, consulter séparément :

- `opponents/bot/sport/*`
- `opponents/snapshot_old/sport/*`
- `opponents/snapshot_recent/sport/*`

Chaque groupe expose `count`, `goal_scored_fraction`, `goal_conceded_fraction` et `timeout_fraction`, ainsi que son retour et sa durée dans `episodes/*`. Les interruptions restent exclues du bilan sportif. Un lot sans épisode terminé pour un groupe n'ajoute pas de point pour ce groupe. Pondérer les taux par `sport/count` pour comparer des périodes : 50/25/25 % des terrains n'implique pas 50/25/25 % des épisodes, dont les durées diffèrent.

Pour revenir à une évaluation contre un seul adversaire, **désactiver `ArenaOpponentMix` hors Play**, puis configurer la scène comme indiqué plus haut. Les évaluations sur le mélange sont possibles, mais `--evaluation-episodes 500` compte alors 500 épisodes sportifs au total, pas 500 par adversaire. Garder une évaluation séparée contre une référence fixe.

### Démarrer de zéro

Depuis le dépôt Python, avec les dépendances et le pont installés :

```powershell
.\.venv\Scripts\Activate.ps1
python -m football_rl.ppo_unity --device cuda --rollout-size 512 --ball-approach-shaping 0.1 --run-name solo_random_pbrs_02 --verbose
```

Choisir un **nom encore inutilisé**, puis lancer **Play** dans Unity. Sans `--resume`, les réseaux repartent de zéro. Remplacer `cuda` par `cpu` si nécessaire.

Pour l'essai **256 × 256 depuis zéro**, commencer en solo avec apparitions aléatoires. Hors Play, désactiver `ArenaOpponentMix`, désactiver le GameObject adverse et garder l'apprenant `Foot` avec `Record Experience`. Sur les terrains dupliqués, garder `Clone Scene Arena` coché. Conserver les observations, récompenses et règles du jeu ; mettre `Distance To Ball Per Second` à `0` puisque le shaping Python est actif.

```powershell
python -m football_rl.ppo_unity --device cuda --actor-lr 0.0001 --critic-lr 0.0003 --entropy-coef 0.001 --ball-approach-shaping 0.1 --rollout-size 2048 --mini-batch-size 256 --run-name solo_256_random_scratch
```

Le LR acteur `0,0001` est un point de départ pour l'apprentissage, pas un réglage d'exploration. Le bonus d'entropie `0,001` encourage à conserver la diversité des actions ; les écarts-types initiaux restent à `0,5`. Le démarrage affiche `Architecture : acteur 256 x 256, critique 256 x 256`. Ce choix n'est pas une garantie de meilleure performance : comparer ensuite aux anciens modèles contre les mêmes adversaires, avec les mêmes conditions de départ.

### Reprendre le run existant

```powershell
python -m football_rl.ppo_unity --device cuda --resume checkpoints/solo_random_pbrs_01/latest.pt --run-name solo_random_pbrs_01 --verbose
```

Le même `--run-name` conserve le dossier TensorBoard et le dossier de sauvegarde. Le fichier est bien **`latest.pt`**.

La reprise restaure les réseaux, les optimiseurs, les compteurs et les réglages d'apprentissage, dont le shaping. Les options explicites `--actor-lr`, `--rollout-size`, `--mini-batch-size`, `--parity-atol` et `--entropy-coef` remplacent les valeurs sauvegardées. En leur absence, les valeurs du checkpoint sont conservées. Le LR acteur est réappliqué après le chargement d'Adam, qui conserve ses moments et son compteur. Les valeurs effectives sont affichées au démarrage. Le device et les options d'évaluation peuvent aussi changer.

Pour reprendre l'essai contre la snapshot avec 32 terrains, un LR acteur de `1e-4`, un rollout de `2048` et des mini-lots de `256` donnent un essai avec des mises à jour plus prudentes. Le nouveau nom conserve les courbes et checkpoints précédents pour comparaison.

```powershell
python -m football_rl.ppo_unity --device cuda --resume checkpoints/snapshot_v1_lr_1e4/latest.pt --actor-lr 0.0001 --rollout-size 2048 --mini-batch-size 256 --parity-atol 0.0005 --opponent-checkpoint checkpoints/snapshots/adversaire_v1.pt --opponent-behavior FootSnapshot --run-name snapshot_v1_lr_1e4_rollout2048
```

Python et Unity calculent maintenant les log-probabilités des boutons directement depuis les logits (`direct_logits`). L'ancien calcul `log(1 - p)` perdait de la précision lorsque `p` était presque égal à 1 : une rare action « ne pas sauter » pouvait arrêter l'évaluation ou l'entraînement. Le contrôle de parité reste actif ; augmenter sa tolérance ne corrige pas cette cause. Un dépassement sauvegarde le lot et les poids dans `parity_failure.pt`.

Cette correction nécessite aussi la version Unity du décodeur `direct_logits` dans `Policy.cs`. Arrêter Play et le serveur Python, laisser Unity compiler, puis relancer les deux. Les checkpoints existants restent utilisables, sans repartir de zéro. Un ancien pont refuse explicitement ce nouveau décodeur. Pour recommencer une évaluation interrompue, choisir un nouveau `--run-name` afin de séparer ses statistiques du résultat incomplet.

Un checkpoint est enregistré après chaque cycle réussi, avec une archive tous les dix cycles. La partie Unity et le rollout incomplet ne sont pas sauvegardés. Une déconnexion conserve les réseaux en mémoire tant que Python reste ouvert, mais abandonne la collecte incomplète.

Essai avec le critique qui termine ses époques : reprendre l'archive à **10 408 415 transitions**, proche de la référence évaluée à 10,4 M (le checkpoint exact évalué à 10 427 710 n'a pas été archivé). Le nouveau nom sépare cet essai des anciennes courbes ; les adversaires et réglages Unity restent identiques.

```powershell
python -m football_rl.ppo_unity --device cuda --resume checkpoints/mixed_reference_10m_lr5e5/cycle_007440.pt --actor-lr 0.00005 --opponent FootSnapshotOld checkpoints/snapshots/adversaire_v1.pt --opponent FootSnapshotRecent checkpoints/snapshots/mix_recent_9263696.pt --run-name mixed_critic_full_10m4
```

Les anciens checkpoints restent compatibles. Après environ 200 000 à 300 000 nouvelles transitions, évaluer ce nouvel essai contre les mêmes adversaires. Une amélioration de l'ajustement du critique ne garantit pas une amélioration sportive.

### Évaluer sans apprendre

Configurer l'adversaire voulu dans Unity, puis :

```powershell
python -m football_rl.ppo_unity --device cuda --resume checkpoints/solo_random_pbrs_01/latest.pt --evaluate --evaluation-episodes 200 --run-name eval_solo_pbrs_01
```

Les poids restent fixes, mais **les actions restent tirées au hasard dans les distributions apprises**. L'évaluation compte uniquement les épisodes sportifs : but marqué, encaissé ou timeout. Les interruptions et `match_over` sont exclus de cette cible.

Pour comparer deux checkpoints, utiliser les mêmes conditions : bot, règles d'apparition, durée maximale et récompenses. Donner un nom distinct à chaque évaluation. Les retours affichés sont ceux d'Unity, sans shaping Python.

En self-play partagé, chaque but produit une victoire et une défaite parmi les deux joueurs. Les résultats globaux peuvent donc rester équilibrés même si le niveau change. **L'évaluation contre un bot fixe sert de référence.**

### Comparer avec et sans bonus d'entropie

Une copie figée du run `snapshot_contact_005` est conservée dans `checkpoints/snapshots/entropy_reference_7899330.pt`, à **7 899 330 transitions**. Conserver les mêmes règles Unity, récompenses, apparitions et snapshot adverse pour les deux essais. Les deux commandes suivantes sont à lancer **séparément** :

```powershell
# Témoin : poursuivre sans bonus
python -m football_rl.ppo_unity --device cuda --resume checkpoints/snapshots/entropy_reference_7899330.pt --opponent-checkpoint checkpoints/snapshots/adversaire_v1.pt --opponent-behavior FootSnapshot --entropy-coef 0 --run-name entropy_control_7899330

# Essai : même point de départ, petit bonus
python -m football_rl.ppo_unity --device cuda --resume checkpoints/snapshots/entropy_reference_7899330.pt --opponent-checkpoint checkpoints/snapshots/adversaire_v1.pt --opponent-behavior FootSnapshot --entropy-coef 0.001 --run-name entropy_001_7899330
```

Faire environ **300 000 transitions supplémentaires par essai** : arrêter vers `8 199 330` transitions entraînées, à la fin d'un cycle. Ce seuil n'est pas un arrêt automatique. Pour reprendre un essai interrompu, utiliser **son propre** `checkpoints/<nom_du_run>/latest.pt` et le même nom de run.

Évaluer ensuite chaque `latest.pt` sur **500 épisodes sportifs**, avec `--evaluate --evaluation-episodes 500`, le même adversaire et un nom d'évaluation distinct. Comparer buts marqués, encaissés et timeouts ; une entropie plus élevée n'est pas à elle seule une réussite. Les tirages Unity restent aléatoires : les mêmes réglages ne garantissent pas des parties identiques.

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
| `sport/goal_scored_fraction`, `sport/goal_conceded_fraction`, `sport/timeout_fraction` | Proportions calculées uniquement sur les buts et timeouts. `sport/count` donne le nombre d'épisodes sportifs dans le lot. |
| `episodes/return_mean` | Récompense totale moyenne reçue d'Unity, **sans shaping Python**. Comparer à réglages de récompense identiques. |
| `approach/distance_mean_normalized` | Distance joueur–ballon : une baisse indique qu'il en est plus proche en moyenne. |
| `approach/episodes_with_kick_opportunity_fraction` | Part des épisodes sportifs complets suivis où une occasion de frappe a existé. Une occasion n'est pas un contact réel. |
| `reward/environment_mean`, `reward/shaping_mean`, `reward/training_mean` | Récompense Unity, ajout du shaping et somme utilisée pour apprendre, **par transition**. |
| `loss/critic` | Qualité d'ajustement des valeurs aux cibles du rollout. Une petite loss ne prouve pas que le joueur est bon. |
| `critic/explained_variance_before`, `critic/explained_variance_after` | Part de la variation des cibles expliquée avant/après les mises à jour, sur le même rollout. À lire avec `critic/mse_before` et `critic/mse_after`. |
| `ppo/approx_kl`, `ppo/epochs_done` | Ampleur du changement de politique et nombre d'époques commencées par l'acteur. La limite KL ne stoppe plus le critique. |
| `ppo/optimizer_steps`, `ppo/critic_optimizer_steps`, `ppo/critic_epochs_done` | Nombre réel de pas Adam acteur/critique et nombre d'époques terminées par le critique. |
| `ppo/clip_fraction` | Part des ratios hors de la zone de clipping. Une valeur élevée indique des changements importants, pas de meilleures actions. |
| `exploration/std_move_x`, `exploration/std_move_z` | Dispersion des actions de déplacement. Leur baisse ne suffit pas à démontrer une progression. |
| `exploration/entropy_continuous`, `exploration/entropy_binary`, `exploration/entropy_total` | Diversité des quatre commandes continues, des trois boutons et leur somme, après optimisation sur les observations du rollout. |
| `exploration/entropy_coef`, `loss/entropy_bonus` | Coefficient utilisé et terme soustrait à la loss acteur. `loss/actor_policy_loss` isole la loss PPO sans bonus. |

L'entropie est exprimée en **nats**. Celle des normales est calculée sur les **actions brutes avant bornage Unity** : elle peut être négative et ne mesure pas exactement la diversité des mouvements exécutés. L'entropie des trois boutons est comprise entre `0` et `3 × ln(2) ≈ 2,08`. La loss et le bonus affichés sont des moyennes pendant les mises à jour ; les entropies sont mesurées après celles-ci.

La loss acteur peut être négative : inutile de chercher à la faire tendre vers zéro. Les buts et les occasions de frappe donnent une mesure plus directe du progrès.

Pendant l'entraînement, les proportions `episodes/outcome_*` portent sur tous les résumés reçus. Les nouvelles séries `sport/*` excluent les interruptions et `match_over`. Elles restent des mesures par lot et peuvent fluctuer lorsque peu de parties finissent ensemble. `match_over` et `disabled` sont désormais écrits à zéro lorsqu'ils sont absents, pour éviter un lissage sur les seuls lots où ils apparaissent. En évaluation, l'axe horizontal compte les épisodes sportifs.

## 7. Où se trouve le code ?

Tous ces fichiers sont dans `src/football_rl/` :

| Fichier | Rôle |
| --- | --- |
| `ppo_network.py` | Acteur, critique, tirage des actions et log-probabilités |
| `ppo_export.py` | Export de l'acteur et du décodeur vers Unity |
| `ppo_opponent.py` | Snapshots figées et attente de tous leurs accusés avant l'apprenant |
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
