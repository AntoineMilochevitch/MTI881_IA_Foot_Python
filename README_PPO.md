# Construire PPO pour le football Unity

Ce document décrit l'implémentation PPO en PyTorch et son cycle avec Unity.
Le [README général](README.md) précise les observations/actions et le
[guide du pont](README_UNITY.md) explique l'installation et la collecte.

## 1. État de l'implémentation

| Composant | État |
| --- | --- |
| `PPOActorNetwork.__init__` et `forward` | Implémentés pour le schéma Unity |
| `PPOValueNetwork` | Implémenté |
| `export_actor` | Implémenté avec le modèle et le décodeur du pont |
| `sample_action` / `evaluate_action` en Python | Implémentés pour les sept commandes |
| Buffer et reconstruction des segments | Implémentés, stockage CPU, filtrage par version |
| Valeurs de référence, GAE et cibles du critique | Implémentés |
| PPO-Clip, optimiseurs séparés, mini-lots et arrêt KL | Implémentés |
| Collecte → entraînement → export → accusé de réception | Raccordés dans `PPOUnityTrainer` |
| Contrôle des log-probabilités Python/Unity | À chaque lot retenu, avant entraînement |
| Tests versionnés | Sauvegarde/restauration du critique et d'Adam, journalisation TensorBoard |
| Validation dans une partie Unity réelle | Collecte, entraînement CUDA et reprise après redémarrage vérifiés |
| Checkpoints | Sauvegarde après chaque cycle et archives tous les dix cycles |
| TensorBoard | Métriques d'apprentissage et résumés d'épisodes enregistrés |
| Adversaire `SimpleBotInput` | Configuration réseau contre bot utilisée dans Unity |
| Évaluation à politique figée | `--evaluate` avec reprise d'un checkpoint |

La boucle d'apprentissage est désormais exécutable. Unity produit les
observations, joue les actions, calcule les récompenses et transmet les
transitions. Python conserve l'acteur, le critique et leurs optimiseurs,
entraîne les réseaux puis renvoie l'acteur et son décodeur. Le moteur et le
protocole Unity ne sont pas modifiés.

### Installation et premier diagnostic

Depuis ce dépôt, avec `.venv` activé et les dépendances installées selon le
[guide du pont](README_UNITY.md), relier le paquet `iafoot` du dépôt voisin :

```powershell
python scripts/link_unity_bridge.py
python -m football_rl.ppo_unity --collect-only --batch-size 256 --rollout-size 2048 --verbose
```

Lancer ensuite **Play** dans la scène Unity préparée pour l'entraînement.
Le serveur écoute par défaut sur `127.0.0.1:5005`, pour le comportement `Foot`.
Dans Unity, activer `includeNextObservations` sur `TrainingBridge` et conserver
`recordExperience` activé sur les joueurs à observer.
Tous les agents inscrits à `Foot` utilisent la même politique.

Le mode `--collect-only` exporte une politique exploratoire, attend son
acceptation, contrôle les transitions et reconstruit les segments temporels.
À chaque seuil de collecte, il affiche le diagnostic puis vide le buffer,
sans modifier les poids et sans envoyer une nouvelle version. La mémoire ne
croît donc pas avec la durée de ce diagnostic.

À observer dans la console :

- `Politique v1 acceptée` : Unity a confirmé l'installation du modèle ;
- `écart logp max` : différence absolue maximale des log-probabilités sur les
  transitions retenues, inférieure ou égale à `--parity-atol` ;
- `ratio moyen` proche de 1, affiché par lot avec `--verbose` ;
- `Diagnostic : ... segments valides` : le seuil a été atteint et les
  trajectoires ont pu être reconstruites.

Une incohérence arrête la collecte PPO dans l'état `error` ; les messages
suivants ne déclenchent aucun apprentissage. Le serveur TCP reste ouvert.
Vérifier les observations, les actions **brutes**, l'ordre des commandes et
le décodeur avant de modifier la tolérance. Sa valeur initiale `1e-4` est
configurable ; les premiers essais C# ont montré des écarts de l'ordre de
`1e-6`, inférieurs à ce seuil. Le contrôle reste nécessaire à chaque collecte.

### Lancer l'entraînement

Arrêter le diagnostic avec `Ctrl+C`, relancer Python avec la commande suivante,
puis reconnecter Unity (arrêter et relancer Play si nécessaire) :

```powershell
python -m football_rl.ppo_unity --batch-size 256 --rollout-size 2048 --mini-batch-size 128 --num-epochs 4 --device cpu
```

Sans `--collect-only`, l'entraînement est activé. Le contrôle des
log-probabilités reste actif. Les valeurs ci-dessus constituent un point de
départ, pas des hyperparamètres validés pour gagner au football.

Avec 2048 transitions et des mini-lots de 128, une époque contient 16 mises à
jour de chaque réseau. Quatre époques en donnent au maximum 64, avant l'envoi
de la politique suivante. L'arrêt KL peut réduire le nombre d'époques.

Les lots Unity sont conservés en entier : si le seuil est dépassé, toutes les
transitions retenues sont utilisées. Il s'agit d'un nombre total de transitions
des agents de `Foot`, pas d'un nombre de parties ni d'un quota par joueur.

`--device cuda` demande un GPU CUDA disponible ; les calculs d'entraînement et
de comparaison s'y exécutent. Le buffer reste sur CPU et l'export se fait sur
une copie CPU. Le choix CPU par défaut permet le premier essai sans GPU.

Pour afficher toutes les options :

```powershell
python -m football_rl.ppo_unity --help
```

### Sauvegarder et reprendre

La commande crée automatiquement `runs/<run_name>` pour TensorBoard et
`checkpoints/<run_name>` pour les sauvegardes. Sans `--run-name`, le nom est
horodaté. `--checkpoint-dir` remplace le dossier de sauvegarde complet.

```powershell
# Nouvel entraînement : à utiliser pour une nouvelle expérience.
python -m football_rl.ppo_unity --device cuda --rollout-size 512 --run-name ppo_bot_reference_01

# Reprise : après avoir arrêté la précédente exécution.
python -m football_rl.ppo_unity --device cuda --resume checkpoints/ppo_bot_reference_01/latest.pt --run-name ppo_bot_reference_01_reprise

# Evaluation
python -m football_rl.ppo_unity --device cuda --resume checkpoints/solo_positions_variees/latest.pt --evaluate --evaluation-episodes 200 --run-name eval_bot_solo_01
```

Pour prolonger l'essai existant, le point de départ disponible est par exemple
`checkpoints/checkpoint_test_reprise/latest.pt`. Utiliser `--resume` : un
lancement sans cette option initialise une nouvelle politique.

- `latest.pt` est remplacé après chaque cycle réussi, avant l'envoi du modèle suivant.
- Une archive `cycle_000010.pt`, `cycle_000020.pt`, etc. est conservée tous les dix cycles.
- L'écriture passe par un fichier temporaire dans le même dossier, puis le remplace.
- Aucune sauvegarde automatique n'est créée avant le premier cycle terminé.
- `--collect-only` ne déclenche ni optimisation, ni sauvegarde automatique, ni TensorBoard.

Le checkpoint contient l'acteur complet, dont `log_std`, le critique, les deux
Adam, les quatre compteurs, le schéma Unity, la configuration et l'état aléatoire
PyTorch CPU. Il ne sauvegarde pas le buffer, les sockets, la partie Unity, les
états aléatoires Unity/CUDA ou les résumés d'épisodes en attente.

À la reprise, les réglages sauvegardés sont prioritaires. Seuls `device`, `log_dir`, `collect_only`, `evaluate` et
`evaluation_episodes` sont remplacés dans la configuration par les options
du nouveau lancement. Les options de transport et de dossier restent celles
du lancement. Par exemple, `--rollout-size 2048` ne remplace pas une valeur 512
enregistrée dans le checkpoint. La console annonce cette règle.

Les réseaux sont restaurés après validation du schéma, avant leur publication.
Le placement CPU/GPU est choisi avec `--device`, qui vaut CPU par défaut.
Unity doit acquitter le modèle avant la création d'un nouveau buffer.
Une reconnexion ultérieure conserve les progrès en mémoire et ne recharge pas
le fichier. Le compteur de versions peut différer du nombre de cycles :
il compte les publications, dont celles provoquées par les reconnexions.

Une interruption au milieu d'un cycle repart du dernier checkpoint réussi ;
il n'y a pas de sauvegarde des pas d'optimisation partiels à la fermeture.
Conserver les mêmes réglages de scène : le schéma ne vérifie pas les récompenses
ni les valeurs `positionScale` et `velocityScale`, absentes du `hello`.

### Évaluer un checkpoint

L'évaluation exige un checkpoint et n'entraîne pas les réseaux. Elle retient
uniquement les épisodes dont toutes les transitions proviennent de la politique
évaluée et arrête le comptage après 200 épisodes sportifs par défaut
(`goal_scored`, `goal_conceded` ou `timeout`) :

```powershell
python -m football_rl.ppo_unity --resume checkpoints/ppo_bot_reference_01/latest.pt --evaluate
```

Modifier la cible avec `--evaluation-episodes`. Les épisodes exclus pour
interruption technique sont signalés séparément et ne sont pas comptés comme
des défaites. Les autres issues (`interrupted`, `match_over`, `disabled`, etc.)
ne font pas avancer la cible. La console distingue les épisodes sportifs des
épisodes complets reçus ; l'axe TensorBoard correspond au nombre d'épisodes
sportifs, et les statistiques sont cumulées sur ces épisodes.
Pendant l'évaluation, la console affiche aussi, pour chaque
agent et chaque batch, les moyennes prédites de `move_x`/`move_z`, leurs bornes,
l'écart-type de la politique et les proportions d'actions Unity dépassant
`-1` ou `1`. Elle compare également la norme de la commande horizontale après
le bornage Unity à la vitesse horizontale observée dans `next_obs` (convertie
avec `velocityScale = 6`), ainsi que les proportions de transitions
`self_stunned`, `self_tackling` et `self_grounded`. Les probabilités prédites
de tacle et de saut sont comparées aux commandes binaires effectivement tirées.
Les transitions où le joueur est libre de marcher donnent aussi la commande et
la vitesse moyennes ; l'alignement entre la commande et la direction du ballon
est affiché séparément, en excluant les vecteurs presque nuls. Les issues
sportives inconnues sont exclues du bilan principal mais comptées à part.

Pour isoler le déplacement des effets du tacle et du saut, lancer l'évaluation
avec `--diagnostic-no-jump-tackle`. Cette option n'est disponible qu'avec
`--evaluate` : elle annule les poids des sorties tacle et saut dans
l'acteur en mémoire et fixe leurs biais à `-20` avant l'export. Leur probabilité
d'activation devient ainsi proche de zéro. Le checkpoint
original n'est pas modifié, et cette variante n'est ni entraînée ni sauvegardée.

### Ouvrir TensorBoard

Les métriques sont activées par défaut pendant l'entraînement. Depuis la racine
du dépôt, ouvrir un deuxième terminal, activer le même `.venv`, puis lancer :

```powershell
python -m tensorboard.main --logdir runs --port 6006
```

Ouvrir [TensorBoard](http://localhost:6006). Ce serveur lit les fichiers
d'événements ; il ne commande pas l'entraînement. Il peut rester ouvert pendant
que Unity joue et afficher les fichiers après l'arrêt de Python.

1. Dans **Runs**, sélectionner l'expérience en cours, par exemple
   `checkpoint_test_reprise`. Une reprise est une continuation, pas un essai indépendant.
2. Dans **Scalars** ou **Time Series**, afficher les groupes `episodes`, `ppo`,
   `loss` et `exploration`.
3. Choisir **Step** comme axe horizontal : pendant l'entraînement, c'est le total
   de transitions utilisées pour l'apprentissage. Pendant l'évaluation, c'est le
   nombre d'épisodes sportifs retenus.
4. Commencer avec **Smoothing = 0**, puis essayer un lissage modéré, par exemple
   0,5, pour voir une tendance. Revenir aux points bruts pour contrôler les pics.
5. L'onglet **Text** contient la configuration enregistrée au lancement.

Un point est écrit après un cycle terminé ; les métriques d'optimisation
proviennent de sa dernière époque effectuée. Les courbes d'épisodes ne reçoivent
pas de point si aucun résumé d'épisode n'a été accumulé. Une collecte encore
en cours peut donc laisser les courbes immobiles.

Les réglages disponibles sont `--log-dir`, `--run-name` et `--no-tensorboard`.
Utiliser un nouveau nom de run à la reprise, tout en conservant le compteur de
transitions restauré. Les nouveaux points commencent après ce compteur.

Si aucune courbe n'apparaît :

- vérifier le dossier affiché au lancement, le répertoire courant et la sélection des runs ;
- attendre la fin d'un cycle et rafraîchir l'interface ;
- vérifier que `--no-tensorboard` et `--collect-only` ne sont pas actifs ;
- inspecter les fichiers avec `python -m tensorboard.main --inspect --logdir runs`.

Le message `TensorFlow installation not found - running with reduced feature set`
n'empêche pas l'affichage de ces métriques PyTorch. Si le port est occupé, choisir
par exemple `--port 6007` et ouvrir le même port dans le navigateur.

### Lire les courbes de jeu en premier

| Courbe | Ce qu'elle mesure | Comment la lire |
| --- | --- | --- |
| `episodes/outcome_goal_scored` | Fraction des résumés terminés par un but de l'équipe de l'agent | Une hausse durable est encourageante, mais un but adverse contre son camp peut aussi compter |
| `episodes/outcome_goal_conceded` | Fraction terminée par un but encaissé | Chercher une baisse en vérifiant les autres types de fin |
| `episodes/outcome_timeout` | Fraction terminée par la limite de pas | Une hausse peut signifier mieux défendre ou simplement rester bloqué |
| `episodes/return_mean` | Moyenne des sommes de récompenses par épisode | Contient les buts, contacts et récompenses intermédiaires ; ce n'est pas le score du match |
| `episodes/return_std` | Écart-type de ces retours | Décrit leur dispersion, pas un intervalle de confiance |
| `episodes/length_mean` | Nombre moyen de décisions par épisode | Plus long n'est pas toujours mieux : lire avec les buts et limites de temps |
| `episodes/count` | Nombre de résumés utilisés pour ce point | Ce n'est pas un cumul ; une proportion basée sur deux épisodes est fragile |
| `episodes/outcome_interrupted` et autres `outcome_*` | Fractions des autres types de fin, dont `disabled` ou `match_over` lorsqu'ils apparaissent | Les distinguer des résultats de jeu |

Les agrégats portent sur les résumés accumulés depuis le précédent enregistrement,
pas sur une fenêtre glissante fixe ni sur tous les épisodes depuis le lancement.
Par exemple, 0,25 pour `outcome_goal_scored` signifie 25 % des résumés de ce point.
Un épisode correspond à une manche jusqu'au but ou à une coupure, pas à un match
complet. Le lissage de l'interface ne pondère pas les points par `episodes/count`.

Ces statistiques décrivent la collecte, pendant laquelle la politique évolue.
Un épisode peut traverser plusieurs versions et ses résumés sont traités à part
du filtrage des transitions. Avec plusieurs agents, ils sont regroupés :
les résultats de deux équipes opposées peuvent se compenser. Pour le duel
contre `SimpleBotInput`, seul le joueur réseau doit avoir `FootAgent` actif.

Le suivi actuel n'enregistre pas le nombre de contacts avec le ballon,
les tirs utiles, la possession ou les résultats par équipe. Une récompense
négative reste un signal d'apprentissage ; le problème observé au démarrage est
surtout la rareté des contacts et des expériences positives. Les vérifier aussi
en regardant quelques manches.

### Lire les courbes techniques ensuite

| Courbe | Interprétation |
| --- | --- |
| `loss/actor` | Objectif PPO du lot ; peut être négatif, n'a pas à tendre vers zéro et ne mesure pas le niveau au football |
| `loss/critic` | Erreur quadratique sur les retours cibles ; une valeur faible peut simplement refléter des défaites très prévisibles |
| `ppo/approx_kl` | Changement de politique par rapport à la collecte, mesuré après l'époque ; à rapprocher du seuil enregistré |
| `ppo/clip_fraction` | Fraction des ratios hors de `[1 - epsilon, 1 + epsilon]` ; avec epsilon 0,2, une valeur 0,20 signifie 20 % hors de `[0,8 ; 1,2]` |
| `ppo/ratio_mean` | Ratio moyen après mise à jour ; proche de 1 ne garantit pas que chaque action a peu changé |
| `ppo/epochs_done` | Nombre d'époques réellement effectuées, au maximum 4 par défaut |
| `ppo/kl_limit_reached` | 1 si le seuil KL a arrêté les époques suivantes ; les mises à jour déjà effectuées sont conservées |
| `grad/actor_norm`, `grad/critic_norm` | Moyennes des normes des gradients avant leur limitation ; elles peuvent dépasser `max_grad_norm` sans que la limitation soit défaillante |
| `exploration/std_move_x`, `std_move_z`, `std_shoot_power`, `std_shoot_curve` (préfixe `exploration/`) | Écarts-types des quatre actions continues brutes, initialement 0,5 ; ce n'est pas une probabilité de 50 % ni une mesure du niveau de jeu |
| `parity/max_log_prob_error` | Écart maximal Python/Unity avant optimisation ; doit rester sous `parity_atol`, par défaut `1e-4` |
| `data/rollout_transitions` | Taille effective du rollout, qui peut dépasser le seuil demandé |
| `data/ignored_transitions_total` | Total cumulé de lignes ignorées ; les changements de version en expliquent une partie |
| `time/collection_s`, `time/optimization_s` | Durées réelles par cycle pour repérer où passe le temps ; pas des mesures de qualité du jeu |

Le suivi d'exploration concerne les actions continues. Les probabilités et
entropies des boutons ne sont pas encore journalisées. Une exploration qui baisse
rapidement avec des résultats toujours mauvais mérite un diagnostic, mais il
n'existe pas de valeur cible universelle. Un arrêt KL après une ou deux époques
ne justifie pas à lui seul de changer les réglages.

### Collecteur générique

Cette commande reste une **collecte de démonstration sans apprentissage** :

```powershell
python -m football_rl.unity --policy-factory football_rl.ppo_unity:build_policy --verbose
```

Elle utilise `build_policy` pour exporter l'acteur initial. Elle n'est plus
équivalente à `python -m football_rl.ppo_unity`, qui instancie le trainer PPO.
Le collecteur commun `football_rl.unity` reste indépendant de l'algorithme.

### Organisation du code

| Fichier dans `src/football_rl/` | Responsabilité |
| --- | --- |
| `ppo_network.py` | Acteur, critique, distributions et log-probabilités |
| `ppo_export.py` | Copie cohérente du MLP et des `log_std` pour Unity |
| `ppo_buffer.py` | Copie CPU des transitions, filtrage de version, segments |
| `ppo_return.py` | Valeurs avant mise à jour, GAE et cibles du critique |
| `ppo_training.py` | Pertes, normalisation des avantages, mini-lots, optimisation, métriques KL |
| `ppo_trainer.py` | Configuration, callbacks, état de la connexion et cycle d'apprentissage |
| `ppo_unity.py` | Arguments de lancement et branchement des callbacks de `iafoot.TrainingServer` |
| `ppo_checkpoint.py` | Construction, écriture, validation et restauration des checkpoints |
| `ppo_logging.py` | Configuration et métriques envoyées à TensorBoard |

Les fichiers d'algorithme et ce document appartiennent au travail PPO ;
`unity.py` et le paquet `iafoot` restent utilisables par les autres algorithmes.

## 2. Acteur, politique et critique

L'acteur produit les paramètres de la distribution des actions. La politique
complète comprend ce réseau et son décodage : quatre normales et trois
Bernoulli indépendantes conditionnellement à l'observation.

Le critique estime le retour actualisé attendu sous la politique suivie :

$$
V^\pi(s_t)=\mathbb E_\pi\left[\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}\mid s_t\right].
$$

Ici `s` représente le vecteur d'observation. La valeur peut être négative ou
supérieure à un. Elle fournit une référence pour estimer si une action a donné
un résultat meilleur ou moins bon que prévu.

Les pertes et les optimiseurs mettent à jour les réseaux. Le critique reste
dans Python ; l'acteur et son décodeur sont exécutés dans Unity.

## 3. Observations et tenseurs

Les réseaux prennent des tenseurs `float32` de forme `(B, D)` :

- `B` : nombre d'observations, y compris `B=1` pour une situation ;
- `D` : `obs_size` reçu dans `hello`, avec l'ordre donné par `obs_names`.

La scène examinée utilise 46 valeurs. La formule Unity est
`28 + 6 * (max_teammates + max_opponents)`. Unity fournit les observations dans
le repère d'équipe et les met à l'échelle. L'ancien prototype Gymnasium a été
supprimé ; la dimension d'entrée vient directement du pont.

Une normalisation supplémentaire éventuelle devrait être identique dans
Python et dans l'export ; elle n'est pas appliquée actuellement.

## 4. Acteur actuel

Le MLP complet est stocké dans `actor.network` :

```text
(B, D) → Linear(D, 128) → Tanh → Linear(128, 128) → Tanh → Linear(128, 7)
```

La dernière couche est linéaire. Ses colonnes suivent cet ordre :

| Indice | Commande | Paramètre produit |
| ---: | --- | --- |
| 0 | `move_x` | Moyenne gaussienne |
| 1 | `move_z` | Moyenne gaussienne |
| 2 | `shoot` | Logit de Bernoulli |
| 3 | `shoot_power` | Moyenne gaussienne |
| 4 | `shoot_curve` | Moyenne gaussienne |
| 5 | `tackle` | Logit de Bernoulli |
| 6 | `jump` | Logit de Bernoulli |

`actor.log_std` est un `nn.Parameter` de forme `(4,)`, indépendant de
l'observation, initialisé à `log(0.5)`. Il contient les log-écarts-types des
colonnes continues `[0, 1, 3, 4]`. Le code les borne dans `[-5, 2]` avant usage.

`forward` calcule les paramètres, sans tirer d'action :

| Clé | Forme | Contenu |
| --- | --- | --- |
| `model_output` | `(B, 7)` | Sortie brute du MLP dans l'ordre Unity |
| `continuous_mean` | `(B, 4)` | Moyennes des commandes continues |
| `continuous_log_std` | `(B, 4)` | Paramètres bornés et étendus sur le lot |
| `binary_logits` | `(B, 3)` | Logits de `shoot`, `tackle`, `jump` |

Moyennes et logits dépendent de l'observation. Les écarts-types sont partagés
entre observations, mais appris pendant l'entraînement. L'optimiseur reçoit
`actor.parameters()` pour inclure ces paramètres externes au MLP.

### Exemple de lecture

Pour une sortie `[0.8, -0.2, 1.1, -0.6, 0, -2, -3]`, les moyennes continues
sont `[0.8, -0.2, -0.6, 0]` et les logits binaires `[1.1, -2, -3]`.
La probabilité de frappe vaut `sigmoid(1.1)`, environ 75 %. La puissance brute
est centrée sur −0,6 ; si cette valeur est tirée et la frappe déclenchée, Unity
utilise une intensité de 0,2 via `clip((a + 1) / 2, 0, 1)`.

## 5. Critique actuel

Le critique possède ses propres poids :

```text
(B, D) → Linear(D, 128) → Tanh → Linear(128, 128) → Tanh → Linear(128, 1)
```

`squeeze(-1)` transforme `(B, 1)` en `(B,)` sans supprimer l'axe du lot quand
`B=1`. La valeur est accessible sous la clé `value`.

## 6. Échantillonnage et log-probabilités

Unity utilise déjà :

$$
a_i\sim\mathcal N(\mu_i(s),\sigma_i^2),\quad
\sigma_i=\exp(\operatorname{clamp}(\ell_i,-5,2)),\quad i\in\{0,1,3,4\},
$$

$$
a_j\sim\operatorname{Bernoulli}(\operatorname{sigmoid}(z_j(s))),
\quad j\in\{2,5,6\}.
$$

Contrats implémentés en Python :

- `sample_action(observations)` retourne `actions` de forme `(B, 7)` et
  `log_prob` de forme `(B,)` après un seul passage dans le MLP ;
- `evaluate_action(observations, actions)` réévalue les actions enregistrées
  sous les paramètres actuels, sans nouveau tirage.

La log-probabilité totale additionne les sept contributions :

$$
\log\pi(a\mid s)=\sum_{i\in\{0,1,3,4\}}\log\mathcal N(a_i;\mu_i(s),\sigma_i^2)
+\sum_{j\in\{2,5,6\}}\log\operatorname{Bernoulli}(a_j;p_j(s)).
$$

La somme porte sur les commandes et conserve une valeur par observation.
Les commandes continues restent brutes : le moteur borne les valeurs lors de
leur application physique. Aucune transformation `tanh` de l'action ni
correction de Jacobien n'est utilisée. Les activations `Tanh` internes au MLP
restent en place.

Puissance et effet contribuent à la log-probabilité même lorsque `shoot=0`.
Unity les échantillonne dans tous les cas. Une commande ignorée par les règles
du jeu reste également présente dans l'action enregistrée.

Unity applique un plancher de `1e-8` à la probabilité binaire sélectionnée avant
son logarithme. Python applique la même convention. Le trainer compare chaque
log-probabilité reçue à son recalcul avec l'acteur resté fixe pendant la
collecte. Cette vérification mesure notamment les différences d'arrondi ;
les tests NumPy locaux ne remplacent pas le contrôle de l'exécution C#.

Les actions et anciennes log-probabilités du buffer, ainsi que les valeurs de
référence calculées par le critique, sont détachées du graphe PyTorch. La
nouvelle évaluation des log-probabilités pendant l'optimisation
doit rester différentiable par rapport à l'acteur.

## 7. Export et cycle de collecte

`export_actor` copie l'acteur, prépare sa copie CPU `float32` et exporte le MLP
complet avec `iafoot.models.from_torch`. Le contrôle `check=True` joint une
entrée et la sortie PyTorch attendue pour que Unity vérifie son calcul.

Les log-écarts-types bornés de la même copie sont envoyés au décodeur :

```text
[log_std_move_x, log_std_move_z, 0, log_std_power, log_std_curve, 0, 0]
```

Les positions binaires sont ignorées. `deterministic=False` conserve
l'exploration. `ModelExport` et `DecoderExport` sont envoyés ensemble par
`UnityConnection.send_model`, avec une version commune.

Le cycle implémenté dans `PPOUnityTrainer` est :

1. **Connexion** : lire `hello`, vérifier le schéma, créer les réseaux et Adam
   à la première connexion. La dimension `D` provient d'Unity. En mode reprise,
   restaurer le checkpoint après validation du schéma et avant publication.
2. **Publication** : réserver une version supérieure à la version locale et
   à celle annoncée par Unity, envoyer acteur + décodeur et passer à
   `waiting_for_ack`. Aucun buffer n'est collecté dans cet état.
3. **Acceptation** : seul un `model_ack` positif de la connexion, du
   comportement et de la version attendus active un nouveau buffer vide.
4. **Collecte** : garder les lignes portant cette version, même si un message
   mélange plusieurs versions ; vérifier les formes, les valeurs finies, les
   boutons binaires et les log-probabilités. L'acteur reste fixe.
5. **Apprentissage** : au seuil `rollout_size`, appeler `train_ppo`. Reconstruire
   les segments, calculer une fois les valeurs de référence, GAE et retours,
   normaliser les avantages puis entraîner sur plusieurs époques.
6. **Suivi et sauvegarde** : incrémenter les compteurs, écrire les métriques
   TensorBoard et sauvegarder le cycle terminé.
7. **Nouvelle publication** : exporter les poids et les `log_std` mis à jour,
   abandonner le buffer utilisé et attendre l'acceptation de la version suivante.

Unity continue à simuler pendant les mises à jour Python. Les tags
`model_version` permettent de traiter les lots en attente et les changements
au milieu d'un épisode. Avant toute optimisation, réévaluer avec la politique
exportée doit donner `exp(log_prob_python - log_prob_unity)` proche de 1 pour
les mêmes observations et actions. Ce contrôle s'effectue avec `no_grad` et
ne remplace jamais `old_log_probs` par les valeurs recalculées : la référence
PPO reste celle réellement enregistrée par Unity.

Les callbacks du serveur sont sérialisés. Pour cette première intégration,
`train_ppo` s'exécute dans `on_batch`. Les messages en attente sont traités
ensuite : les anciens lots sont ignorés pendant l'attente d'ack ou filtrés
par version après celui-ci. Il n'y a pas de mélange entre collectes ni de
replay des anciennes versions. La perte de ces données réduit le rendement
de la simulation ; la durée des mises à jour et le compteur des transitions
ignorées sont affichés pour pouvoir la mesurer.

Une déconnexion abandonne la collecte incomplète, mais conserve les réseaux,
les états Adam et le compteur de version **dans le même processus Python**.
La reconnexion avec le même schéma republie les poids courants. Un changement
d'ordre des observations, d'actions ou de période de décision est refusé.
Un `hello` répété pour la même session n'effectue pas de nouvelle initialisation.
Le trainer accepte une seule connexion Unity active et un comportement.

Un refus du modèle, une incohérence de données ou une erreur d'entraînement
vide le buffer et place le trainer en `error`. Aucune nouvelle mise à jour
n'est tentée sur ces données. Une erreur après plusieurs pas d'optimisation
n'annule pas les pas déjà effectués. Après correction de la cause, relancer
Python avec `--resume` sur le dernier checkpoint réussi permet de repartir de
ce cycle terminé ; une simple reconnexion conserve les poids déjà en mémoire.
Il n'y a pas de restauration automatique après une erreur.

## 8. PPO-Clip : principe

PPO est model-free : il n'apprend pas le moteur physique. C'est une méthode
on-policy de gradient de politique, utilisée ici avec un critique. Elle
réutilise un lot récent pendant plusieurs passages, puis collecte de nouvelles
données. Le ratio entre politiques n'autorise pas un replay arbitraire de
vieilles expériences. [PPO-Clip, OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

Pour une transition, comparer la politique actuelle à celle qui a joué :

$$
\rho_t=\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)).
$$

L'objectif de l'acteur est :

$$
L^{\mathrm{clip}}=\mathbb E_t\left[\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].
$$

Un avantage positif favorise l'action ; un avantage négatif pousse à réduire
sa probabilité. Le clipping limite l'incitation à changer cette probabilité
dans la direction favorisée. Il ne borne pas directement les poids et ne
garantit pas une amélioration.

La perte acteur minimisée est `-L_clip`, sans bonus d'entropie actuellement.
La perte du critique est l'erreur quadratique moyenne entre sa prédiction
et la cible de retour. Chaque réseau possède son optimiseur Adam ; le clipping
des gradients s'applique séparément aux deux réseaux. Les pertes et normes
de gradient non finies provoquent un arrêt avant les pas des optimiseurs du
mini-lot concerné.

Après chaque époque, l'acteur courant réévalue tout le rollout sans gradient.
Pour `d = log_prob_new - log_prob_old`, les métriques sont :

- `approx_kl = mean(exp(d) - 1 - d)`, calculé avec `expm1` ;
- `clip_fraction` : fraction de ratios sortant de `[1-epsilon, 1+epsilon]` ;
- `ratio_mean` : moyenne des ratios.

Si `approx_kl > target_kl`, les époques restantes sont annulées. L'époque
déjà effectuée est conservée. C'est un contrôle pratique du changement de
politique, pas une garantie mathématique de rester sous ce seuil.

## 9. Buffer, avantages et fins de trajectoire

Le buffer utilise les champs de `iafoot.Batch`, notamment actions
brutes, récompenses, observations suivantes, anciennes log-probabilités,
versions et identifiants temporels. Les tableaux sont copiés en tenseurs CPU,
sans graphe de gradient. Le buffer appartient à une session, un comportement
et une version. Il regroupe les lignes par `(agent_id, episode_id)`, trie par
`step`, refuse les pas dupliqués et coupe après les fins d'épisode ou les trous.
Les trajectoires de joueurs différents ne sont jamais raccordées.

Les segments sont reconstruits avant le mélange pour l'optimisation. Les
avantages et retours sont calculés dans l'ordre d'origine, puis une permutation
commune mélange observations, actions et cibles à chaque époque.

GAE utilise les valeurs du critique avant mise à jour :

$$
\delta_t=r_t+\gamma b_tV_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t),
\qquad
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}.
$$

La récurrence se calcule à rebours. `b_t` autorise le bootstrap de valeur ;
`c_t` autorise la propagation depuis la transition suivante du même segment.
La combinaison des erreurs temporelles règle le compromis entre biais et
variance. [Publication GAE](https://arxiv.org/abs/1506.02438)

| Situation dans notre tâche | `b_t` | `c_t` |
| --- | ---: | ---: |
| Transition ordinaire avec la suite disponible | 1 | 1 |
| But : `terminated=True` | 0 | 0 |
| Coupure technique : `truncated=True` | 1 | 0 |
| Fin de segment non terminal, trou de pas ou frontière de version | 1 | 0 |

Le bootstrap d'une troncature utilise `next_obs` avant reset. La récurrence ne
traverse pas le reset. Cette distinction suit le traitement des
[terminaisons et limites de temps](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
Dans notre tâche, un but termine l'épisode ; une fin de match signalée comme
troncature par le pont conserve cette convention.

La cible du critique est `returns = advantages + old_values`, calculée avant
la normalisation des avantages pour l'acteur. Cette normalisation est effectuée
une fois sur tout le rollout avec l'écart-type de population et `1e-8` au
dénominateur. Les retours du critique ne sont pas normalisés. Ces cibles et les
anciennes log-probabilités restent fixes pendant les passages sur le lot.

Les données préparées pour les optimiseurs sont transférées sur le périphérique
des réseaux. Le stockage initial CPU évite de garder les messages et le buffer
de collecte dans la mémoire GPU.

## 10. Paramètres de lancement

| Option | Défaut | Rôle |
| --- | ---: | --- |
| `--batch-size` | 256 | Nombre de transitions par message Unity |
| `--rollout-size` | 2048 | Seuil de transitions valides d'une version avant entraînement, minimum 2 |
| `--mini-batch-size` | 128 | Nombre de transitions par pas d'optimisation ; dernier mini-lot conservé |
| `--num-epochs` | 4 | Nombre maximal de passages sur un rollout |
| `--actor-lr`, `--critic-lr` | 0.0003 | Taux d'apprentissage des deux Adam |
| `--gamma` | 0.99 | Actualisation des récompenses par décision enregistrée |
| `--gae-lambda` | 0.95 | Pondération de la récurrence GAE |
| `--clip-epsilon` | 0.2 | Clipping du ratio PPO |
| `--max-grad-norm` | 0.5 | Norme maximale des gradients de chaque réseau |
| `--target-kl` | 0.01 | Arrêt des époques suivantes si le seuil est dépassé |
| `--no-target-kl` | désactivé | Désactive cet arrêt anticipé |
| `--parity-atol` | 0.0001 | Écart absolu maximal de log-probabilité accepté par transition |
| `--time-scale` | 1 | Vitesse Unity, entre 0.1 et 50 |
| `--device` | `cpu` | Périphérique PyTorch : `cpu` ou `cuda[:index]` |
| `--seed` | 0 | Initialisation et RNG PyTorch, sans piloter le RNG Unity |
| `--collect-only` | désactivé | Diagnostic sans mise à jour des réseaux |
| `--evaluate` | désactivé | Évalue un checkpoint sans entraînement |
| `--evaluation-episodes` | 200 | Nombre d'épisodes sportifs à évaluer (but marqué, encaissé ou timeout) |
| `--diagnostic-no-jump-tackle` | désactivé | Désactive tacle et saut pendant une évaluation |
| `--resume` | aucun | Checkpoint à restaurer au démarrage |
| `--checkpoint-dir` | `checkpoints/<run_name>` | Dossier des sauvegardes |
| `--log-dir` | `runs` | Racine des événements TensorBoard ; le nom du run est ajouté |
| `--run-name` | `ppo_AAAAMMJJ-HHMMSS` | Nom de cette exécution |
| `--no-tensorboard` | désactivé | Coupe la journalisation TensorBoard, sans couper les checkpoints |
| `--verbose` | désactivé | Détail des messages et des lots retenus |

`--host`, `--port` et `--behavior` permettent également de modifier la connexion.
Une graine Python identique ne rend pas à elle seule les matchs Unity reproductibles.

## 11. Vérification locale

```powershell
python -m unittest discover -s tests -p "test_ppo*.py" -v
```

Les fichiers présents dans `tests/` couvrent :

- `test_ppo_checkpoint.py` : aller-retour après une mise à jour du critique,
  avec vérification d'un paramètre et de son historique Adam ;
- `test_ppo_logging.py` : écriture et relecture des événements TensorBoard,
  agrégation des épisodes et comportement du suivi.

Des vérifications ponctuelles ont également couvert un cycle simulé,
la reprise avec une mise à jour identique sur CPU à données et RNG identiques,
la reconnexion, les archives, le refus de schéma et le passage CPU/CUDA.
Ces scénarios restent à intégrer durablement à la suite de tests.
La reprise a aussi été essayée dans Unity. Ces vérifications techniques
ne démontrent pas une amélioration du niveau de jeu.

## 12. Suite du travail

La priorité actuelle est de prolonger l'expérience contre le même bot, en
conservant PPO, les hyperparamètres, les récompenses, la scène et les réglages du
bot. Le rollout de l'essai repris reste à 512. La continuation doit utiliser
`--resume` après un arrêt de Python ; si le processus tourne déjà, le laisser continuer.

Se donner des points de contrôle vers **10 000, 50 000 et 100 000 transitions
entraînées au total**, en sautant un palier déjà dépassé. Ce sont des repères
pratiques pour une première expérience, pas une garantie de progression ni un
budget suffisant pour maîtriser le jeu. Il n'y a pas d'arrêt automatique à ces
valeurs dans la commande actuelle.

À chaque palier, relever les courbes d'épisodes sur plusieurs cycles, leurs
effectifs, les statistiques PPO et quelques séquences de jeu. Garder le chemin
du checkpoint correspondant. Observer les contacts avec le ballon en vidéo :
ce compteur n'existe pas encore dans TensorBoard.

Une hausse durable des buts marqués, une baisse des buts encaissés sans
remplacement systématique par des timeouts, et des actions visiblement plus utiles
justifient de prolonger. Une perte critique faible ou une manche gagnée isolée
ne suffisent pas. Une erreur de parité, des valeurs non finies ou une collecte
bloquée demandent un diagnostic immédiat, pas davantage de temps d'entraînement.

Après cette phase de référence :

1. Construire une évaluation à politique figée, avec le même bot et des côtés alternés.
   Les courbes de collecte servent au suivi, pas à une comparaison contrôlée.
2. Si les contacts et résultats restent rares sans tendance favorable, étudier
   un apprentissage progressif : adversaire immobile, variante plus lente,
   puis bot normal. Garder le bot de référence inchangé pour l'évaluation.
3. Étudier ensuite les améliorations RL de [TODO_PPO.md](TODO_PPO.md), une idée
   à la fois : exploration, critique, représentation, tâches auxiliaires, etc.
