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
| Tests locaux d'intégration | Connexion simulée, vrais exports et mises à jour PyTorch |
| Validation dans une partie Unity réelle | À effectuer avec `--collect-only`, puis en entraînement |
| Checkpoints, suivi TensorBoard et évaluation | À écrire |

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
configurable ; elle n'a pas encore été validée sur une exécution C# réelle.

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
   à la première connexion. La dimension `D` provient d'Unity.
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
6. **Nouvelle publication** : exporter les poids et les `log_std` mis à jour,
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
n'annule pas les pas déjà effectués : il n'y a pas encore de checkpoint de
restauration. Après correction de la cause, relancer Python permet de repartir
d'un état neuf ; une simple reconnexion conserve les poids déjà en mémoire.

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
| `--verbose` | désactivé | Détail des messages et des lots retenus |

`--host`, `--port` et `--behavior` permettent également de modifier la connexion.
Une graine Python identique ne rend pas à elle seule les matchs Unity reproductibles.

## 11. Vérification locale

```powershell
python -m unittest discover -s tests -p "test_ppo*.py" -v
```

Les tests de `tests/test_ppo_trainer.py` utilisent une connexion simulée, les
vrais `iafoot.Batch`, l'évaluation NumPy des modèles exportés et un décodeur
de test indépendant de `evaluate_action`. Ils couvrent notamment :

- deux cycles complets de collecte, optimisation et publication ;
- modification des réseaux et des `log_std`, cohérence du nouvel export ;
- filtrage des versions mélangées et des messages en attente ;
- attente du bon ack, refus, changement de schéma et erreurs d'envoi/export ;
- reconnexion conservant les poids et les états Adam ;
- arrêt sur incohérence des log-probabilités, données ou calculs non finis ;
- mode diagnostic sans optimisation et branchement de la commande.

Ces tests valident le raccordement Python. Ils ne lancent pas le moteur Unity
et ne démontrent pas encore un apprentissage utile au football.

## 12. Suite du travail

1. Exécuter le diagnostic avec Unity, puis vérifier quelques cycles réels
   `collecting → training → waiting_for_ack → collecting`.
2. Ajouter les checkpoints : acteur, critique, optimiseurs et configuration.
   Pour l'instant, arrêter Python perd l'entraînement ; la reconnexion seule
   n'est pas une sauvegarde.
3. Ajouter le suivi TensorBoard. La dépendance est installée, mais aucun
   `SummaryWriter` n'est encore branché ; les métriques actuelles sont en console.
4. Construire une évaluation à politique figée contre des adversaires de
   référence, séparée des parties de collecte, pour mesurer les progrès.
5. Étudier ensuite l'exploration et un éventuel bonus d'entropie, puis ajuster
   les hyperparamètres à partir des mesures. Le critic loss ou le score d'une
   partie isolée ne suffisent pas pour juger la qualité de la politique.
