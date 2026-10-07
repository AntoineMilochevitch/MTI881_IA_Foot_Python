# Construire PPO pour le football Unity

Ce document décrit le code actuel et les étapes d'apprentissage à réaliser.
Le [README général](README.md) précise les observations/actions et le
[guide du pont](README_UNITY.md) explique l'installation et la collecte.

## 1. État de l'implémentation

| Composant | État |
| --- | --- |
| `PPOActorNetwork.__init__` et `forward` | Implémentés pour le schéma Unity |
| `PPOValueNetwork` | Implémenté |
| `export_actor` | Implémenté avec le modèle et le décodeur du pont |
| Envoi de l'acteur / réception des lots | Disponible dans `football_rl.ppo_unity` via le collecteur commun |
| `sample_action` / `evaluate_action` en Python | À écrire pour les sept commandes |
| Buffer, GAE, pertes et optimisations PPO | À écrire |
| Checkpoints, suivi TensorBoard et évaluation | À écrire |

Le serveur joue avec l'acteur initial et affiche les transitions. Les poids ne
sont pas encore entraînés. Les tests de l'ancienne interface ont été retirés.
Les contrôles des réseaux et distributions seront reconstruits autour du
contrat actuel, au fil des prochaines étapes.

### Lancer cet acteur

```powershell
python -m football_rl.ppo_unity --batch-size 256 --verbose
```

Cette commande fournit `ppo_unity.build_policy` au collecteur commun. La forme
équivalente, avec choix explicite de la fabrique, est :

```powershell
python -m football_rl.unity --policy-factory football_rl.ppo_unity:build_policy --verbose
```

Le lancement générique sans fabrique utilise une politique linéaire de démonstration.
Les fichiers propres à cet algorithme restent sur la branche `ppo` :
`ppo_network.py`, `ppo_export.py`, `ppo_unity.py` et ce document.

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

Les pertes et les optimiseurs mettront à jour les réseaux. Le critique reste
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
entre observations, mais appris pendant l'entraînement. L'optimiseur devra
recevoir `actor.parameters()` pour inclure ces paramètres externes au MLP.

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

## 6. Échantillonnage et log-probabilités à implémenter

Unity utilise déjà :

$$
a_i\sim\mathcal N(\mu_i(s),\sigma_i^2),\quad
\sigma_i=\exp(\operatorname{clamp}(\ell_i,-5,2)),\quad i\in\{0,1,3,4\},
$$

$$
a_j\sim\operatorname{Bernoulli}(\operatorname{sigmoid}(z_j(s))),
\quad j\in\{2,5,6\}.
$$

Contrats proposés pour Python :

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
son logarithme. La comparaison avec Python devra couvrir les logits extrêmes
et les différences d'arrondi entre les deux implémentations.

Les actions, anciennes log-probabilités et anciennes valeurs du buffer doivent
être détachées du graphe PyTorch. La nouvelle évaluation des log-probabilités
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

Le cycle d'apprentissage futur sera :

1. Publier la politique `v` et traiter son accusé de réception.
2. Accumuler suffisamment de transitions de cette version.
3. Calculer valeurs, avantages et cibles avec les paramètres de référence.
4. Effectuer plusieurs mises à jour PPO sur ce lot récent.
5. Publier `v+1` et préparer une nouvelle collecte.

Unity continue à simuler pendant les mises à jour Python. Les tags
`model_version` permettent de traiter les lots en attente et les changements
au milieu d'un épisode. Avant toute optimisation, réévaluer avec la politique
exportée doit donner `exp(log_prob_python - log_prob_unity)` proche de 1 pour
les mêmes observations et actions.

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

La perte acteur à minimiser sera `-L_clip`, avec éventuellement un bonus
d'entropie. La perte du critique sera une erreur quadratique entre sa prédiction
et la cible de retour. Les réseaux pourront avoir des optimiseurs séparés.

## 9. Buffer, avantages et fins de trajectoire

Le futur buffer utilisera les champs de `iafoot.Batch`, notamment actions
brutes, récompenses, observations suivantes, anciennes log-probabilités,
versions et identifiants temporels. Il faut reconstruire les segments avant de
mélanger les transitions pour l'optimisation.

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

La cible du critique sera `returns = advantages + old_values`, calculée avant
la normalisation éventuelle des avantages pour l'acteur. Ces cibles et les
anciennes log-probabilités restent fixes pendant les passages sur le lot.

## 10. Prochain exercice

Écrire `sample_action`, puis `evaluate_action`, selon la section 6. Les futurs
contrôles porteront sur `B=1` et `B=8`, la somme des sept log-probabilités, les
gradients des sorties et des quatre `log_std`, puis la cohérence Python/Unity.
Le buffer et les pertes viendront ensuite.
