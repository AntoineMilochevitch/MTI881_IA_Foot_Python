# TODO PPO — idées pour la suite

PPO tourne, les checkpoints permettent de reprendre et TensorBoard enregistre
les courbes. La priorité est maintenant de laisser une vraie chance à
l'entraînement actuel contre le bot fixe, puis de faire un bilan.
Les idées ci-dessous restent des pistes à choisir, pas toutes à faire.

Les tâches ne sont pas attribuées. On pourra choisir les sujets et se répartir
le travail plus tard. Certaines pistes améliorent les outils autour de PPO,
d'autres touchent directement au modèle et à sa façon d'apprendre.

Les détails de l'algorithme sont dans [README_PPO.md](README_PPO.md).
L'installation et les réglages Unity sont dans [README_UNITY.md](README_UNITY.md).

## 1. Ce qui tourne déjà

- [x] Acteur et critique implémentés en PyTorch.
- [x] Buffer, segments de trajectoire, GAE, pertes PPO et optimisation implémentés.
- [x] Cycle complet connecté : collecte → entraînement → export → acceptation Unity.
- [x] Diagnostic réel réussi : 748 transitions contrôlées et 2 segments valides.
- [x] Entraînement réel lancé sur CUDA.
- [x] Cohérence des log-probabilités conservée après plusieurs mises à jour.
- [x] Reconnexion Unity testée en conservant les poids et Adam dans Python.
- [x] Sauvegarde sur disque et reprise après redémarrage de Python.
- [x] Suivi TensorBoard des métriques d'apprentissage et des épisodes.
- [x] Configuration d'un joueur réseau contre `SimpleBotInput` dans Unity.
- [ ] Évaluation du niveau de jeu contre un adversaire fixe.

Repère au moment de la mise à jour du 2 novembre 2026, pour
`checkpoint_test_reprise` (ce tableau n'est pas un compteur en direct) :

| Indicateur | Valeur |
| --- | --- |
| Cycles PPO terminés dans le checkpoint lu | 14 |
| Transitions utilisées pour l'entraînement | 10 669 |
| Seuil de collecte de cette expérience | 512 |
| Appareil | CUDA |
| Trois derniers points TensorBoard | 100 % `goal_conceded`, 0 % `goal_scored` |
| Retours moyens de ces points | Environ -1,03 |

La version compte les publications : une reconnexion republie les poids avec
une nouvelle version, sans ajouter un cycle d'apprentissage. Les essais anciens
et les nouveaux lancements sans reprise ne s'additionnent pas comme s'il
s'agissait d'un seul modèle.

## 2. Ce que nous pouvons conclure

**La boucle technique fonctionne. Une amélioration du niveau de jeu n'a pas
encore été mesurée.** Le modèle peut déjà progresser, mais les pertes et la KL
ne suffisent pas à le démontrer.

Python envoie la politique au comportement `Foot`. Tous les `FootAgent`
actifs avec ce nom utilisent les mêmes poids, mais chacun reçoit sa propre
observation et choisit sa propre action. Les trajectoires restent séparées
par agent dans le buffer.

Si ces agents jouent dans des équipes opposées, le modèle joue contre
lui-même. Des scores équilibrés peuvent persister même si les deux côtés
progressent. S'ils sont réellement tous les deux bleus, ce sont des
coéquipiers : il faut vérifier la configuration des équipes avant de parler
d'un duel rouge contre bleu.

### Lire les métriques actuelles

- `arrêt_KL=True` : la KL approximative dépasse `target_kl=0.01`. Le cycle
  s'arrête après l'époque effectuée ; ses mises à jour sont conservées.
- `num_epochs=4` est un maximum, pas une obligation d'effectuer quatre époques.
- Le contrôle KL intervient après chaque époque entière : le seuil peut donc
  être dépassé avant que les époques suivantes soient annulées.
- `clip` mesure la fraction des ratios hors de `[0.8, 1.2]` avec les réglages
  actuels. Il ne mesure pas la réussite au football.
- `ratio` est une moyenne : des variations individuelles peuvent se compenser.
- Une perte acteur négative est possible et n'indique pas à elle seule un problème.
- Une perte critique faible indique un bon ajustement aux cibles du lot ; son
  interprétation dépend aussi de l'échelle et de la variabilité de ces cibles.

Pour l'instant, garder les hyperparamètres actuels comme point de départ.
Éviter de les modifier simplement pour obtenir quatre époques à chaque cycle.

### Priorité : poursuivre la référence avant de changer quelque chose

- [ ] Continuer le processus actuel s'il tourne déjà ; utiliser `--resume`
  après un arrêt de Python pour conserver ce qui a été appris.
- [ ] Garder les mêmes réglages PPO, le rollout enregistré de 512, la scène,
  les récompenses et les paramètres du bot.
- [ ] Ouvrir TensorBoard dans un second terminal et suivre les transitions
  entraînées sur l'axe **Step**.
- [ ] Faire des points vers 10 000, 50 000 et 100 000 transitions au total ;
  passer au palier suivant lorsqu'un seuil est déjà dépassé.
- [ ] À chaque point, noter le checkpoint, les buts marqués/encaissés, les
  timeouts, retours, longueurs d'épisode et le nombre d'épisodes observés.
- [ ] Regarder quelques manches pour voir si le joueur rejoint et touche
  davantage le ballon. Le nombre de contacts n'est pas journalisé actuellement.
- [ ] Faire un bilan avant de modifier l'algorithme, le bot ou l'environnement.

Ce budget est un premier cadre d'observation, pas un seuil auquel le modèle
devrait forcément savoir jouer. La commande n'arrête pas automatiquement
l'entraînement à ces paliers. Garder plusieurs cycles pour juger une tendance.

Le joueur reçoit déjà des pénalités lorsqu'il encaisse un but. Ce qui manque
surtout, ce sont des expériences utiles et des récompenses positives. Perdre
n'empêche pas tout apprentissage, mais prolonger ne garantit pas non plus
qu'il découvrira seul les bons comportements.

Si les résultats progressent durablement, prolonger. Si les contacts et les
buts restent rares sans tendance favorable au bilan, étudier les pistes plus
bas. Une erreur de parité, des valeurs non finies ou une collecte bloquée
demandent un diagnostic immédiat.

## 3. Équipe bleue/rouge et type de joueur : deux réglages différents

**Mettre un joueur dans l'équipe rouge ne le transforme pas en bot scripté.**
L'équipe détermine son camp ; le composant actif détermine qui le pilote.

| Réglage | À quoi ça sert |
| --- | --- |
| `MatchManager` → `Teams` → `Players` | Répartir les joueurs entre bleu et rouge |
| `FootAgent` actif | Utiliser la politique du comportement, par exemple `Foot` |
| `SimpleBotInput` actif | Utiliser le bot à règles déjà présent dans le projet |
| `HumanPlayerInput` actif | Piloter le joueur avec les commandes humaines |
| `Record Experience` dans `FootAgent` | Envoyer les expériences à Python |

Le champ `Team` de `PlayerController` vaut `Blue` par défaut, mais le
`MatchManager` le remplace au lancement. Voir `Blue` sur les deux joueurs
avant Play ne prouve donc pas qu'ils jouent dans la même équipe.
Dans la scène enregistrée au moment de cette vérification, les listes du
`MatchManager` contiennent bien un joueur bleu et un joueur rouge.

À vérifier dans Unity, hors Play :

- [ ] Sur `Arena`, ouvrir `Match Manager` → `Teams`.
- [ ] Vérifier les identifiants `Blue`/`Red`, les listes `Players` et les points d'apparition.
- [ ] Pour un duel, mettre un joueur dans chaque équipe, sans le référencer dans les deux listes.
- [ ] Vérifier les composants actifs directement sur chaque joueur.
- [ ] Relancer Play et contrôler l'équipe effective dans `Player Controller`.

Pour essayer **PPO contre le bot de base**, on peut choisir cette configuration :

| Joueur | `FootAgent` | `SimpleBotInput` | `HumanPlayerInput` |
| --- | --- | --- | --- |
| Bleu contrôlé par PPO | Activé, comportement `Foot` | Désactivé | Désactivé |
| Rouge contrôlé par le bot | Désactivé | Activé, à ajouter s'il manque | Désactivé |

On peut évidemment inverser les couleurs. Garder `PlayerController` actif
des deux côtés : c'est lui qui exécute les déplacements et les frappes.
Ne pas laisser plusieurs contrôleurs donner des commandes au même joueur.

Le menu **Préparer la scène (tous les joueurs deviennent des agents)** réactive
`FootAgent` et désactive les bots sur tous les joueurs. Il faut donc vérifier
à nouveau les composants si on l'utilise après avoir préparé PPO contre bot.

Cette configuration choisit les contrôleurs. Pour une vraie évaluation,
il faudra aussi figer la politique et empêcher les mises à jour PPO.

## 4. Checkpoints et reprise

La sauvegarde et la reprise fonctionnent. Les détails et commandes sont dans
[README_PPO.md](README_PPO.md).

### Contenu d'un checkpoint

- [x] `state_dict` de l'acteur, incluant le paramètre `log_std`.
- [x] `state_dict` du critique.
- [x] `state_dict` de chacun des deux optimiseurs Adam.
- [x] Configuration PPO et schéma des observations/actions pour vérifier la compatibilité.
- [x] Compteurs de cycles, transitions entraînées/ignorées et versions.
- [x] Numéro de format et état du générateur aléatoire PyTorch CPU.

Adam conserve des statistiques des gradients précédents. Restaurer seulement
les réseaux ne restaure donc pas complètement l'état d'apprentissage.
Les états aléatoires CUDA/Unity et l'état physique du match ne sont pas
sauvegardés. La reprise ouvre une nouvelle collecte ; elle ne reprend pas
une partie à l'image exacte où elle a été interrompue.

### Intégration réalisée

- [x] Module `src/football_rl/ppo_checkpoint.py`.
- [x] Sauvegarde après un cycle PPO terminé, avant la publication suivante.
- [x] Écriture dans un fichier temporaire puis remplacement du fichier final pour
  éviter de laisser un checkpoint incomplet en cas d'interruption d'écriture.
- [x] `latest.pt` après chaque cycle et archive `cycle_XXXXXX.pt` tous les dix cycles.
- [x] Options `--checkpoint-dir` et `--resume`.
- [x] À la reprise, vérification du schéma et restauration des réseaux, Adam et compteurs.
- [x] Publication, attente de l'acceptation et nouveau buffer vide.
- [x] Chargement CPU/GPU vérifié avec de nouvelles mises à jour.

Les réglages d'apprentissage du checkpoint sont conservés à la reprise.
Seuls `device`, `log_dir` et `collect_only` sont remplacés dans la configuration.
Utiliser un nouveau nom de run pour les nouveaux événements. Il n'y a pas
de sauvegarde automatique avant le premier cycle terminé, ni en `--collect-only`.

### À consolider plus tard

- [x] Test versionné du critique et de son historique Adam après une mise à jour.
- [x] Reprise réelle dans Unity et poursuite de l'apprentissage.
- [ ] Ajouter aux tests versionnés les vérifications ponctuelles déjà effectuées :
  reprise complète, compteurs, RNG, schéma incompatible, erreur d'écriture et reconnexion.
- [ ] Étendre la couverture à l'acteur entraîné, dont `log_std`.

## 5. Suivi TensorBoard

Le `SummaryWriter` est branché dans `ppo_logging.py`. Le guide de lecture complet
est dans [README_PPO.md](README_PPO.md).

Dans un deuxième terminal, depuis la racine du dépôt avec `.venv` activé :

```powershell
python -m tensorboard.main --logdir runs --port 6006
```

Ouvrir [TensorBoard](http://localhost:6006), sélectionner le run dans **Runs**
et les groupes dans **Scalars** ou **Time Series**. Choisir l'axe **Step**
(transitions entraînées), puis commencer avec **Smoothing = 0** avant de lisser.
Un point arrive à la fin d'un cycle, pas à chaque décision Unity.

### Données déjà transmises par Unity

Le pont fournit `batch.episodes`, que le trainer agrège pour le suivi.

| Champ | Contenu |
| --- | --- |
| `agent_id` | Joueur concerné ; le `hello` permet de retrouver son nom et son équipe |
| `episode_id` | Identifiant de l'épisode |
| `return` | Somme des récompenses de l'épisode |
| `length` | Nombre de décisions de l'épisode |
| `outcome` | Par exemple `goal_scored`, `goal_conceded`, `timeout`, `interrupted` |

- [x] Dossier TensorBoard distinct avec `--run-name`.
- [x] Pertes acteur/critique, KL, fraction de clipping et normes des gradients.
- [x] Nombre d'époques effectuées et durées d'optimisation/collecte.
- [x] Retours moyens et écarts-types sur les résumés accumulés entre deux écritures.
- [x] Longueurs d'épisode, nombre de résumés et proportions de chaque type de fin.
- [x] Écarts-types des quatre actions continues.
- [x] Vidage du writer après écriture et fermeture à l'arrêt.
- [ ] Distinguer les statistiques par joueur/équipe lorsque cela aide l'interprétation.
- [ ] Ajouter les probabilités ou entropies des boutons.
- [ ] Si nécessaire, ajouter un compteur explicite des contacts avec le ballon ;
  il n'est pas déductible précisément du retour total actuel.

À regarder en premier : `episodes/outcome_goal_scored`,
`episodes/outcome_goal_conceded`, `episodes/outcome_timeout`,
`episodes/return_mean`, `episodes/length_mean` et `episodes/count`.
Les proportions concernent des manches, pas des matchs complets.
`episodes/count` est l'effectif du point, pas un cumul. Une durée plus longue
peut signifier mieux défendre ou rester bloqué ; confronter aux autres courbes.

Ensuite : KL, clipping, pertes, exploration et parité Python/Unity.
Une perte critique basse peut correspondre à des défaites prévisibles.
Les métriques d'optimisation sont celles de la dernière époque du cycle ;
un arrêt KL peut laisser une ou deux époques seulement, ce qui est prévu.

Les résumés sont traités séparément du filtrage des lignes du buffer PPO :
un épisode peut traverser plusieurs versions de politique. Ces statistiques
décrivent la collecte, pas une évaluation à politique figée.

Le logger regroupe les résumés des agents actifs. Avec deux équipes apprenantes,
un but peut produire un résumé positif et un négatif et masquer les progrès
dans la moyenne. Dans le duel actuel, seul le joueur réseau est enregistré.
Une récompense croissante reste à confronter aux résultats de jeu et à
l'évaluation suivante. Aucune métrique n'est écrite en `--collect-only`.

## 6. Préparer l'évaluation

L'idée : pouvoir répondre à « est-ce que cette version joue mieux que la précédente ? ».

On a déjà `SimpleBotInput`. C'est un bot à heuristiques : il se place derrière
le ballon, le pousse vers le but, tire quand il est aligné et tente des tacles.
Pas besoin d'en refaire un pour commencer. D'abord vérifier qu'il se comporte
correctement dans notre scène, puis garder ses paramètres fixes pour comparer.

- [x] Configurer un joueur PPO contre le bot fixe dans Unity.
- [x] Garder un seul contrôleur actif par joueur : agent PPO ou bot.
- [ ] Charger un checkpoint et garder ses poids figés pendant toute l'évaluation.
- [ ] Empêcher les transitions d'évaluation de déclencher une optimisation.
- [ ] Comparer le modèle initial et plusieurs checkpoints contre le même bot.
- [ ] Alterner les côtés rouge/bleu et conserver les mêmes règles et conditions de départ.
- [ ] Choisir et conserver le même mode d'action pour les modèles comparés
  (stochastique ou déterministe). `actor.eval()` seul ne désactive pas les tirages du décodeur.
- [ ] Commencer par exemple avec 20 à 50 épisodes par modèle pour une première
  estimation, puis répéter pour juger la variabilité.
- [ ] Mesurer séparément buts marqués, buts encaissés et fins par limite de temps.
- [ ] Comparer les résultats et leur incertitude avant de conclure à une amélioration.
- [ ] Sortir un petit fichier CSV avec le modèle, le côté joué, l'adversaire,
  les réglages de la scène, le résultat et la durée de chaque épisode.
- [ ] Faire un tableau comparatif lisible, réutilisable pour les autres algorithmes du projet.

**À faire après la première phase de référence :** brancher un lancement
d'évaluation sur le chargement de checkpoint déjà disponible, fixer les règles
de comparaison et le format des résultats. Le trainer actuel continue à apprendre.
`--collect-only --resume ...` garde les poids fixes et vérifie les données,
mais désactive TensorBoard et ne produit pas un bilan d'évaluation dédié.

**Résultat attendu :** une procédure qu'on peut relancer avec plusieurs modèles,
sans refaire la configuration à la main à chaque fois.

Décocher `Record Experience` sur un joueur ne change pas la politique qui le
contrôle. Pour avoir un adversaire différent, il faut utiliser le bot ou un
autre comportement auquel on fournit une politique distincte.

## 7. Idées pour améliorer le modèle et l'apprentissage RL

Ici, l'idée est de changer ce que le modèle apprend, sa représentation du jeu
ou la manière dont PPO le met à jour. Ce sont des pistes à tester, pas des
améliorations garanties. Garder une version de référence et ajouter une seule
idée à la fois pour pouvoir comprendre son effet.

### Encourager l'exploration avec un bonus d'entropie

Aujourd'hui, les actions sont aléatoires, mais rien dans la perte acteur
n'encourage explicitement à conserver de la diversité. Le modèle peut donc
devenir très sûr de lui avant d'avoir trouvé de bonnes stratégies.

- [ ] Calculer l'entropie des quatre normales et des trois Bernoulli.
- [ ] Ajouter un bonus d'entropie dans la perte acteur, avec un coefficient réglable.
- [ ] Vérifier le signe : comme on minimise la perte, le bonus doit favoriser
  une entropie plus grande, pas la réduire.
- [ ] Suivre séparément la diversité des déplacements/frappes et celle des boutons.
- [ ] Vérifier les gradients, puis comparer avec la version sans bonus.

**À regarder :** le modèle explore-t-il des comportements utiles plus longtemps,
ou devient-il simplement plus agité ? Cette piste reste côté Python et garde
les mêmes distributions d'actions dans Unity.

### Donner au critique son propre rythme d'apprentissage

Actuellement, l'arrêt KL termine les époques de l'acteur et du critique en
même temps. Pourtant, un acteur qui a déjà suffisamment changé n'implique
pas que le critique ait fini d'apprendre les valeurs des situations.

- [ ] Séparer les mises à jour acteur/critique pour pouvoir continuer à ajuster
  le critique lorsque l'acteur atteint sa limite KL.
- [ ] Garder les valeurs de référence, les avantages et les retours calculés
  au début du cycle : ne pas les recalculer à chaque mise à jour du critique.
- [ ] Mesurer la qualité des valeurs avec l'erreur et la variance expliquée,
  en traitant le cas où les retours ont une variance presque nulle.
- [ ] En essai séparé, comparer l'erreur quadratique à une perte de Huber pour
  voir l'effet des erreurs exceptionnellement grandes. Les buts rares doivent
  quand même rester un signal utile, pas être systématiquement sous-pondérés.

**Pourquoi ça peut aider :** des estimations de valeur plus fiables peuvent
donner des avantages moins bruités aux collectes suivantes. C'est une évolution
de `ppo_training.py` ; le critique reste dans Python.

### Travailler la représentation des observations

Pour l'instant, deux couches de 128 neurones reçoivent directement le vecteur
d'observation. Avant de simplement grossir le réseau, on peut améliorer les
conditions dans lesquelles il apprend.

- [ ] Tester une initialisation orthogonale des couches, avec une dernière
  couche acteur de faible amplitude pour éviter une politique très tranchée dès le départ.
- [ ] Essayer `LayerNorm` dans les couches cachées pour normaliser leurs activations.
- [ ] Examiner les distributions des entrées. Unity les met déjà à l'échelle :
  ajouter une normalisation seulement si un problème concret apparaît.
- [ ] Si une normalisation d'entrée est ajoutée, figer ses statistiques pendant
  chaque collecte et ses mises à jour PPO, puis publier exactement la même
  transformation avec la politique suivante.
- [ ] Vérifier à nouveau les sorties exportées et les log-probabilités Python/Unity.

**À garder en tête :** l'initialisation ne change pas le format d'export et
`LayerNorm` est déjà pris en charge par l'exporteur. Une normalisation appliquée
seulement en Python ferait en revanche jouer et entraîner deux politiques différentes.

### Apprendre aussi à prédire ce qui va se passer

Le buffer contient déjà `obs`, `actions` et `next_obs`. On peut s'en servir
pour ajouter une tâche auxiliaire : prévoir, par exemple, la position ou la
vitesse suivante du ballon à partir de la situation et de l'action.

- [ ] Ajouter une tête de prédiction sur les caractéristiques internes du critique.
- [ ] Lui fournir l'action lorsque la prédiction porte sur ses conséquences.
- [ ] Construire les cibles à partir des champs correspondants de `next_obs`,
  dans le bon repère, sans raccorder deux épisodes à travers un reset.
- [ ] Combiner cette erreur auxiliaire avec la perte de valeur et mesurer les deux séparément.
- [ ] Comparer avec le critique habituel : mieux prédire le ballon ne garantit
  pas automatiquement de mieux estimer les récompenses futures.

**L'idée :** pousser le réseau à comprendre des régularités du jeu en plus du
signal de récompense. En commençant par le critique, cette expérience reste
entièrement en Python et ne modifie pas l'acteur exporté.

### Utiliser les symétries du terrain dans l'apprentissage

Des situations en miroir gauche/droite devraient souvent appeler des actions
en miroir. On peut essayer d'apprendre cette cohérence au réseau au lieu de
lui faire découvrir chaque cas séparément.

- [ ] Vérifier quelles symétries sont réellement respectées par la scène et les récompenses.
- [ ] Définir la transformation correspondante des positions, vitesses,
  orientations et actions, notamment le déplacement latéral et l'effet du tir.
- [ ] Ajouter une perte auxiliaire de cohérence entre les distributions
  d'actions de deux observations en miroir, plutôt que comparer deux tirages aléatoires.
- [ ] Tester aussi la cohérence des valeurs prédites par le critique.
- [ ] Garder les transitions PPO originales et leurs anciennes log-probabilités.
  Ne pas dupliquer naïvement des transitions transformées dans la perte PPO :
  rien ne garantit que l'ancienne politique ait produit ces actions en miroir.

**Ce qu'on cherche :** réutiliser plus efficacement les situations déjà vues
et réduire les préférences artificielles pour un côté. Il faut d'abord valider
la transformation ; une mauvaise symétrie enseignerait le mauvais comportement.

### Ajouter une récompense de nouveauté calculée en Python

Si le modèle découvre très peu de situations utiles, une piste est de lui
donner un petit bonus lorsqu'il rencontre quelque chose de peu familier.
Une possibilité : un réseau cible aléatoire figé et un autre réseau qui
essaie de prédire sa sortie. Une erreur de prédiction élevée sert de signal
de nouveauté ; c'est le principe de la distillation de réseau aléatoire, ou RND.

- [ ] Calculer ce bonus à partir des observations reçues, avec les deux réseaux côté Python.
- [ ] Conserver séparément la récompense du jeu et le bonus interne.
- [ ] Calculer les bonus utilisés par GAE avant les époques PPO et les garder
  fixes sur ce rollout, même si le prédicteur est ensuite entraîné.
- [ ] Sauvegarder aussi le prédicteur, le réseau cible et les statistiques nécessaires.
- [ ] Vérifier que le modèle découvre davantage de situations intéressantes,
  plutôt que chercher des états inhabituels sans intérêt pour le football.

**Piste plus avancée :** elle change le signal d'apprentissage côté Python.
L'évaluation doit continuer à mesurer les résultats du jeu, sans ajouter le
bonus de nouveauté au score présenté comme une performance au football.

### Apprendre contre plusieurs versions figées de la politique

Jouer uniquement contre la politique courante peut pousser le modèle à
s'adapter à un seul style qui change en même temps que lui. Un petit ensemble
d'anciens checkpoints permettrait de rencontrer différents comportements.

- [ ] Conserver quelques politiques historiques comme adversaires.
- [ ] Garder l'adversaire choisi fixe pendant une collecte et gérer proprement
  les frontières de trajectoire lorsqu'il change.
- [ ] Utiliser des comportements distincts pour la politique entraînée et l'adversaire.
- [ ] Ne mettre dans le buffer PPO que les expériences de la politique entraînée,
  avec ses propres anciennes log-probabilités ; ne pas rejouer les vieux rollouts.
- [ ] Vérifier les résultats contre des adversaires réservés à l'évaluation.

**À prévoir :** sélection des checkpoints et gestion de plusieurs politiques
côté Python, plus la configuration des joueurs. Le pont sait déjà envoyer un modèle à un
comportement donné ; il n'y a pas besoin de changer le moteur physique.

### Pour plus tard : donner de la mémoire au modèle

Si une observation seule ne suffit pas à choisir une bonne action, un modèle
avec mémoire pourrait utiliser plusieurs instants pour comprendre la situation.
À vérifier d'abord : les observations contiennent déjà des vitesses, donc
ajouter de la mémoire n'est pas automatiquement utile.

- [ ] Chercher des exemples où l'historique apporte une information manquante.
- [ ] Étudier un acteur/critique récurrent, par exemple avec un GRU ou un LSTM.
- [ ] Adapter le buffer et les mini-lots à des séquences, avec masques et états
  cachés séparés par agent, remis à zéro aux bons moments.
- [ ] Reproduire le même traitement temporel lors de la collecte et de l'optimisation.

**Limite actuelle :** le pont et l'exporteur utilisés ici ne gèrent pas ces
réseaux récurrents et leurs états cachés. Cette piste demanderait une extension
dédiée ; elle reste à part tant que le pont doit rester inchangé. Empiler des
observations uniquement en Python ne suffirait pas non plus à équiper Unity
du même historique.

### Quelques points de départ raisonnables

| Piste | Ce qui change | Faisable sans étendre le pont actuel ? |
| --- | --- | --- |
| Bonus d'entropie | Objectif de l'acteur | Oui |
| Critique entraîné séparément | Organisation des mises à jour | Oui |
| Initialisation / `LayerNorm` | Représentation et démarrage du réseau | Oui, avec vérification de l'export |
| Prédiction auxiliaire sur le critique | Informations apprises par le critique | Oui |
| Cohérence par symétrie | Régularisation des réseaux | Oui, après validation des transformations |
| Bonus de nouveauté | Récompense utilisée pour apprendre | Oui, avec des réseaux Python supplémentaires |
| Adversaires historiques | Diversité des situations rencontrées | Oui, avec adaptation du lancement et des comportements |
| Réseau avec mémoire | Politique dépendant d'un historique | Non, extension nécessaire |

Après le bilan de la référence, le bonus d'entropie et la séparation des mises
à jour du critique sont deux sujets assez ciblés. La prédiction auxiliaire et
les symétries donnent des expériences plus poussées. Ces pistes restent en
attente pendant la phase où les réglages sont conservés.

## 8. Autres idées

Les pistes déjà présentes restent utiles : elles donnent des situations de
test et des mesures pour juger les changements apportés au modèle.

### Préparer quelques situations faciles à comparer

- [ ] Un ballon immobile à aller chercher.
- [ ] Un joueur déjà bien placé face au but pour tester la frappe.
- [ ] Un ballon près de notre but pour observer la défense.
- [ ] Les mêmes situations en échangeant rouge et bleu.
- [ ] Noter les critères de réussite : ballon touché, but marqué, but évité,
  nombre de décisions nécessaires, etc.

Ça permet de voir ce que PPO sait faire avant de juger un match entier.
On peut préparer ces essais dans une scène séparée avec les composants
existants, sans changer la physique ni le fonctionnement du pont.

### Si nécessaire après le bilan : apprendre progressivement

- [ ] Commencer avec un adversaire immobile pour découvrir les contacts et les frappes.
- [ ] Passer à une variante de `SimpleBotInput` moins rapide ou moins réactive.
- [ ] Revenir progressivement au bot normal.
- [ ] Garder le bot de référence inchangé pour les évaluations.
- [ ] Conserver les capacités et l'ordre des observations pendant ces étapes.

Cette piste est différée. D'abord observer l'entraînement actuel sur le budget
prévu ; la difficulté ne sera modifiée qu'après ce bilan.

### Vérifier les récompenses et chercher les comportements bizarres

- [ ] Relire les récompenses déjà configurées dans `FootAgent`.
- [ ] Observer ce qui est récompensé quand un joueur reste immobile, suit le
  ballon, tire dans le vide ou marque un but.
- [ ] Vérifier que rouge et bleu sont traités de la même manière dans des
  situations symétriques.
- [ ] Chercher si une stratégie peu utile au football permet quand même
  d'accumuler des récompenses.
- [ ] Écrire les observations et les pistes éventuelles avant de changer quoi que ce soit.

Le but est de comprendre le signal donné à PPO. Une grosse récompense ne
garantit pas automatiquement un bon comportement sur le terrain.

### Faire des expériences sur les réglages PPO

- [ ] Choisir une expérience de référence et garder ses réglages.
- [ ] Comparer quelques taux d'apprentissage de l'acteur.
- [ ] Tester la taille du rollout ou le nombre maximal d'époques, un réglage à la fois.
- [ ] Garder le même budget de transitions et le même protocole d'évaluation.
- [ ] Refaire les essais avec plusieurs graines pour éviter de conclure sur un seul lancement chanceux.
- [ ] Noter les résultats, y compris quand une modification n'aide pas.

Les checkpoints et les courbes sont disponibles. Attendre le bilan de la
référence avant de lancer ces comparaisons.

### Renforcer les tests de la partie RL

- [ ] Construire de petites trajectoires dont on peut calculer les retours et GAE à la main.
- [ ] Couvrir un but, une limite de temps, un trou dans les pas et deux joueurs mélangés.
- [ ] Vérifier le clipping avec des avantages positifs et négatifs.
- [ ] Vérifier que les anciennes log-probabilités et les cibles restent fixes pendant les époques.
- [ ] Comparer les résultats CPU/GPU à une tolérance raisonnable.

Ça permet de comprendre le cœur de PPO en profondeur et de repérer des
erreurs qui ne se voient pas forcément en regardant les joueurs.

### Pour plus tard : varier les adversaires

- [ ] Préparer quelques réglages du bot existant, par exemple avec un temps
  de réaction plus long ou une vitesse réduite.
- [ ] Garder un bot de référence inchangé, même si on crée des variantes.
- [ ] Essayer des styles plus défensifs si le modèle exploite toujours la même faiblesse.
- [ ] Comparer le modèle à d'anciens checkpoints figés.

Les anciens checkpoints demanderaient des comportements distincts pour les
deux politiques et une adaptation du lancement Python ; le pont sait déjà
cibler un modèle par comportement. C'est une piste après l'évaluation simple,
pas quelque chose que le trainer actuel fait automatiquement.

Garder des formats communs pour les checkpoints et les résultats, ainsi que
des règles d'évaluation stables. Les changements de scène ou de bot éventuels
peuvent rester sur une branche dédiée, avec le moteur de jeu et le protocole
du pont conservés.

## 9. Rappels pour relancer les essais

Depuis le dépôt Python, avec `.venv` activé et la variante CUDA de PyTorch installée :

```powershell
# Diagnostic : aucune optimisation.
python -m football_rl.ppo_unity --collect-only --rollout-size 512 --device cuda --verbose

# Continuer l'expérience existante après un arrêt de Python.
python -m football_rl.ppo_unity --device cuda --resume checkpoints/checkpoint_test_reprise/latest.pt --run-name ppo_bot_reference_01
```

Si l'entraînement tourne déjà, le laisser continuer. Après un arrêt du nouveau
run, reprendre son propre `checkpoints/ppo_bot_reference_01/latest.pt` avec un
nouveau nom d'exécution. Reprendre systématiquement l'ancien fichier ferait
perdre les progrès plus récents.

Dans un deuxième terminal :

```powershell
python -m tensorboard.main --logdir runs --port 6006
```

Dans Unity :

- `TrainingBridge` : `Include Next Observations` activé.
- Joueurs à enregistrer : `Foot Agent` → `Record Experience` activé.
- Exécution en arrière-plan : **Edit → Project Settings → Player → Resolution
  and Presentation → Run In Background** à activer pour continuer hors focus.
- Démarrer le serveur Python, puis lancer Play.

Les valeurs par défaut pour un nouveau lancement sont : lots Unity de 256,
seuil de collecte de 2048, mini-lots de 128, maximum de 4 époques et seuil KL
de 0,01. Le checkpoint de l'essai actuel conserve un seuil de 512. Le buffer
peut dépasser son seuil car les lots reçus sont conservés en entier.

