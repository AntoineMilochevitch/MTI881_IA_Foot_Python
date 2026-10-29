# TODO PPO — idées pour la suite

PPO tourne déjà ; maintenant, on veut
garder les entraînements, comprendre ce que le modèle apprend et voir s'il
joue vraiment mieux. Les idées ci-dessous sont à choisir, pas toutes à faire.

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
- [ ] Sauvegarde sur disque et reprise après redémarrage de Python.
- [ ] Suivi TensorBoard des métriques d'apprentissage et des épisodes.
- [ ] Évaluation du niveau de jeu contre un adversaire fixe.

Dans le dernier extrait de logs partagé :

| Indicateur | Valeur |
| --- | --- |
| Cycles PPO terminés | 5 |
| Transitions utilisées pour l'entraînement | 11 445 |
| Dernière politique acceptée par Unity | v7 |
| Écart maximal de log-probabilité Python/Unity | Au plus `2.86e-06` |
| Tolérance de comparaison | `1e-4` |
| Temps mesuré par cycle d'optimisation | Environ 0,5 à 1,15 seconde, hors collecte Unity |
| Époques effectuées par cycle | 1, 1, 3, 2, 3 |

La version compte les publications : une reconnexion republie les poids avec
une nouvelle version, sans ajouter un cycle d'apprentissage.

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

L'idée : pouvoir fermer Python et reprendre demain avec les mêmes réseaux
et le même état d'apprentissage.

### Contenu d'un checkpoint

- [ ] `state_dict` de l'acteur, incluant le paramètre `log_std`.
- [ ] `state_dict` du critique.
- [ ] `state_dict` de chacun des deux optimiseurs Adam.
- [ ] Configuration PPO et schéma des observations/actions pour vérifier la compatibilité.
- [ ] Compteurs de cycles, de transitions entraînées et de versions.
- [ ] Numéro de format du checkpoint pour gérer les futures évolutions.
- [ ] États des générateurs aléatoires utilisés, notamment PyTorch CPU/CUDA.

Adam conserve des statistiques des gradients précédents. Restaurer seulement
les réseaux ne restaure donc pas complètement l'état d'apprentissage.
Les graines Python ne suffisent pas à restaurer l'état physique et le hasard
d'une partie Unity : la reprise ouvrira une nouvelle collecte.

### Intégration prévue

- [ ] Créer un module dédié, par exemple `src/football_rl/ppo_checkpoint.py`.
- [ ] Sauvegarder automatiquement après un cycle PPO terminé, avant la publication suivante.
- [ ] Écrire dans un fichier temporaire puis remplacer le fichier final pour
  éviter de laisser un checkpoint incomplet en cas d'interruption d'écriture.
- [ ] Conserver le dernier checkpoint et quelques versions historiques.
- [ ] Ajouter les options de lancement pour choisir le dossier et reprendre un fichier.
- [ ] À la reprise, vérifier le schéma Unity puis restaurer réseaux et optimiseurs.
- [ ] Republier l'acteur avec une version adaptée à la connexion, attendre son
  acceptation et créer un buffer vide. Ne pas rejouer un ancien rollout.
- [ ] Prévoir le chargement CPU/GPU avec le placement correct des états Adam.

Les options de sauvegarde et de reprise sont **à ajouter** : elles ne sont
pas encore disponibles dans la commande actuelle.

### Comment vérifier que ça marche

- [ ] Sauvegarder après une vraie mise à jour, puis recréer les réseaux et recharger.
- [ ] Vérifier les mêmes sorties de l'acteur, les mêmes valeurs du critique
  et les mêmes `log_std` pour des observations fixes.
- [ ] Vérifier la restauration des états Adam et des compteurs.
- [ ] Effectuer un nouveau cycle après rechargement.
- [ ] Refuser proprement un checkpoint incompatible avec le schéma Unity.

## 5. Suivi TensorBoard

L'idée : avoir des courbes qu'on peut regarder ensemble, avec le nombre de
transitions comme axe pour comparer les entraînements.

TensorBoard est déjà une dépendance du projet. Le `SummaryWriter` reste à brancher.

### Données déjà transmises par Unity

Le pont fournit `batch.episodes`. Le trainer actuel ne traite pas encore ces résumés.

| Champ | Contenu |
| --- | --- |
| `agent_id` | Joueur concerné ; le `hello` permet de retrouver son nom et son équipe |
| `episode_id` | Identifiant de l'épisode |
| `return` | Somme des récompenses de l'épisode |
| `length` | Nombre de décisions de l'épisode |
| `outcome` | Par exemple `goal_scored`, `goal_conceded`, `timeout`, `interrupted` |

- [ ] Créer un dossier TensorBoard distinct pour chaque expérience.
- [ ] Tracer les pertes acteur/critique, KL, fraction de clipping et normes des gradients.
- [ ] Tracer le nombre d'époques réellement effectuées et les durées d'optimisation/collecte.
- [ ] Tracer les retours moyens et leur dispersion sur une fenêtre d'épisodes.
- [ ] Tracer les longueurs d'épisode et les proportions de chaque type de fin.
- [ ] Distinguer les statistiques par joueur/équipe lorsque cela aide l'interprétation.
- [ ] Suivre l'exploration : écarts-types continus et probabilités ou entropies des boutons.
- [ ] Fermer et vider correctement le writer lors de l'arrêt.

Traiter les résumés d'épisodes séparément du filtrage des lignes du buffer PPO :
un épisode peut traverser plusieurs versions de politique. Ces statistiques
décrivent la collecte, pas une évaluation à politique figée.

Éviter aussi de compter un but deux fois : dans notre duel, le même but peut
produire un résumé `goal_scored` pour un joueur et `goal_conceded` pour l'autre.
Utiliser la session, l'agent et l'épisode pour identifier les résumés sans
mélanger les reconnexions.

Ces mesures de base peuvent être ajoutées côté Python avec les données déjà
fournies par le pont. Une récompense croissante est un indice à confronter aux
résultats de jeu et à l'évaluation suivante.

## 6. Préparer l'évaluation

L'idée : pouvoir répondre à « est-ce que cette version joue mieux que la précédente ? ».

On a déjà `SimpleBotInput`. C'est un bot à heuristiques : il se place derrière
le ballon, le pousse vers le but, tire quand il est aligné et tente des tacles.
Pas besoin d'en refaire un pour commencer. D'abord vérifier qu'il se comporte
correctement dans notre scène, puis garder ses paramètres fixes pour comparer.

- [ ] Préparer une configuration d'évaluation avec un joueur PPO et un bot fixe.
- [ ] Vérifier qu'un joueur n'a qu'un contrôleur actif : agent PPO ou bot.
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

**Ce qui peut être commencé tout de suite :** une copie de la scène pour les
évaluations, la configuration PPO/bot, les règles de comparaison et le format
des résultats. Le chargement des différentes versions sera branché une fois
les checkpoints prêts. La première vérification du bot peut se faire face à
un humain ou à un autre bot.

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

**À prévoir :** checkpoints et gestion de plusieurs politiques côté Python,
plus la configuration des joueurs. Le pont sait déjà envoyer un modèle à un
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

Pour démarrer, le bonus d'entropie et la séparation des mises à jour du
critique sont deux sujets assez ciblés. Ensuite, la prédiction auxiliaire
ou les symétries donnent des expériences plus poussées sur ce que le réseau apprend.

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

Ça devient vraiment utile après les checkpoints et les courbes. Les
configurations et le tableau de comparaison peuvent être préparés en parallèle.

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

# Entraînement : utiliser cette commande après avoir arrêté le diagnostic.
python -m football_rl.ppo_unity --device cuda
```

Dans Unity :

- `TrainingBridge` : `Include Next Observations` activé.
- Joueurs à enregistrer : `Foot Agent` → `Record Experience` activé.
- Exécution en arrière-plan : **Edit → Project Settings → Player → Resolution
  and Presentation → Run In Background** à activer pour continuer hors focus.
- Démarrer le serveur Python, puis lancer Play.

Les valeurs par défaut sont : lots Unity de 256, seuil de collecte de 2048,
mini-lots de 128, maximum de 4 époques et seuil KL de 0,01. Le buffer peut
dépasser 2048 car les lots reçus sont conservés en entier.

