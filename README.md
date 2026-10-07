# Football 3D — backend Python

Backend PyTorch pour comparer des algorithmes de RL dans un jeu de football Unity,
d'abord en 1v1, puis en équipe. Unity simule les matchs, exécute la politique et
calcule les récompenses. Python exporte les modèles et reçoit les transitions.

## État actuel

| Élément | État |
| --- | --- |
| Accès au paquet `iafoot` du dépôt Unity | Script de liaison pour chaque environnement virtuel |
| Envoi d'une politique et réception des lots | Disponible avec `python -m football_rl.unity` |
| Export | Modèles et décodeurs fournis par `iafoot.models` |
| Algorithmes, buffers, pertes, optimiseurs et checkpoints | Développés dans les branches dédiées |
| TensorBoard | Dépendance installée ; métriques à brancher |

La commande de collecte utilise par défaut une politique linéaire de démonstration.
Une fabrique configurable permet d'envoyer le modèle d'un algorithme. Les lots
sont affichés puis libérés, sans apprentissage ni sauvegarde des transitions.

- [Installation et utilisation du pont](README_UNITY.md)

## Installation

Prérequis : Python 3.12 ou plus récent et les deux dépôts côte à côte :

```text
MTI881_IA_Foot/
├── MTI881_IA_Foot_Python/
└── MTI881_IA_Foot_Unity/
    └── python/iafoot/
```

Si le dépôt parent vient d'être cloné, exécuter depuis celui-ci :

```powershell
git submodule update --init --recursive
```

Puis, **depuis `MTI881_IA_Foot_Python`** :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python scripts/link_unity_bridge.py
```

Pour un environnement existant, conserver `.venv` et exécuter les deux dernières
commandes avec son interpréteur. Sous Linux/macOS, l'activation est
`source .venv/bin/activate`.

Le script enregistre le chemin du paquet Unity dans un fichier `.pth` de cet
environnement. Le relancer après un déplacement des dépôts ou une recréation de
`.venv`. Chaque collègue configure son propre environnement ; aucun chemin
propre à une machine n'est versionné.

Pour un autre emplacement :

```powershell
python scripts/link_unity_bridge.py --unity-python "D:\projets\MTI881_IA_Foot_Unity\python"
```

## Lancer la collecte

Depuis le dépôt Python, avec `.venv` activé :

```powershell
python -m football_rl.unity --batch-size 256 --verbose
```

La commande installée `football-rl-unity` propose les mêmes options. Le serveur
écoute sur `127.0.0.1:5005` et cible le comportement `Foot` par défaut.

Dans Unity, ouvrir la scène de démonstration, utiliser le menu
**IA Foot > Entraînement > Préparer la scène (tous les joueurs deviennent des agents)**,
puis lancer **Play**. Les détails sont dans [README_UNITY.md](README_UNITY.md).

Le serveur récupère la dimension et l'ordre des observations dans `hello`,
exporte la politique, affiche `model_ack`, puis les formes et versions des lots.
`Ctrl+C` arrête le serveur. La graine `--seed` contrôle l'initialisation Python ;
Unity possède son propre générateur pour les tirages d'actions.

## Observations envoyées par Unity

L'ordre et la dimension viennent de `hello.behaviors[].obs_names` et `obs_size`.
Le code source est [FootSpaces.cs](../MTI881_IA_Foot_Unity/Assets/IAFoot/Scripts/Learning/FootSpaces.cs).

| Groupe, dans l'ordre du vecteur | Contenu | Taille |
| --- | --- | ---: |
| Joueur contrôlé | Position 3D, vitesse 3D, orientation X/Z, charge, deux cooldowns, quatre indicateurs d'état | 15 |
| Ballon | Position relative 3D, position dans le terrain 3D, vitesse 3D | 9 |
| Buts | Joueur vers son but en X/Z, ballon vers le but adverse en X/Z | 4 |
| Coéquipiers | Présence, position relative X/Z, vitesse X/Z, état étourdi | 6 par emplacement |
| Adversaires | Même encodage que les coéquipiers | 6 par emplacement |

La dimension vaut **28 + 6 × (maximum de coéquipiers + maximum d'adversaires)**.
La scène examinée réserve un coéquipier et deux adversaires : **46 valeurs**.
Avec deux coéquipiers et trois adversaires, elle serait de 58. Les emplacements
absents sont remplis de zéros. Les capacités restent fixes pour un acteur donné.

Le repère dépend de l'équipe : **+X est dirigé vers le but adverse**, Y est
vertical et Z latéral. Il ne suit pas l'orientation du joueur. Certaines
positions sont relatives, d'autres absolues dans ce repère. Unity divise les
positions par `positionScale` et les vitesses par `velocityScale` ; les valeurs
ne sont pas nécessairement bornées dans `[-1, 1]`. Python reçoit ces données déjà
préparées. La largeur et la hauteur des buts, le score et le temps de match ne
figurent pas dans ce vecteur.

## Actions appliquées

| Indice | Nom | Sortie du MLP | Utilisation par Unity |
| ---: | --- | --- | --- |
| 0 | `move_x` | Moyenne gaussienne | Déplacement longitudinal |
| 1 | `move_z` | Moyenne gaussienne | Déplacement latéral |
| 2 | `shoot` | Logit de Bernoulli | Déclenchement de la frappe |
| 3 | `shoot_power` | Moyenne gaussienne | Intensité : `clip((a + 1) / 2, 0, 1)` |
| 4 | `shoot_curve` | Moyenne gaussienne | Effet latéral borné dans `[-1, 1]` |
| 5 | `tackle` | Logit de Bernoulli | Déclenchement du tacle |
| 6 | `jump` | Logit de Bernoulli | Déclenchement du saut |

Le décodeur `direct` tire quatre gaussiennes et trois Bernoulli. Les actions
enregistrées sont les **tirages bruts** : les bornes sont appliquées ensuite par
le jeu. Le déplacement est une commande de direction/vitesse, avec composantes
bornées et norme limitée ; l'acteur apprend ainsi à se placer pour frapper.

Une seule mécanique de frappe sert aux passes et aux tirs. L'agent choisit le
déclenchement, la puissance et l'effet ; la direction initiale dépend de
l'orientation du joueur. Une commande peut rester sans effet si les règles du
jeu l'empêchent, par exemple pendant un cooldown.

## Épisodes et récompenses

Un but termine l'épisode (`terminated=True`). Les coupures de durée ou remises
en jeu techniques sont signalées par `truncated`. Conserver
`includeNextObservations` activé sur `TrainingBridge` pour le futur bootstrap.

Les récompenses sont calculées par `FootAgent` dans Unity et cumulées entre deux
décisions. Valeurs par défaut du code, modifiables dans l'Inspector :

| Composante | Valeur par défaut |
| --- | ---: |
| But de l'équipe / but adverse | +1 / −1 |
| Contact ou frappe du ballon | +0,02 |
| Vitesse du ballon vers le but adverse | 0,1 × vitesse X normalisée × durée |
| Temps écoulé | −0,005 × durée en secondes |
| Distance au ballon et tacle réussi | Désactivées par défaut, coefficients nuls |

Les algorithmes utilisent directement `batch.rewards`, calculé par Unity.

## Architecture Python

| Fichier | Rôle |
| --- | --- |
| `scripts/link_unity_bridge.py` | Lie le paquet Unity à l'environnement virtuel |
| `src/football_rl/unity.py` | Collecteur commun, fabrique de politique, callbacks `iafoot` |
| `src/football_rl/network.py` | Base des réseaux PyTorch |
| `README_UNITY.md` | Configuration, protocole, utilisation et dépannage |

Le transport, les lots et les exports sont définis dans le paquet du dépôt Unity :
`iafoot.Batch`, `iafoot.TrainingServer`, `iafoot.UnityConnection`, `ModelExport`
et `DecoderExport`. Les anciens contrats abstraits et l'environnement Gymnasium
factice ont été supprimés. Aucune seconde définition du protocole n'est maintenue ici.

## Brancher un algorithme

Le collecteur accepte une fabrique `module:fonction`. Elle reçoit
`(behavior, seed)` et retourne `(ModelExport, DecoderExport)`.

```powershell
python -m football_rl.unity --policy-factory mon_paquet.politique:build_policy
```

Le dictionnaire `behavior` provient du message `hello` : taille et noms des
observations, description des actions, période de décision et agents. Le guide
[README_UNITY.md](README_UNITY.md) précise le contrat de la fabrique.

Pour une véritable boucle d'apprentissage, chaque algorithme conserve son propre
état et utilise les callbacks du `TrainingServer` pour traiter les lots et publier
ses modèles. Le collecteur commun sert à l'intégration et au diagnostic.

## Organisation du travail

`main` contient uniquement l'infrastructure et la documentation communes. Chaque
algorithme dispose d'une branche et d'une documentation dédiées. Les contributions
communes sont intégrées aux branches d'algorithmes avant leur publication sur
`main` ; les fichiers communs restent identiques entre les branches synchronisées.

Les buffers, fonctions de coût, optimisations et stratégies d'évaluation relèvent
de chaque algorithme. TensorBoard est disponible avec `tensorboard --logdir runs`
sur `http://localhost:6006`. La collecte de diagnostic n'écrit pas de métriques.
