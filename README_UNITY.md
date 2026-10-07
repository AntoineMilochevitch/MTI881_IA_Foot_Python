# Utiliser le pont Unity depuis le dépôt Python

## 1. Lien entre les dépôts

Le paquet `iafoot` est fourni par `MTI881_IA_Foot_Unity/python/iafoot`. Il contient
le protocole, le serveur TCP et les fonctions d'export. Le dépôt parent Git
versionne les deux sous-modules ; Python utilise son propre chemin d'import.

Depuis le dépôt Python, après installation avec `python -m pip install -e .` :

```powershell
.\.venv\Scripts\python.exe scripts/link_unity_bridge.py
```

Le script recherche automatiquement le dépôt Unity voisin de ce dépôt Python,
indépendamment du dossier courant. Il vérifie la présence des fichiers du paquet
et écrit `iafoot_bridge.pth` dans le `site-packages` de l'environnement utilisé.
Il exige un environnement virtuel. Relancer la commande remplace le même lien.

Chaque collègue configure son propre `.venv`. Le fichier local n'est pas
versionné. Si les dépôts sont déplacés ou l'environnement recréé, relancer le
script. Pour un autre emplacement, utiliser `--unity-python` suivi du chemin
du dossier contenant `iafoot`.

Le mécanisme repose sur les [fichiers de chemins `.pth` de Python](https://docs.python.org/3/library/site.html).
Les fichiers du pont restent dans le dépôt Unity ; leurs modifications sont
chargées au prochain démarrage Python.

### VS Code

Sélectionner `MTI881_IA_Foot_Python/.venv/Scripts/python.exe` comme interpréteur
(`.venv/bin/python` sous Linux/macOS). Les paramètres `.vscode/settings.json`
facilitent l'analyse si le dépôt Python est ouvert seul. Si le dépôt parent est
ouvert, sélectionner explicitement ce même environnement. Redémarrer le serveur
de langage après la création du lien si un import reste souligné.

Diagnostic de l'interpréteur et de l'import :

```powershell
python -c "import sys, iafoot; print(sys.executable); print(iafoot.__file__)"
```

Le second chemin doit désigner `MTI881_IA_Foot_Unity/python/iafoot/__init__.py`
dans le checkout local.

## 2. Lancer une politique dans Unity

Avec `.venv` activé :

```powershell
python -m football_rl.unity --batch-size 256 --verbose
```

| Option | Défaut | Usage |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Adresse d'écoute Python |
| `--port` | `5005` | Port TCP, identique dans `TrainingBridge` |
| `--behavior` | `Foot` | Comportement auquel envoyer notre acteur |
| `--batch-size` | `256` | Seuil de transitions déclenchant un envoi Unity |
| `--time-scale` | `1` | Vitesse de simulation, entre 0,1 et 50 |
| `--seed` | `0` | Graine transmise à la fabrique de politique |
| `--policy-factory` | Démonstration linéaire | Fonction `module:fonction` construisant les exports |
| `--verbose` | Désactivé | Protocole, noms des observations et exemple d'action |

La commande `football-rl-unity` appelle le même point d'entrée après installation
avec `pip install -e .`. `Ctrl+C` arrête le serveur.

### Préparer Unity

1. Ouvrir `MTI881_IA_Foot_Unity` avec la version indiquée dans son
   `ProjectSettings/ProjectVersion.txt` (6000.3.10f1 dans le checkout examiné).
2. Ouvrir `Assets/Lightning Poly/Football Essentials 3D/Demo Scene/Demo Scene.unity`.
3. Utiliser **IA Foot > Entraînement > Préparer la scène (tous les joueurs deviennent des agents)**.
4. Vérifier `TrainingBridge` : hôte et port du serveur, et
   `includeNextObservations` activé.
5. Conserver `recordExperience` activé sur les joueurs à observer, puis lancer Play.

Le menu prépare les agents et désactive les contrôleurs humains/bots de la scène.
Les agents de même `behaviorName` partagent la politique. Désactiver
`recordExperience` supprime les enregistrements de cet agent, sans lui attribuer
une politique différente. Un adversaire fixe nécessite un autre comportement
ou un contrôleur de bot.

### Ce que fait le serveur Python

1. `on_connect` reçoit `hello` et sélectionne le comportement demandé.
2. Il vérifie les sept actions et lit la dimension, les noms d'observations et la
   période de décision.
3. Il appelle la fabrique choisie et envoie le modèle et son décodeur sous un
   même numéro de version. Par défaut, il utilise une politique linéaire non entraînée.
4. `on_model_ack` affiche l'acceptation ou le motif de refus côté Unity.
5. `on_batch` affiche les formes, les versions et les fins d'épisodes. Les
   données sont ensuite libérées.

Un nouveau `hello` ou une reconnexion réutilise les exports déjà présents en
mémoire. Un changement de schéma pour le même comportement est refusé par ce
serveur. Un redémarrage du programme rappelle la fabrique. Un algorithme peut y
charger un modèle enregistré ou construire une politique initiale.

Les échelles `positionScale` et `velocityScale` ne sont pas annoncées dans
`hello`. Garder les mêmes réglages pour les agents partageant un comportement.

Cette commande permet la chaîne observation → modèle exporté → action Unity →
transition reçue. Elle ne réalise aucun apprentissage et ne sauvegarde pas les
lots. La politique de démonstration ne sait pas jouer au football.

## 3. Protocole et données

Python écoute ; Unity se connecte et tente de se reconnecter après une
interruption. Le pont utilise TCP et des trames comprenant `IAFT`, des longueurs,
un en-tête JSON et des tableaux `float32` ou `int32` en little endian.
`iafoot.protocol` gère cet encodage.

| Message | Direction | Contenu |
| --- | --- | --- |
| `hello` | Unity → Python | Session, comportements, schémas, agents, versions |
| `set_model` | Python → Unity | Modèle, décodeur, version, exemple de contrôle |
| `model_ack` | Unity → Python | Résultat du chargement, erreur éventuelle |
| `batch` | Unity → Python | Transitions et résumés d'épisodes |
| `set_config` | Python → Unity | Vitesse, taille des lots, envoi partiel |

`UnityConnection.set_config` modifie `time_scale`, `batch_size` et peut demander
`flush=True`. Ce flush envoie un lot partiel sans terminer les épisodes. Une
vitesse nulle ne met pas Unity en pause : le moteur la borne au minimum accepté.

### Un lot contient des transitions

Pour `N` transitions et `D` observations, `iafoot.Batch` expose :

| Champ | Forme | Signification |
| --- | --- | --- |
| `obs` | `(N, D)` | Observations ayant produit les actions |
| `actions` | `(N, 7)` | Actions brutes échantillonnées |
| `rewards` | `(N,)` | Récompenses cumulées jusqu'à l'observation suivante |
| `next_obs` | `(N, D)` ou `None` | Observation suivante, avant reset en fin d'épisode |
| `terminated`, `truncated` | `(N,)` | Indicateurs de fin d'épisode |
| `log_probs` | `(N,)` | Log-probabilités calculées pendant la collecte |
| `model_version` | `(N,)` | Version ayant choisi chaque action |
| `agent_id`, `episode_id`, `step` | `(N,)` | Identifiants et ordre temporel |
| `action_index` | `(N,)` | `-1` pour le décodeur `direct` |
| `episodes` | Liste | Résumés des épisodes terminés |

Plusieurs joueurs peuvent être mélangés, et un épisode peut traverser plusieurs
lots. `batch.trajectories()` regroupe par agent et épisode puis trie par pas.
Cette fonction ne découpe ni les changements de version ni les trous entre
pas : les algorithmes qui ont besoin de segments homogènes devront le faire.

En regroupant plusieurs connexions, conserver également la session de
`connection.hello` et le comportement pour éviter les collisions d'identifiants.

## 4. Fabrique de politique et export

La fabrique reçoit deux arguments :

- `behavior` : dictionnaire du comportement issu de `hello` ;
- `seed` : entier choisi par `--seed`.

Elle retourne exactement un couple `(ModelExport, DecoderExport)`. Le collecteur
conserve cet export entre les reconnexions ; cette fabrique n'est pas une boucle
d'apprentissage. Pour les mises à jour, utiliser directement le serveur `iafoot`
et conserver l'état de l'algorithme dans ses callbacks.

Les fonctions `iafoot.models` convertissent des modèles : `from_torch` pour un
`nn.Sequential` compatible, `linear`, `mlp`, arbres et ensembles. Les décodeurs
`direct_decoder` et `discrete_decoder` définissent comment les sorties deviennent
les sept commandes du jeu. Le choix du décodeur appartient à l'algorithme.

Avec le décodeur `direct`, le réseau produit sept sorties : quatre moyennes
continues aux indices `0, 1, 3, 4`, et trois logits binaires aux indices `2, 5, 6`.
`log_std` accepte une valeur partagée ou un vecteur de sept valeurs ; les positions
binaires sont ignorées. Les tirages continus sont conservés avant le bornage du
moteur. La log-probabilité inclut chaque composante, même puissance et effet
lorsqu'aucune frappe n'est déclenchée.

L'envoi utilise `connection.send_model(model, version=..., behavior=..., decoder=...)`.
Un export avec un exemple de contrôle permet à Unity de comparer sa sortie à
celle attendue avant de l'installer. Ce contrôle porte sur le modèle ; un
algorithme utilisant les log-probabilités devra également comparer ses calculs
à ceux du décodeur Unity.

Les callbacks du serveur sont sérialisés. Rendre la main après `send_model`
permet au serveur de traiter `model_ack`. Une attente bloquante de cet accusé de
réception à l'intérieur du callback empêcherait sa réception.

## 5. Contraintes de collecte

Unity continue à simuler pendant que Python travaille. Un lot peut contenir
plusieurs versions, car le remplacement du modèle peut arriver au milieu d'un
épisode. Chaque algorithme définit son traitement des versions, des segments et
de la réutilisation des expériences selon ses hypothèses d'apprentissage.

Le mode stochastique fournit des log-probabilités de collecte. Le mode
`deterministic=True` du décodeur direct renvoie des log-probabilités nulles ; il
faut en tenir compte pour les méthodes qui apprennent à partir de ces valeurs.

Le pont n'est pas un stockage durable : la livraison des lots n'est pas garantie
pendant une déconnexion. La récupération après interruption relève de la boucle
d'entraînement propre à l'algorithme.

## 6. Dépannage

| Symptôme | Action |
| --- | --- |
| `No module named iafoot` | Relancer le script de liaison avec le Python du `.venv` utilisé |
| `No module named football_rl` | Installer le dépôt avec `python -m pip install -e .` |
| Paquet Unity incomplet | Initialiser le sous-module ou corriger `--unity-python` |
| Import souligné dans l'éditeur | Sélectionner l'interpréteur et redémarrer l'analyse Python |
| Aucun `hello` | Lancer Play, vérifier hôte, port et présence de `TrainingBridge` |
| Port déjà occupé | Arrêter l'autre serveur, notamment la démonstration Unity |
| Comportement absent | Vérifier `behaviorName` ou utiliser `--behavior` |
| Modèle refusé | Lire `model_ack`, vérifier schéma, sorties et contrôle numérique |
| `next_obs` absent | Activer `includeNextObservations` sur `TrainingBridge` |

Le serveur de démonstration d'origine reste utilisable depuis le dépôt Python :

```powershell
python ../MTI881_IA_Foot_Unity/python/train.py --model mlp --verbose
```

Il exporte ses propres modèles de démonstration. Le collecteur de ce dépôt
utilise une démonstration linéaire par défaut et accepte une fabrique personnalisée.
