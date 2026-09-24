# Football 3D — Reinforcement Learning Backend

Backend Python pour comparer plusieurs algorithmes d'apprentissage par renforcement avec PyTorch dans un jeu de football 3D sous Unity, avec des cages en hauteur.

L'apprentissage commence en **1v1**, puis évolue vers le jeu en équipe afin d'étudier les comportements coopératifs.

Le projet commence en **1v1** et est structuré pour évoluer vers des équipes et du multi-agents.

## Architecture

```text
.
├── pyproject.toml
├── requirements.txt
├── README.md
└── src/
    └── football_rl/
        ├── __init__.py
        ├── agent.py          # Apprentissage sur un lot et export du modèle
        ├── communication.py  # Réception des épisodes et envoi du réseau à Unity
        ├── env.py            # Interface Gymnasium, raccordement à définir
        ├── network.py        # Base PyTorch commune aux futurs réseaux
        ├── rewards.py        # Composantes et coefficients à déterminer
        ├── train.py          # Contrat d'orchestration et configuration
        └── types.py          # Observations, actions, épisodes, lots et modèle
```

Unity assure la simulation et l'exécution du modèle pendant les parties. Python reçoit les expériences, effectue la rétropropagation avec PyTorch et prépare le réseau mis à jour pour Unity. L'architecture ne fixe aucun algorithme particulier.

## Installation

Prérequis : Python 3.12 ou plus récent et pip.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sous macOS/Linux, activer l'environnement avec :

```bash
source .venv/bin/activate
```

Les dépendances incluent Gymnasium, NumPy, PyTorch et TensorBoard. Après activation de l'environnement, lancer TensorBoard avec :

```powershell
tensorboard --logdir runs
```

L'interface est accessible à l'adresse `http://localhost:6006`. 

## Communication Unity ↔ Python

L'entraînement suit la boucle suivante :

1. Unity exécute le modèle et enregistre des parties.
2. Unity transmet les épisodes par lots au backend Python.
3. Python entraîne le réseau avec PyTorch et met à jour ses poids.
4. Python renvoie le réseau à Unity, qui le charge sur les agents.
5. La collecte reprend jusqu'à convergence ou interruption de l'entraînement.

```text
Unity : modèle en jeu → collecte d'épisodes
                  │
                  └── lot d'épisodes → Python : entraînement PyTorch
                                               │
Unity : chargement du réseau mis à jour ←───────┘
```

`UnityTrainingBridge` décrit la réception d'un `EpisodeBatch` et l'envoi d'un `ModelArtifact`. Les types Python représentent les échanges attendus ; leur sérialisation, le protocole de communication et le format du réseau exporté restent à choisir selon les contraintes de performance et de simplicité. Cette organisation s'inspire de ML-Agents sans imposer son utilisation.

`Football3DEnv` fournit le contrat d'une interface compatible Gymnasium. Son raccordement à Unity et son articulation avec la collecte par lots restent à préciser. Ses méthodes `reset()` et `step()` ne sont pas implémentées.

## Observations

L'agent dispose d'une représentation globale normalisée dans **[-1, 1]**, avec un repère centré sur la position de l'agent et des axes parallèles à ceux du terrain (`x`, `y` vertical, `z`). Ce repère ne tourne pas avec l'agent. Les bornes physiques et les détails d'encodage devront être définis avec Unity.

Pour conserver le même réseau entre le 1v1 et les équipes, `ObservationLayout` fixe dès le départ un maximum de coéquipiers et d'adversaires. Chaque emplacement de joueur contient sa position relative 3D, sa vélocité 3D et un masque de présence. Un emplacement absent contient des zéros, avec un masque nul ; le modèle devra exploiter ces masques pour ignorer les joueurs absents.

La convention Python utilise un vecteur `Box` de type `float32`, dans cet ordre :

| Groupe | Valeurs | Dimensions |
| --- | --- | ---: |
| Ballon | Position relative et vélocité 3D | 6 |
| Agent | Vélocité 3D | 3 |
| Coéquipiers | Position relative, vélocité 3D et présence par emplacement | 7 × maximum de coéquipiers |
| Adversaires | Position relative, vélocité 3D et présence par emplacement | 7 × maximum d'adversaires |
| Buts | Centre relatif (3), largeur (1) et hauteur (1) de chaque ouverture | 10 |
| Match | Temps et deux scores | 3 |

La dimension est **22 + 7 × (maximum de coéquipiers + maximum d'adversaires)**. Les maxima restent identiques entre les phases ; seul le nombre de joueurs présents change. Leurs valeurs restent à définir.

Chaque but est décrit par `(centre_x, centre_y, centre_z, largeur, hauteur)`, d'abord le but adverse puis le but défendu. Le centre est celui de l'ouverture rectangulaire ; sa coordonnée verticale indique donc aussi la hauteur à laquelle se trouve le but.

Avec 2 emplacements de coéquipiers et 3 d'adversaires (jusqu'au 3v3), l'observation contient **22 + 7 × (2 + 3) = 57 valeurs**, y compris pendant la phase 1v1.

## Actions

Les actions prévues sont le déplacement vers un point, une frappe à intensité variable, le saut et, éventuellement, le tacle. La commande unique `shoot` sert aussi bien à tirer qu'à faire une passe : l'agent apprend à ajuster son placement et l'intensité selon la situation. Aucun seuil d'intensité ne distingue automatiquement une passe d'un tir ; cette distinction dépend des événements de jeu.

L'agent doit apprendre à se déplacer pour se placer avant de tirer ou de faire une passe. La direction de frappe découle de son placement dans Unity, sans commande de direction explicite. La règle reliant ce placement à la direction du ballon reste à définir côté moteur.

Le contrat Python propose le `Dict` suivant. Ce codage est une convention d'interface à valider avec Unity.

| Clé | Espace Gymnasium | Sémantique |
| --- | --- | --- |
| `move_target` | `Box(-1, 1, (3,))` | Cible de déplacement relative normalisée |
| `shoot` | `Box(0, 1, (1,))` | Intensité de la frappe (tir ou passe) ; zéro sans déclenchement |
| `jump` | `Discrete(2)` | Déclenchement du saut |
| `tackle` | `Discrete(2)`, si activé | Déclenchement du tacle optionnel |

Le tacle est activé par `enable_tackle` lors de la construction de l'interface. Les règles d'exécution des commandes restent à définir avec Unity.

## Récompenses prévues

`rewards.py` décrit les composantes de récompense prévues :

- **Objectifs** : catégorie prévue, dont les règles d'attribution restent à définir.
- **Apprentissage** : réduction de la distance au ballon, contact avec le ballon et tir cadré. En équipe, passes et buts précédés de passes, avec d'autres actions collaboratives éventuellement à préciser.
- **Défense** : contact avec le ballon alors qu'il se dirige vers le propre but de l'agent et aurait pénétré dans la zone de but sans interception.
- **Pénalités** : légère perte à chaque frame et pénalité de sortie du terrain.

Aucune valeur de récompense n'est imposée. Les coefficients seront déterminés expérimentalement et feront l'objet d'une étude de sensibilité. La détection des événements, le lieu du calcul entre Unity et Python et les règles d'attribution restent à définir. Les transitions fournies à l'algorithme contiennent la récompense associée.

## Entraînement et évaluation prévus

La progression des récompenses et des statistiques de jeu sera suivie dans **TensorBoard**. Plusieurs logiques d'exploration/exploitation et des techniques d'accélération, dont plusieurs agents jouant simultanément, sont envisagées. L'auto-compétition contre le même modèle ou un autre algorithme RL est également envisagée.

Les algorithmes seront comparés à une **heuristique réactive** orientant le joueur vers le ballon puis vers le but adverse. L'évaluation portera sur la vitesse d'apprentissage, la stabilité et la généralisation : taux de victoire, différence moyenne de buts, buts marqués et encaissés, récompense moyenne, temps d'entraînement et performances face à des adversaires jamais rencontrés ou dans d'autres configurations. Les modèles pourront aussi être affrontés par des joueurs humains. Ces mécanismes ne sont pas encore implémentés.

## Prochaines étapes

1. Définir le protocole, le format des lots et l'export/chargement du réseau avec Unity.
2. Implémenter la collecte dans Unity et la boucle réception, entraînement et renvoi du réseau dans Python ; préciser le raccordement Gymnasium.
3. Définir les maxima de joueurs, la normalisation et les événements de récompense, puis étudier les coefficients.
4. Implémenter les algorithmes à comparer, l'heuristique de référence et le suivi TensorBoard.
5. Évaluer le 1v1, puis augmenter progressivement les effectifs en conservant le format d'observation ; étudier la coopération et la généralisation.
