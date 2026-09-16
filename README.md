# Football 3D — Reinforcement Learning Backend

Backend Python dédié à l'entraînement, avec PyTorch, d'un agent d'apprentissage par renforcement pour un jeu de football 3D développé sous Unity.

Le moteur Unity est responsable de toute la simulation : physique, collisions, règles de match, rendu et progression temporelle. Ce dépôt n'implémente donc **ni simulateur, ni physique, ni environnement factice**. Il définit uniquement les contrats Python qui seront reliés au moteur par un pont TCP/UDP.

Le projet commence en **1v1** et est amené à évoluer vers des équipes et du multi-agents.

## Architecture

```text
.
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
└── src/
    └── football_rl/
        ├── __init__.py
        ├── agent.py              # Interface de l'agent RL
        ├── env.py                # Contrat Gymnasium avec Unity
        ├── network.py            # Interfaces des réseaux PyTorch
        ├── rewards.py            # Interface de calcul des récompenses
        ├── train.py              # Point d'entrée d'entraînement
        ├── types.py              # Types partagés des observations/actions
```

## Installation

Prérequis : Python 3.12 ou plus récent et `pip`.

Créer puis activer un environnement virtuel :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances du projet :

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sous macOS/Linux, l'activation devient :

```bash
source .venv/bin/activate
```

## Contrat Unity ↔ Python

Le futur pont réseau transmettra à chaque frame la vérité terrain issue de Unity et recevra une action de l'agent. `Football3DEnv` hérite de `gymnasium.Env`.

L'enchaînement visé est le suivant :

```text
Unity (simulation 3D) → pont TCP/UDP → Football3DEnv → agent PyTorch
Unity (simulation 3D) ← pont TCP/UDP ← Football3DEnv ← action de l'agent
```

## Observations

L'environnement est entièrement observable. Chaque valeur continue envoyée au réseau doit être normalisée dans l'intervalle **[-1, 1]**, avec des bornes définies à partir des dimensions et vitesses maximales du jeu Unity. Les indicateurs binaires et les scores devront eux aussi être encodés dans cet intervalle.

Le repère est **égocentré** : les positions et vecteurs de monde sont exprimés par rapport à la position et à l'orientation de l'agent contrôlé. Ainsi, un ballon « devant » l'agent possède la même représentation quelle que soit l'extrémité du terrain où il se trouve.

En 1v1, l'espace d'observation est un `gymnasium.spaces.Box` de forme `(24,)`, `float32`, borné à `[-1, 1]` :

Ce choix garde l'entrée du réseau simple et stable :

- **`Box`** décrit un vecteur de valeurs continues bornées. C'est adapté aux positions, vitesses, temps et scores normalisés, puis directement exploitable par un réseau PyTorch.
- **`(24,)`** correspond exactement aux 24 caractéristiques du scénario 1v1 détaillées ci-dessous. Une taille fixe est nécessaire pour que le réseau reçoive toujours le même format.

| Groupe | Valeurs | Dimensions |
| --- | --- | ---: |
| Ballon | position relative, vélocité 3D | 6 |
| Agent | vélocité 3D | 3 |
| Adversaire | position relative, vélocité 3D | 6 |
| Buts | position relative du but adverse et du but défendu | 6 |
| Match | temps restant, score de l'agent, score adverse | 3 |
| **Total** |  | **24** |

En mode équipe, chaque joueur supplémentaire occupera 6 valeurs (position relative 3D + vélocité 3D). Le réseau de neurones exigeant une taille d'entrée fixe, nous fixerons un nombre maximum de joueurs et utiliserons un système de masque pour ignorer les emplacements vides.

## Actions

L'action est une `gymnasium.spaces.Dict` hybride :

Le `Dict` permet de regrouper les deux natures d'actions sans les confondre :

- **`Box(-1, 1, (3,))`** pour `move_target` : les trois coordonnées continues de la cible de déplacement, exprimées dans le repère égocentré et normalisées. Chaque composante vaut entre `-1` et `1`.
- **`Discrete(2)`** pour `shoot`, `pass_action`, `jump` et `tackle` : une commande binaire, où `0` signifie « ne pas déclencher » et `1` signifie « déclencher ».

| Clé | Espace | Sémantique |
| --- | --- | --- |
| `move_target` | `Box(-1, 1, (3,))` | Cible de déplacement relative normalisée ; si le ballon est devant l'agent, le déplacement le pousse. |
| `shoot` | `Discrete(2)` | Déclenche un tir. |
| `pass_action` | `Discrete(2)` | Déclenche une passe. |
| `jump` | `Discrete(2)` | Déclenche un saut. |
| `tackle` | `Discrete(2)` | Déclenche un tacle. |

L'interprétation exacte (durée, puissance, priorité lorsque plusieurs commandes sont actives et validation par les règles) appartiendra à Unity et au protocole partagé.

## Récompenses prévues

La composition des récompenses sera implémentée dans `rewards.py`, à partir des événements fiables transmis par Unity.

- **Objectifs** : `+1` pour un but marqué ; `-1` pour un but encaissé.
- **Progression** : rapprochement du ballon, premier contact, possession dirigée vers le camp adverse et tir cadré.
- **Défense** : récompense importante pour l'interception d'un tir dangereux, notamment par saut ou tacle.
- **Pénalités** : faible coût temporel à chaque frame et pénalité lors d'une sortie de terrain.

Les coefficients, plafonds et règles d'attribution seront versionnés avec la configuration de l'expérience afin de conserver des entraînements reproductibles et d'éviter de modifier le signal de récompense silencieusement
