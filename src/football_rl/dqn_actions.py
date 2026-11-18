"""Tables d'actions discrètes pour le DQN dans Unity.

Le DQN sort une Q-valeur par ligne de table ; le décodeur ``discrete`` d'Unity
choisit une ligne (argmax ou ε-greedy) et l'applique telle quelle. Une ligne est
un vecteur d'action complet, dans l'ordre de ``FootActions`` (FootSpaces.cs) :

    move_x, move_z, shoot, shoot_power, shoot_curve, tackle, jump

- move_x / move_z dans [-1, 1], repère d'équipe (+x = vers le but adverse) ;
- shoot, tackle, jump : boutons (> 0.5 = appuyé) ;
- shoot_power dans [-1, 1], ramené par Unity à une intensité (p + 1) / 2 :
  1 → frappe maximale, 0 → moitié, -0.5 → quart.

La table fait partie du modèle : elle est sauvegardée dans les checkpoints et
une reprise avec une autre table est refusée (les indices n'auraient plus de sens).
"""

from __future__ import annotations

import math

import numpy as np

ACTION_COLUMNS = ("move_x", "move_z", "shoot", "shoot_power", "shoot_curve", "tackle", "jump")
ACTION_TYPES = ("continuous", "continuous", "binary", "continuous", "continuous", "binary", "binary")


def _directions(count: int = 8) -> list[tuple[float, float]]:
    return [(math.cos(k * 2 * math.pi / count), math.sin(k * 2 * math.pi / count)) for k in range(count)]


def foot_table() -> np.ndarray:
    """Table de démonstration fournie par le pont (``iafoot.models.foot_action_table``), 20 lignes.

    Immobile, 8 directions, 8 directions + tir fort, tir faible vers l'avant,
    tacle vers l'avant, saut sur place.
    """
    rows = [[0, 0, 0, 0, 0, 0, 0]]
    rows += [[x, z, 0, 0, 0, 0, 0] for x, z in _directions()]
    rows += [[x, z, 1, 1, 0, 0, 0] for x, z in _directions()]
    rows += [[1, 0, 1, -0.5, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]]
    return np.asarray(rows, dtype=np.float32)


def extended_table() -> np.ndarray:
    """Table par défaut, 28 lignes : ajoute une frappe moyenne dans les 8 directions (passes).

    Immobile, 8 directions, 8 directions + tir fort (intensité 1),
    8 directions + tir moyen (intensité 0.5), tacle vers l'avant,
    saut sur place, saut vers l'avant.
    """
    rows = [[0, 0, 0, 0, 0, 0, 0]]
    rows += [[x, z, 0, 0, 0, 0, 0] for x, z in _directions()]
    rows += [[x, z, 1, 1, 0, 0, 0] for x, z in _directions()]
    rows += [[x, z, 1, 0, 0, 0, 0] for x, z in _directions()]
    rows += [[1, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1]]
    return np.asarray(rows, dtype=np.float32)


ACTION_TABLES = {"extended": extended_table, "foot": foot_table}


def build_action_table(name: str) -> np.ndarray:
    try:
        table = ACTION_TABLES[name]()
    except KeyError:
        raise ValueError(f"Table d'actions inconnue : {name!r} (choix : {', '.join(ACTION_TABLES)}).") from None
    assert table.ndim == 2 and table.shape[1] == len(ACTION_COLUMNS)
    return table


def describe_action(row: np.ndarray) -> str:
    """Libellé court d'une ligne, pour les logs (ex. « →↗ tir 1.0 »)."""
    move_x, move_z, shoot, power, _curve, tackle, jump = (float(v) for v in row)
    parts = []
    if abs(move_x) > 1e-6 or abs(move_z) > 1e-6:
        angle = math.degrees(math.atan2(move_z, move_x)) % 360
        parts.append(f"dir {angle:.0f}°")
    else:
        parts.append("immobile")
    if shoot > 0.5:
        parts.append(f"tir {(power + 1) / 2:.2f}")
    if tackle > 0.5:
        parts.append("tacle")
    if jump > 0.5:
        parts.append("saut")
    return " + ".join(parts)
