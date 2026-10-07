"""Conversion de l'acteur en modèle et décodeur du pont iafoot.

Le paquet iafoot provient du dépôt Unity voisin. scripts/link_unity_bridge.py
configure son import dans .venv. L'envoi réseau est réalisé par send_model.
"""

import copy

import torch

from iafoot.models import DecoderExport, ModelExport, from_torch, direct_decoder

from football_rl.ppo_network import PPOActorNetwork


def export_actor(actor: PPOActorNetwork) -> tuple[ModelExport, DecoderExport]:
    """
    Exporte une copie cohérente de l'acteur pour Unity.

    Retourne :
        model_export   : représentation exportée du MLP
        decoder_export : paramètres nécessaires au décodage des actions
    """

    # Une même copie fournit les poids du MLP et les paramètres du décodeur.
    actor_snapshot = copy.deepcopy(actor)

    # L'exporteur calcule son exemple de contrôle sur CPU en float32.
    # eval() concerne le MLP ; l'exploration reste activée dans le décodeur.
    actor_snapshot = actor_snapshot.cpu().float()
    actor_snapshot.eval()

    # 3. Export du MLP uniquement
    model_export = from_torch(
        actor_snapshot.network,
        check=True,
    )

    # 4. Récupération des mêmes log_std que ceux associés
    #    aux poids que nous venons d'exporter.
    bounded_log_std = torch.clamp(
        actor_snapshot.log_std,
        min=-5.0,
        max=2.0,
    )

    # [move_x, move_z, shoot_power, shoot_curve]
    continuous_log_std = bounded_log_std.detach().cpu().tolist()

    # Le décodeur exige 1 ou 7 valeurs. Les valeurs des boutons sont ignorées.
    decoder_log_std = [
        continuous_log_std[0],  # move_x
        continuous_log_std[1],  # move_z
        0.0,                    # shoot : Bernoulli
        continuous_log_std[2],  # shoot_power
        continuous_log_std[3],  # shoot_curve
        0.0,                    # tackle : Bernoulli
        0.0,                    # jump : Bernoulli
    ]

    decoder_export = direct_decoder(
        log_std=decoder_log_std,
        deterministic=False,
    )

    return model_export, decoder_export
