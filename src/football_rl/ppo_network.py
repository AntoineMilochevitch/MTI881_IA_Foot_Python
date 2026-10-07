"""Réseaux de base pour PPO : critique et politique de déplacement.

Les observations sont des lots de forme (B, D), où B est le nombre de
situations et D la dimension définie par ObservationLayout. Le critique
prédit un retour par situation ; l'acteur décrit trois distributions pour
la cible de déplacement. Frappe, saut, tacle et entraînement PPO restent
à construire. Voir README_PPO.md à la racine pour le fonctionnement prévu.

Ce module ne normalise pas les données Unity et ne simule pas le match.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from football_rl.network import FootballNetwork


class PPOValueNetwork(FootballNetwork):
    """Estime V(s), le retour actualisé attendu sous la politique courante.

    La sortie est un réel, pas une probabilité de victoire. Ce réseau sera
    entraîné en Python à partir de cibles construites avec les trajectoires.
    """

    def __init__(self, observation_size: int) -> None:
        super().__init__(observation_size)

        # Les deux couches cachées apprennent des combinaisons d'observations.
        # La sortie linéaire permet des retours négatifs ou supérieurs à 1.
        self.network = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Transforme (B, D) en une valeur par observation, de forme (B,)."""
        value = self.network(x)

        # (B, 1) -> (B,) : conserve l'axe du lot, même lorsque B vaut 1.
        value = value.squeeze(-1)
        return {"value": value}


class PPOActorNetwork(FootballNetwork):
    """Décrit la politique de déplacement par des gaussiennes transformées.

    Pour chaque observation, trois moyennes et trois log-écarts-types
    définissent des coordonnées indépendantes conditionnellement à l'état.
    Une action est obtenue en échantillonnant, puis en appliquant tanh.
    Les autres composantes de FootballAction ne sont pas encore produites.
    """

    def __init__(self, observation_size: int) -> None:
        super().__init__(observation_size)

        # Tronc partagé par les deux têtes de l'acteur. Le critique possède
        # son propre réseau et ses propres paramètres.
        self.network = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        # Les têtes sont linéaires : mu et log(sigma) ne sont pas bornés ici.
        # mu est la moyenne avant tanh, pas la moyenne des actions finales.
        self.mean = nn.Linear(128, 3)
        self.log_std_head = nn.Linear(128, 3)

        self.jump_head = nn.Linear(128, 1)

        self.tackle_head = nn.Linear(128, 1)

        self.shoot_trigger_head = nn.Linear(128, 1)
        self.shoot_alpha_head = nn.Linear(128, 1)
        self.shoot_beta_head = nn.Linear(128, 1)


    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Retourne mean et log_std, chacun de forme (B, 3), sans tirage."""
        features = self.network(x)

        mean = self.mean(features)
        log_std = self.log_std_head(features)
        jump_logit = self.jump_head(features)
        tackle_logit = self.tackle_head(features)
        shoot_trigger_logit = self.shoot_trigger_head(features)
        shoot_alpha_raw = self.shoot_alpha_head(features)
        shoot_beta_raw = self.shoot_beta_head(features)

        return {
            "mean": mean,
            "log_std": log_std,
            "jump_logit": jump_logit,
            "tackle_logit": tackle_logit,
            "shoot_trigger_logit": shoot_trigger_logit,
            "shoot_alpha_raw": shoot_alpha_raw,
            "shoot_beta_raw": shoot_beta_raw
        }

    def sample_move(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Choisit des déplacements et calcule leur log-densité.

        Retourne raw_move (B, 3), move_target (B, 3) et log_prob (B,).
        move_target est la cible normalisée destinée à Unity. raw_move
        servira à réévaluer la même action pendant l'apprentissage.

        sample() détache le tirage, mais log_prob reste différentiable si
        Autograd est actif. La future collecte utilisera torch.no_grad()
        ou détachera les données avant de les conserver dans le buffer.
        """
        outputs = self.forward(x)
        mean = outputs["mean"]
        log_std = outputs["log_std"]

        # Bornes initiales pour limiter les écarts-types extrêmes. La même
        # convention doit être utilisée lors de l'évaluation et dans Unity.
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)

        # Normal attend un écart-type, et non une variance. sample() fournit
        # une donnée fixe : PPO apprendra via sa log-probabilité réévaluée.
        normal_dist = torch.distributions.Normal(mean, std)
        raw_move = normal_dist.sample()

        # Chaque coordonnée de la cible doit respecter les bornes [-1, 1].
        move_target = torch.tanh(raw_move)

        # Le changement de variable a = tanh(u) modifie la densité :
        # log pi(a|s) = log p(u|s) - log(1 - tanh(u)^2).
        # Cette écriture avec softplus reste stable près de la saturation.
        log_prob_raw = normal_dist.log_prob(raw_move)
        correction = 2 * (math.log(2) - raw_move - F.softplus(-2 * raw_move))
        log_prob = log_prob_raw - correction
        # Somme les trois coordonnées, en conservant une valeur par observation.
        log_prob = log_prob.sum(dim=-1)

        return {"raw_move": raw_move,
                "move_target": move_target,
                "log_prob": log_prob}

    def evaluate_move(self, x: torch.Tensor, raw_move: torch.Tensor) -> torch.Tensor:
        """Recalcule la log-densité d'un déplacement enregistré, sans tirage.

        x : observations (B, D) ; raw_move : données fixes (B, 3), détachées
        du graphe de collecte. Retour : log-densités (B,) reliées aux poids
        actuels de l'acteur. Aucune entropie n'est calculée ici.

        PPO comparera ce résultat à la log-densité stockée lors de la
        collecte, pour la même observation et la même action.
        """
        outputs = self.forward(x)
        mean = outputs["mean"]
        log_std = outputs["log_std"]

        # Reproduit exactement les paramètres de distribution de sample_move.
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)

        normal_dist = torch.distributions.Normal(mean, std)

        # L'action enregistrée reste fixe ; les gradients portent sur les
        # paramètres actuels de la gaussienne. La correction utilise u pour
        # éviter de reconstruire l'inverse de tanh près de -1 ou de 1.
        log_prob_raw = normal_dist.log_prob(raw_move)
        correction = 2 * (math.log(2) - raw_move - F.softplus(-2 * raw_move))
        log_prob = (log_prob_raw - correction).sum(dim=-1)

        return log_prob

    def sample_jump(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Échantillonne un saut binaire et calcule sa log-densité.

        Retourne jump_target (B,) et log_prob (B,). jump_target est la
        cible destinée à Unity. La future collecte utilisera torch.no_grad()
        ou détachera les données avant de les conserver dans le buffer.
        """
        outputs = self.forward(x)
        jump_logit = outputs["jump_logit"].squeeze(-1)  # (B, 1) -> (B,)

        dist = torch.distributions.Bernoulli(logits=jump_logit)
        jump_target = dist.sample()  # (B,)
        log_prob = dist.log_prob(jump_target)  # (B,)

        return {"jump_target": jump_target, "log_prob": log_prob}

    def evaluate_jump(self, x: torch.Tensor, jump_target: torch.Tensor) -> torch.Tensor:
        """Recalcule la log-densité d'un saut enregistré, sans tirage.

        x : observations (B, D) ; jump_target : données fixes (B,), détachées
        du graphe de collecte. Retour : log-densités (B,) reliées aux poids
        actuels de l'acteur. Aucune entropie n'est calculée ici.

        PPO comparera ce résultat à la log-densité stockée lors de la
        collecte, pour la même observation et le même saut.
        """
        outputs = self.forward(x)
        jump_logit = outputs["jump_logit"].squeeze(-1)  # (B, 1) -> (B,)

        dist = torch.distributions.Bernoulli(logits=jump_logit)
        log_prob = dist.log_prob(jump_target)  # (B,)

        return log_prob

    def sample_tackle(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Échantillonne un tacle binaire et calcule sa log-densité.

        Retourne tackle_target (B,) et log_prob (B,). tackle_target est la
        cible destinée à Unity, valide uniquement si enable_tackle est actif.
        """
        outputs = self.forward(x)
        tackle_logit = outputs["tackle_logit"].squeeze(-1)  # (B, 1) -> (B,)

        dist = torch.distributions.Bernoulli(logits=tackle_logit)
        tackle_target = dist.sample()  # (B,)
        log_prob = dist.log_prob(tackle_target)  # (B,)

        return {"tackle_target": tackle_target, "log_prob": log_prob}

    def evaluate_tackle(self, x: torch.Tensor, tackle_target: torch.Tensor) -> torch.Tensor:
        """Recalcule la log-densité d'un tacle enregistré, sans tirage.

        x : observations (B, D) ; tackle_target : données fixes (B,), détachées
        du graphe de collecte. Retour : log-densités (B,) reliées aux poids
        actuels de l'acteur.
        """
        outputs = self.forward(x)
        tackle_logit = outputs["tackle_logit"].squeeze(-1)  # (B, 1) -> (B,)

        dist = torch.distributions.Bernoulli(logits=tackle_logit)
        log_prob = dist.log_prob(tackle_target)  # (B,)

        return log_prob

    def sample_shoot(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Échantillonne une intensité de frappe, avec zéro possible exactement.

            Modèle en deux étapes ("hurdle") : un déclenchement binaire décide si
            une frappe a lieu, puis une intensité continue dans (0, 1) est tirée
            seulement si déclenché. L'intensité vaut exactement 0 sinon.

            Retourne trigger (B,), intensity (B,), shoot_target (B,) et log_prob (B,).
            """
        outputs = self.forward(x)

        trigger_logit = outputs["shoot_trigger_logit"].squeeze(-1)  # (B, 1) -> (B,)

        # softplus garantit la positivité, +1 évite les distributions instables en début d'entraînement (quand poids proches de 0).
        alpha = F.softplus(outputs["shoot_alpha_raw"]).squeeze(-1) + 1.0
        beta = F.softplus(outputs["shoot_beta_raw"]).squeeze(-1) + 1.0

        # 1. Décide si une frappe a lieu
        trigger_dist = torch.distributions.Bernoulli(logits=trigger_logit)
        trigger = trigger_dist.sample()

        # 2. Intensité de la frappe (éventuellement annulée si trigger 0
        beta_dist = torch.distributions.Beta(alpha, beta)
        intensity = beta_dist.sample()

        shoot_target = trigger * intensity

        trigger_log_prob = trigger_dist.log_prob(trigger)
        intensity_log_prob = beta_dist.log_prob(intensity)
        log_prob = trigger_log_prob + trigger * intensity_log_prob

        return {"trigger": trigger, "intensity": intensity, "shoot_target": shoot_target, "log_prob": log_prob}

    def evaluate_shoot(self, x: torch.Tensor, trigger: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        """Recalcule la log-densité d'un frappe enregistré, sans tirage.

        x : observations (B, D) ; trigger (B,) et intensity (B,) : données
        fixes, détachées du graphe de collecte. Retour : log-densités (B,)
        reliées aux poids actuels de l'acteur.
        """
        outputs = self.forward(x)
        trigger_logit = outputs["shoot_trigger_logit"].squeeze(-1)
        alpha = F.softplus(outputs["shoot_alpha_raw"]).squeeze(-1) + 1.0
        beta = F.softplus(outputs["shoot_beta_raw"]).squeeze(-1) + 1.0

        trigger_dist = torch.distributions.Bernoulli(logits=trigger_logit)
        beta_dist = torch.distributions.Beta(alpha, beta)

        trigger_log_prob = trigger_dist.log_prob(trigger)
        intensity_log_prob = beta_dist.log_prob(intensity)
        log_prob = trigger_log_prob + trigger * intensity_log_prob

        return log_prob

    def sample_action(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Échantillonne un déplacement, un saut, un tacle et une frappe.

        Retourne un dictionnaire avec toutes les cibles et log-probabilités.
        """
        move = self.sample_move(x)
        jump = self.sample_jump(x)
        tackle = self.sample_tackle(x)
        shoot = self.sample_shoot(x)

        sum_log_prob = move["log_prob"] + jump["log_prob"] + tackle["log_prob"] + shoot["log_prob"]

        return {
            "move_target": move["move_target"],
            "jump": jump["jump_target"],
            "tackle": tackle["tackle_target"],
            "shoot": shoot["shoot_target"].unsqueeze(-1),  # (B,) -> (B, 1) pour Unity
            "raw_move": move["raw_move"],
            "trigger": shoot["trigger"],
            "intensity": shoot["intensity"],
            "log_prob": sum_log_prob
        }

    def evaluate_action(self, x: torch.Tensor, raw_move: torch.Tensor, jump: torch.Tensor, tackle: torch.Tensor, trigger: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        """Recalcule la log-densité d'une action enregistrée, sans tirage.

        x : observations (B, D) ; raw_move (B, 3), jump (B,), tackle (B,),
        trigger (B,) et intensity (B,) : données fixes, détachées du graphe
        de collecte. Retour : log-densités (B,) reliées aux poids actuels
        de l'acteur.
        """
        move_log_prob = self.evaluate_move(x, raw_move)
        jump_log_prob = self.evaluate_jump(x, jump)
        tackle_log_prob = self.evaluate_tackle(x, tackle)
        shoot_log_prob = self.evaluate_shoot(x, trigger, intensity)

        sum_log_prob = move_log_prob + jump_log_prob + tackle_log_prob + shoot_log_prob

        return sum_log_prob
