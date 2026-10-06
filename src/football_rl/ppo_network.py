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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Retourne mean et log_std, chacun de forme (B, 3), sans tirage."""
        features = self.network(x)

        mean = self.mean(features)
        log_std = self.log_std_head(features)
        jump_logit = self.jump_head(features)

        return {"mean": mean, "log_std": log_std, "jump_logit": jump_logit}

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


if __name__ == "__main__":
    # Démonstration numérique exécutée avec python -m football_rl.ppo_network.
    # Aucun optimiseur : ces vérifications n'entraînent pas une politique PPO.
    actor = PPOActorNetwork(observation_size=57)

    # --- 1. Formes ---
    # Ces entrées aléatoires contrôlent les dimensions ; elles ne décrivent
    # pas un état de match cohérent ni nécessairement normalisé dans [-1, 1].
    obs_8 = torch.randn(8, 57)
    sampled_8 = actor.sample_move(obs_8)
    raw_move_8 = sampled_8['raw_move']
    log_prob_8_sampled = sampled_8['log_prob']
    
    # Détacher raw_move pour simuler une donnée fixe venant d'un buffer de collecte
    raw_move_fixe = raw_move_8.detach()
    log_prob_8_evaluated = actor.evaluate_move(obs_8, raw_move_fixe)
    
    print("--- Vérification des formes ---")
    print(f"Log-prob 8 obs (evaluate) : {log_prob_8_evaluated.shape}") # Attendu: (8,)
    
    obs_1 = torch.randn(1, 57)
    raw_move_1 = actor.sample_move(obs_1)['raw_move'].detach()
    log_prob_1_evaluated = actor.evaluate_move(obs_1, raw_move_1)
    print(f"Log-prob 1 obs (evaluate) : {log_prob_1_evaluated.shape}\n") # Attendu: (1,)
    
    # --- 2. Finitude ---
    is_finite = torch.isfinite(log_prob_8_evaluated).all().item()
    print("--- Vérification des valeurs ---")
    print(f"Toutes les log-probabilités sont finies : {is_finite}")
    
    # --- 3. Cohérence numérique entre échantillonnage et évaluation ---
    # allclose utilise des tolérances. Cette comparaison vérifie la cohérence
    # des deux chemins avec les mêmes poids, pas à elle seule la formule.
    sont_egales = torch.allclose(log_prob_8_sampled, log_prob_8_evaluated)
    print(f"Proximité numérique entre sample et evaluate : {sont_egales}\n")
    
    # --- 4. Le point de repère mathématique (-2.7568) ---
    dummy_mean = torch.zeros(1, 3)
    dummy_std = torch.ones(1, 3)
    dummy_raw = torch.zeros(1, 3)
    
    dist_test = torch.distributions.Normal(dummy_mean, dummy_std)
    lp_raw_test = dist_test.log_prob(dummy_raw)
    # Pour u = 0, la correction est nulle : on retrouve -3/2 * log(2*pi).
    corr_test = 2 * (math.log(2) - dummy_raw - F.softplus(-2 * dummy_raw))
    lp_total_test = (lp_raw_test - corr_test).sum(dim=-1)
    print(f"Sanity check sur N(0,1) en (0,0,0) : {lp_total_test.item():.4f}\n")
    
    # --- 5. Jumps ---
    sampled_jump_8 = actor.sample_jump(obs_8)
    jump_target_8 = sampled_jump_8["jump_target"]
    jump_log_prob_8_sampled = sampled_jump_8["log_prob"]
    jump_log_prob_8_evaluated = actor.evaluate_jump(
        obs_8, jump_target_8.detach()
    )

    print("--- Vérification des jumps ---")
    print(f"Cible de jump 8 obs : {jump_target_8.shape}")  # Attendu: (8,)
    print(
        "Valeurs de jump valides (0 ou 1) : "
        f"{bool(torch.all((jump_target_8 == 0) | (jump_target_8 == 1)))}"
    )
    print(
        "Log-prob 8 jumps (evaluate) : "
        f"{jump_log_prob_8_evaluated.shape}"
    )  # Attendu: (8,)
    print(
        "Log-probabilités de jump finies : "
        f"{bool(torch.isfinite(jump_log_prob_8_evaluated).all())}"
    )
    print(
        "Proximité numérique entre sample et evaluate (jump) : "
        f"{torch.allclose(jump_log_prob_8_sampled, jump_log_prob_8_evaluated)}\n"
    )

    obs_jump_1 = torch.randn(1, 57)
    jump_target_1 = actor.sample_jump(obs_jump_1)["jump_target"].detach()
    jump_log_prob_1_evaluated = actor.evaluate_jump(obs_jump_1, jump_target_1)
    print(
        "Log-prob 1 jump (evaluate) : "
        f"{jump_log_prob_1_evaluated.shape}\n"
    )  # Attendu: (1,)

    # --- 6. Rétropropagation ---
    # Objectif de diagnostic uniquement : vérifie que les deux têtes reçoivent
    # des gradients. backward() les calcule sans modifier les poids du réseau.
    loss = -(
        log_prob_8_evaluated.mean() + jump_log_prob_8_evaluated.mean()
    )
    loss.backward()
    
    grad_mean = actor.mean.weight.grad
    grad_log_std = actor.log_std_head.weight.grad
    grad_jump = actor.jump_head.weight.grad
    
    print("--- Vérification des gradients ---")
    print(f"Gradient présent (mean)    : {grad_mean is not None}")
    print(f"Gradient présent (log_std_head) : {grad_log_std is not None}")
    print(f"Gradient présent (jump_head) : {grad_jump is not None}")
