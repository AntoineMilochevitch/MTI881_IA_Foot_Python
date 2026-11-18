# DQN — entraînement et évaluation dans Unity

Double DQN (Polyak τ = 0.005) branché sur le pont `iafoot`. Le réseau tourne dans
Unity avec le décodeur `discrete` : il donne une Q-valeur par ligne d'une **table
d'actions**, et Unity choisit la ligne (ε-greedy pendant l'entraînement, argmax en
évaluation). Python reçoit les batchs, entraîne et renvoie le réseau.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `src/football_rl/dqn.py` | Cœur du DQN (`DQNConfig`, `DQNLearner`), sans dépendance à Unity |
| `src/football_rl/dqn_actions.py` | Tables d'actions (`extended` : 28 lignes, `foot` : 20) |
| `src/football_rl/dqn_unity.py` | Serveur d'entraînement / d'évaluation branché sur `iafoot` |
| `scripts/dqn_cartpole.py` | Validation du même cœur sur CartPole-v1 |
| `tests/test_dqn_unity.py` | Tests sans Unity (fausse connexion, export vérifié comme Unity) |

## 0. Prérequis (une fois)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python scripts/link_unity_bridge.py      # rend iafoot importable
python -c "import iafoot, football_rl.dqn_unity; print('OK')"
```

## 1. Tester sans Unity (1 minute)

```powershell
python tests/test_dqn.py
python tests/test_dqn_unity.py
```

`test_dqn_unity.py` vérifie notamment que le modèle exporté calcule les mêmes
Q-valeurs que PyTorch avec l'implémentation de référence d'Unity, et qu'un
problème simple est appris de bout en bout à travers le pont.

Validation de l'algorithme sur CartPole (≈ 6 min par graine) :

```powershell
python scripts/dqn_cartpole.py --double --seeds 0 1 2
```

## 2. Entraîner dans Unity

1. Unity : ouvrir la scène, **IA Foot > Entraînement > Préparer la scène**.
   Dans `TrainingBridge`, laisser **Include Next Observations** coché (obligatoire
   pour le DQN). Activer **Run In Background** (Project Settings > Player) pour que
   le jeu continue hors focus.
2. Python :
   ```powershell
   python -m football_rl.dqn_unity --time-scale 10
   ```
3. **Play** dans Unity (l'ordre n'a pas d'importance, Unity se reconnecte).
4. Suivi : `tensorboard --logdir runs` puis http://localhost:6006.
5. `Ctrl+C` arrête et écrit `checkpoints/<run>/latest.pt`. Un checkpoint
   numéroté est aussi écrit toutes les 50 000 transitions.

Ce qu'on doit voir dans la console :

```text
Unity a installé le DQN (v1) : discrete (28 actions, epsilon_greedy).
batch n°0 | +512 transitions | total 512 | ε 0.998 | remplissage du tampon 512/5000 | ...
batch n°9 | +512 transitions | total 5120 | ε 0.976 | 128 pas de gradient en 0.5 s, perte 0.0051, Q moyen +0.07 | ...
```

Avec « Préparer la scène », **tous** les joueurs partagent le même DQN (self-play) :
les transitions des deux équipes servent à l'entraînement.

Options utiles :

| Option | Défaut | Effet |
| --- | --- | --- |
| `--time-scale` | inchangé | Vitesse de simulation |
| `--max-transitions` | aucun | Arrête l'entraînement après N transitions (budget fixe pour comparer les algorithmes) |
| `--seed` | 0 | Graine Python (initialisation, tampon). La graine Unity se règle dans `FootAgent` |
| `--train-freq` | 4 | Transitions par pas de gradient. Si « pas de gradient en X s » dépasse le temps entre deux batchs, l'augmenter |
| `--eps-decay` / `--eps-end` | 200 000 / 0.05 | Décroissance de l'exploration, en transitions |
| `--action-table` | `extended` | `foot` = table de démonstration du pont (20 actions) |
| `--resume` | — | Reprend un checkpoint (réglages DQN et table du checkpoint) |
| `--no-double` | — | DQN classique |
| `--verbose` | — | Affiche chaque message réseau |

`python -m football_rl.dqn_unity --help` liste tout.

## 3. Évaluer un modèle

```powershell
python -m football_rl.dqn_unity --evaluate --resume checkpoints/<run>/latest.pt --evaluation-episodes 200 --evaluation-output resultats/eval.json
```

Le réseau est envoyé en **argmax** (sans exploration) et rien n'est entraîné. Ne
comptent que les épisodes joués **entièrement** avec ce réseau et terminés par un
but ou par la limite de temps ; les manches interrompues sont exclues. Le bilan
(buts marqués / encaissés, temps écoulé, différence de buts, retour moyen,
détail par équipe) s'affiche et peut être écrit en JSON.

**En self-play, victoires et défaites se compensent** : les deux équipes sont le
même réseau. Pour mesurer le niveau, évaluer contre le bot scripté :

| Composant | Joueur(s) DQN | Adversaire(s) |
| --- | --- | --- |
| `FootAgent` | activé, *Record Experience* coché | **désactivé** |
| `SimpleBotInput` | désactivé | activé |
| `HumanPlayerInput` | désactivé | désactivé |

(Même configuration que pour PPO, voir `README_UNITY.md` sur la branche `ppo`.)
Seuls les joueurs DQN envoient des épisodes : le bilan est alors de leur point de vue.

## Points d'attention

- **Le tampon de rejeu n'est pas sauvegardé** : à la reprise, il se remplit à nouveau
  (`--learning-starts` transitions) avant que l'entraînement reprenne.
- Un checkpoint est lié aux observations de la scène (`Max Teammates`, `Max Opponents`
  des `FootAgent`) : une reprise avec une autre taille est refusée.
- Les transitions jouées avant l'installation du DQN (modèle aléatoire v0,
  `action_index = -1`) sont ignorées et comptées dans `train/ignored_transitions`.
- `actions/index` dans TensorBoard : si une seule action domine, la politique est bloquée.
