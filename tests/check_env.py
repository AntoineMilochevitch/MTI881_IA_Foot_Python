from gymnasium.utils.env_checker import check_env

from football_rl.env import Football3DEnv
from football_rl.types import ObservationLayout


def test_football3d_env() -> None:
    layout = ObservationLayout(max_teammates=2, max_opponents=3)
    env = Football3DEnv(layout, enable_tackle=False)
    check_env(env, skip_render_check=True)


if __name__ == "__main__":
    test_football3d_env()
    print("OK")