"""Rend le paquet du dépôt Unity importable dans l'environnement virtuel actif.

À lancer après l'installation du projet Python. Le chemin est enregistré dans
site-packages ; le code du pont continue d'être chargé depuis le dépôt Unity.
"""

from __future__ import annotations

import argparse
import locale
from pathlib import Path
import sys
import sysconfig


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    default_path = Path(__file__).resolve().parents[2] / "MTI881_IA_Foot_Unity" / "python"
    parser.add_argument(
        "--unity-python", type=Path, default=default_path,
        help="dossier contenant le paquet iafoot (par défaut : dépôt Unity voisin)",
    )
    args = parser.parse_args()

    if sys.prefix == sys.base_prefix:
        parser.error("Utilise le Python de .venv pour configurer uniquement cet environnement.")

    bridge_path = args.unity_python.expanduser().resolve()
    required = ("__init__.py", "models.py", "protocol.py", "server.py")
    missing = [name for name in required if not (bridge_path / "iafoot" / name).is_file()]
    if missing:
        parser.error(
            f"Paquet iafoot incomplet dans {bridge_path} : {', '.join(missing)}. "
            "Initialise le sous-module Unity ou indique --unity-python."
        )

    # purelib désigne le site-packages de l'interpréteur utilisé, sur Windows
    # comme sur Linux/macOS. Aucun chemin propre à un utilisateur n'est versionné.
    site_packages = Path(sysconfig.get_path("purelib")).resolve()
    if not site_packages.is_relative_to(Path(sys.prefix).resolve()):
        parser.error(f"Le site-packages est hors de l'environnement virtuel : {site_packages}")
    site_packages.mkdir(parents=True, exist_ok=True)
    link = site_packages / "iafoot_bridge.pth"
    # Python 3.12 lit les .pth dans l'encodage local ; 3.13+ essaie UTF-8.
    encoding = "utf-8" if sys.version_info >= (3, 13) else locale.getpreferredencoding(False)
    content = f"{bridge_path}\n"
    unchanged = link.is_file() and link.read_text(encoding=encoding) == content
    if not unchanged:
        link.write_text(content, encoding=encoding)

    print(f"Interpréteur : {sys.executable}")
    print(f"Pont Unity   : {bridge_path}")
    print(f"Lien {'déjà configuré' if unchanged else 'créé'} : {link}")
    print("Le lien sera utilisé par les prochains processus Python de cet environnement.")


if __name__ == "__main__":
    main()
