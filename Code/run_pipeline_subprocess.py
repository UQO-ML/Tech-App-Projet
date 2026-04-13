"""Runner CLI: exécute un run pipeline dans un process isolé."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from main import run_pipeline


def _parse_args() -> argparse.Namespace:
    """Parse les arguments CLI du runner subprocess."""
    parser = argparse.ArgumentParser(description="Execute run_pipeline in isolated subprocess.")
    parser.add_argument("--run-name", required=True, help="Nom du run.")
    parser.add_argument("--config-json", required=True, help="Configuration JSON du run.")
    parser.add_argument("--result-path", required=True, help="Chemin JSON de sortie du résultat subprocess.")
    return parser.parse_args()


def _save_payload(payload: dict[str, Any], result_path: Path) -> None:
    """Écrit le payload de résultat sur disque."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main() -> None:
    """Point d'entrée CLI du runner subprocess."""
    args = _parse_args()
    run_name = args.run_name
    result_path = Path(args.result_path)

    try:
        run_config = json.loads(args.config_json)
        artifacts = run_pipeline(**run_config)
        payload = {
            "status": "ok",
            "run_name": run_name,
            "artifacts": artifacts,
        }
        _save_payload(payload, result_path)
    except Exception as exc:
        payload = {
            "status": "error",
            "run_name": run_name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _save_payload(payload, result_path)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
