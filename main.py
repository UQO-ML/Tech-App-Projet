#!/usr/bin/env python3
"""
Point d’entrée unique pour Tech-App-Devoir-II (INF 6243).

Lance le pipeline défini dans Code/main.py (chargement → exploration
→ prétraitement → entraînement → évaluation → visualisations).

Usage (depuis la racine du projet) :
  python main.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# Exécuter le pipeline (main.py dans Code/ importe preprocessing, models, utils)
from main import main

if __name__ == "__main__":
    main()
