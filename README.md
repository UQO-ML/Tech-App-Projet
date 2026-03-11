# Tech-App-Devoir-II — INF 6243 (Hiver 2026)

**Classification et apprentissage automatique**  
Département d’Informatique et d’Ingénierie — UQO

---

## Objectifs

- Analyse exploratoire et prétraitement des données
- Implémentation d’au moins 4 algorithmes de classification
- Évaluation (accuracy, précision, rappel, F1, matrice de confusion, validation croisée)
- Visualisation des résultats et comparaison des modèles

---

## Structure du projet

```
Tech-App-Devoir-II/
├── README.md
├── requirements.txt
├── main.py                 # Point d’entrée (lance Code/main.py)
├── Code/
│   ├── main.py             # Pipeline : EDA, préparation, entraînement, évaluation
│   ├── preprocessing.py    # Nettoyage, encodage, split train/val/test
│   ├── models.py           # Définition et entraînement des classificateurs
│   └── utils.py            # Métriques, visualisations, helpers
├── Data/
│   ├── lien_vers_dataset.txt   # URL(s) du dataset
│   └── (fichiers de données)   # Optionnel si trop volumineux
├── Rapport_INF6243_NomEtudiants.pdf
└── Presentation_INF6243_NomEtudiants.pptx
```

Chaque script dans `Code/` est documenté en en-tête avec son rôle et sa structure (sections, fonctions à implémenter). Les commentaires à l’intérieur des fichiers décrivent en détail le rôle de chaque section, les entrées/sorties des fonctions et, pour l’apprentissage profond, l’usage du device (CUDA en priorité, repli sur CPU).

---

## Environnement (venv, Linux & Windows)

### Politique de calcul : CUDA en priorité, repli sur CPU

Le projet est conçu pour **utiliser le GPU (CUDA) en priorité** dès qu’il est disponible (driver NVIDIA + toolkit CUDA + build PyTorch/TensorFlow compatible). Si CUDA n’est pas disponible (pas de GPU, driver manquant, ou librairie installée en version CPU uniquement), **le code bascule automatiquement sur le CPU** sans erreur : l’exécution reste possible, seule la vitesse d’entraînement est réduite.

- **Où c’est géré** : au démarrage du pipeline (ou au premier usage d’un modèle GPU), une fonction dédiée (p.ex. `get_device()` dans `Code/utils.py` ou `Code/models.py`) teste la disponibilité de CUDA ; elle retourne `cuda` si possible, sinon `cpu`. Tous les tenseurs et modèles (PyTorch/TensorFlow) sont ensuite créés ou déplacés sur ce device.
- **Scikit-learn** : les classificateurs classiques (KNN, Random Forest, SVM, etc.) s’exécutent sur CPU ; seuls les réseaux de neurones (PyTorch/TensorFlow) profitent du GPU. La politique CUDA-first s’applique donc surtout à l’apprentissage profond.
- **Vérification** : au lancement, vous pouvez afficher le device choisi (p.ex. `Using device: cuda` ou `Using device: cpu`) pour confirmer le comportement sur votre machine.

### Prérequis

- **Python** : 3.10+ recommandé (3.11 ou 3.12 supportés).
- **CUDA** (recommandé pour accélérer les réseaux de neurones) : driver NVIDIA à jour + toolkit CUDA (p.ex. 11.8 ou 12.x) et, pour PyTorch, une build « cu118 » ou « cu121 » selon votre version. Sans GPU, le projet fonctionne entièrement sur CPU.

### Création du venv (Linux)

```bash
cd Tech-App-Devoir-II
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Création du venv (Windows)

**PowerShell :**

```powershell
cd Tech-App-Devoir-II
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Invite de commandes :**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Installation de CUDA (priorité GPU, repli CPU)

Pour que le code utilise le GPU en priorité, il faut une build PyTorch (ou TensorFlow) compilée pour CUDA. Si vous installez la version CPU uniquement, le code détectera l’absence de CUDA et utilisera le CPU sans plantage.

- **PyTorch avec CUDA (recommandé)**  
  Après `pip install -r requirements.txt`, installer la build GPU correspondant à votre version de CUDA (voir [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)) :
  - **Linux, CUDA 12.1** :  
    `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
  - **Linux, CUDA 11.8** :  
    `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
  - **Windows** : même principe ; choisir `cu118` ou `cu121` selon le toolkit installé.
  Une fois installé, le code qui appelle `torch.cuda.is_available()` obtiendra `True` si le driver et le toolkit sont corrects, et le device sera choisi en priorité comme `cuda`, sinon `cpu`.

- **TensorFlow avec GPU**  
  `pip install tensorflow` suffit souvent : TensorFlow détecte automatiquement le GPU si CUDA et cuDNN sont présents. En l’absence de GPU, il utilise le CPU.

- **Comportement en cas d’impossibilité**  
  Si aucun GPU n’est détecté (machine sans NVIDIA, driver manquant, ou build CPU-only), la fonction de sélection du device retourne `cpu` et tout l’entraînement/inférence se fait sur CPU. Aucune modification de code n’est requise pour faire fonctionner le projet sans CUDA.

---

## Lancer le projet

1. Activer le venv : `source .venv/bin/activate` (Linux/macOS) ou `.venv\Scripts\Activate.ps1` (Windows PowerShell).
2. Renseigner l’URL du dataset dans `Data/lien_vers_dataset.txt` et, si besoin, télécharger les données dans `Data/`.
3. Depuis la racine du projet lancer le pipeline :
   ```bash
   python main.py
   ```
   (ou `python Code/main.py` ; les deux exécutent le même pipeline dans `Code/main.py`.)

Au premier lancement, si vous avez implémenté la sélection du device, un message du type `Using device: cuda` ou `Using device: cpu` indiquera si le GPU est utilisé ou si le repli sur CPU est actif. Les scripts `Code/main.py`, `preprocessing.py`, `models.py` et `utils.py` contiennent des commentaires détaillés sur la structure à implémenter (chemins, chargement, prétraitement, modèles, métriques, visualisations, et utilisation du device pour le deep learning).

---

## Soumission (Moodle)

- Fichier `.zip` nommé : `INF6243_Projet_NomEtudiants.zip`
- Contenu : rapport PDF, présentation PowerPoint, dossier `Code/`, éventuellement `Data/` ou seulement `lien_vers_dataset.txt` si le dataset est trop volumineux.

---

## Licence

Voir [LICENSE](LICENSE).
