# Projet : Détection de Hate Speech et Langage Offensif

<<<<<<< HEAD
## Vue d'ensemble
## Auteur & Attribution
Ce projet implémente un **système de classification multiclasse** pour détecter automatiquement le hate speech, le langage offensif et les tweets neutres. Le modèle est entraîné et évalué sur le jeu de données [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) disponible sur Kaggle.
=======
Projet de classification de tweets en 3 classes:
- `hate_speech`
- `offensive_language`
- `neither`

## Objectif du dépôt

Le projet compare plusieurs familles de modèles (classiques, GPU cuML, DistilBERT) avec une orchestration multi-runs.
Le flux principal:
1. préparer/nettoyer les données;
2. entraîner plusieurs modèles;
3. évaluer (validation + test + CV);
4. sélectionner un meilleur compromis via un score pondéré;
5. générer des artefacts (figures, rapports, modèle sauvegardé).
>>>>>>> main

### Objectif
Comparer plusieurs algorithmes de classification supervisée et sélectionner le meilleur modèle pour prédire si un tweet appartient à l'une des trois catégories :
- **0 : Hate Speech** (langage de haine)
- **1 : Offensive Language** (langage offensif)
- **2 : Neither** (aucun des deux)

<<<<<<< HEAD
---

## Exécution sur Kaggle

**Important** : Ce notebook est **conçu pour tourner sur Kaggle Notebooks** avec accès aux datasets Kaggle.

### Prérequis Kaggle
- Compte Kaggle actif
- Notebook Kaggle (créer depuis [kaggle.com/notebooks](https://kaggle.com/notebooks))
- Accès au dataset [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

### Étapes d'exécution
1. Créer un nouveau **Kaggle Notebook** (Python)
2. Importer le fichier [Code/hate-speech-model.ipynb](Code/hate-speech-model.ipynb) ou copier-coller le code
3. Vérifier que le dataset est accessible dans `/kaggle/input/`
4. **Exécuter les cellules dans l'ordre** (de 1 à 24)
5. Les résultats s'affichent directement + fichiers d'export générés

### Modification pour exécution locale
Si vous souhaitez exécuter ce notebook **localement** en dehors de Kaggle, modifiez la cellule 4 de chargement des données :

```python
# Actuel (Kaggle)
URL = "/kaggle/input/datasets/mrmorj/hate-speech-and-offensive-language-dataset/labeled_data.csv"

# Local
# Téléchargez d'abord le CSV depuis Kaggle
import pandas as pd
df = pd.read_csv("./Data/labeled_data.csv")  # chemin local
```

---
=======
```text
Tech-App-Devoir-II/
├── main.py                         # point d'entrée recommandé (CLI multi-runs)
├── notebook_principal.ipynb        # workflow interactif notebook
├── requirements.txt
├── Code/
│   ├── main.py                     # pipeline unitaire (run unique)
│   ├── notebook_workflow.py        # orchestration partagée notebook/CLI
│   ├── preprocessing.py
│   ├── models.py
│   ├── run_configs.py              # matrice de runs et profils
│   ├── run_pipeline_subprocess.py  # exécution isolée par subprocess
│   ├── result_interpreter.py
│   ├── report_markdown.py
│   ├── utils.py
│   └── model_zoo/
├── Data/
│   ├── labeled_data.csv
│   └── lien_vers_dataset.txt
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.arch
│   ├── Dockerfile.fedora
│   ├── docker-compose.simple.yml
│   └── docker-compose.pipeline.yml
├── Docs/
└── Outputs/                        # créé automatiquement
```

## Prérequis

### Exécution locale (hors Docker)

- Linux/macOS (ou WSL2 sur Windows) recommandé;
- Python 3.11 recommandé;
- `pip` à jour;
- option GPU/cuML: CUDA + pilotes NVIDIA cohérents avec les libs installées.

Important: le `requirements.txt` mentionne explicitement:

```bash
unset CUDA_PATH
```

à exécuter avant installation si votre environnement exporte `CUDA_PATH` et crée des conflits de résolution.

### Exécution Docker (recommandée pour GPU)

- Docker + plugin Compose;
- pilote NVIDIA compatible;
- NVIDIA Container Toolkit configuré.

Test rapide:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 nvidia-smi
```

## Installation locale

Depuis la racine du projet:
>>>>>>> main

## Structure du Notebook

Le notebook est organisé en **24 cellules** couvrant l'ensemble du pipeline Machine Learning :

### 1. Préparation & Chargement des Données (Cellules 1-5)
- Import des bibliothèques essentielles (pandas, scikit-learn, NLTK)
- Installation automatique des dépendances via pip
- **Chargement du dataset Kaggle** depuis `/kaggle/input/`
- Statistiques descriptives du dataset

### 2. Exploration Données (EDA) (Cellules 6-8)
- **Cellule 6** : Distribution des classes (histogramme + pie chart)
- **Cellule 7** : Distribution des longueurs de tweets et comptage des mots
- **Cellule 8** : Matrice de corrélation (heatmap) des features numériques

### 3. Prétraitement Texte NLP (Cellules 9-10)
- **Cellule 9** : Pipeline de nettoyage robuste
  - Conversion minuscules
  - Suppression URLs, mentions (@user), hashtags
  - Suppression entités HTML
  - Suppression caractères spéciaux
  - Lemmatisation des tokens
  - Suppression des stop words anglais
- **Cellule 10** : Vectorisation **TF-IDF** + **Split stratifié** train/validation/test (70%/15%/15%)

### 4. Sélection & Justification des Modèles (Cellule 11)
Cellule **Markdown** expliquant le choix des 4 algorithmes :
- Logistic Regression (baseline linear)
- Linear SVM (performant sur texte sparse)
- Multinomial Naive Bayes (adapté aux fréquences)
- Random Forest (non-linéaire)

### 5. Tuning d'Hyperparamètres (Cellules 12-13)
- **Cellule 12** : **RandomizedSearchCV** pour optimisation d'hyperparamètres
  - 10 itérations d'exploration de grille stochastique
  - Validation interne 3-fold
  - Métrique : F1-macro (équilibrée pour classes déséquilibrées)
- **Cellule 13** : Classement des modèles tuned sur ensemble de validation

### 6. Évaluation Complète (Cellules 14-18)
- **Cellule 14** : Métriques test complet (Accuracy, Precision, Recall, F1-macro pour chaque modèle)
- **Cellule 15** : **Graphiques comparatifs** barplots (performances test)
- **Cellule 16** : **Classification report** (précision/rappel/F1 par classe) pour chaque modèle
- **Cellule 17** : **Matrices de confusion** (2 variantes : counts + normalized) pour chaque modèle
- **Cellule 18** : **Validation croisée stratifiée 5-fold** (robustesse estimation)

### 7. Interprétabilité (Cellules 19-20)
- **Cellule 19** : **Courbe d'apprentissage** (train F1 vs validation F1 en fonction de l'effectif d'entraînement)
- **Cellule 20** : **Top 20 termes influents** (features importance ou |coef| pour modèles linéaires)

### 8. Analyse des Erreurs (Cellule 21)
- Identification des types d'erreurs les plus courants
- Exemples concrets de tweets mal classifiés
- Analyse des confusions entre classes

### 9. Sauvegarde du Modèle (Cellule 22)
- Sérialisation du modèle final en **pickle**
- Inclusion du vectoriseur TF-IDF + label_map
- Affichage taille fichier

### 10. API de Prédiction (Cellule 23)
- Fonction `predict_hate_speech()` robuste
- Demo sur 5 exemples texte
- Gestion modèles avec/sans `predict_proba()`

### 11. **Export Résultats Complets** (Cellule 24)
**Génération automatique de 8 fichiers** dans `../Outputs/` :
- `01_model_comparison.png` — Barplots Accuracy & F1-macro
- `02_confusion_matrices.png` — Matrices tous modèles
- `03_learning_curve.png` — Courbe d'apprentissage du meilleur
- `04_feature_importance.png` — Top 20 termes influents
- `validation_results.csv` — Résultats validation
- `test_results.csv` — Métriques test
- `cross_validation_results.csv` — CV 5-fold
- `summary_report.txt` — Rapport textuel horodaté

---

## Dépendances

### Installation automatique
Le notebook installe automatiquement via `%pip` :
```
pandas numpy scikit-learn matplotlib seaborn nltk scipy
```

### Installation manuelle
```bash
<<<<<<< HEAD
pip install pandas numpy scikit-learn matplotlib seaborn nltk scipy
```

### Tests d'exécution
- **Python 3.8+** requis
- Testé sur Kaggle Notebooks (kernel Python 3.10)
- Compatible Linux/Mac/Windows

---

## Résultats Attendus

### Performance des Modèles
Les résultats varient selon le dataset mais généralement :

| Modèle | Accuracy | Macro-F1 | Temps (s) |
|--------|----------|----------|-----------|
| Logistic Regression | ~93-95% | ~0.90-0.93 | +10 |
| Linear SVM | ~94-96% | ~0.91-0.94 | +30 |
| Multinomial NB | ~90-92% | ~0.87-0.90 | +5 |
| Random Forest | ~85-90% | ~0.82-0.88 | +45 |

**Modèle sélectionné** : Celui avec **Macro-F1 maximal** sur validation
- Raison : Équilibre entre classes déséquilibrées (Class 2 >> Classes 0,1)

### Structure des Résumés Générés
```
Outputs/
├── 01_model_comparison.png              # 16x6 inches, 300 DPI
├── 02_confusion_matrices.png            # Dynamique selon n modèles
├── 03_learning_curve.png                # 8x5 inches, courbe lissée
├── 04_feature_importance.png            # 9x7 inches, bar horizontal
├── validation_results.csv               # 3 colonnes (Model, Val Acc, Val F1)
├── test_results.csv                     # 5 colonnes (Model + 4 métriques)
├── cross_validation_results.csv         # 5 colonnes (Model + 4 CV stats)
└── summary_report.txt                   # Rapport complet horodaté
```

---

## Points Clés de Conformité

### Exigences du Projet
- ✓ **Exploration & Préparation** : Stats descriptives, distributions, corrélations, visualisations
- ✓ **4+ Algorithmes** : Logistic Reg, SVM, Naive Bayes, Random Forest
- ✓ **Réglage hyperparamètres** : RandomizedSearchCV pour chaque modèle
- ✓ **Train/Validation/Test** : Split stratifié 70/15/15
- ✓ **Évaluation robuste** : Accuracy, Precision, Recall, F1, CV k-fold
- ✓ **Matrices de confusion** : Counts + normalized pour tous
- ✓ **Visualisations** : Comparaisons, courbes d'apprentissage, features
- ✓ **Analyse d'erreurs** : Types d'erreurs, exemples mal classifiés
- ✓ **Résultats exportés** : Graphiques + CSVs + rapport texte

### Méthodologie
1. **Nettoyage NLP robuste** : lemmatisation + stop words
2. **Vectorisation TF-IDF** : n-grams (1-2) pour capturer bi-grammes pertinents
3. **Validation stratifiée** : respect du ratio classes dans chaque fold
4. **Comparaison équitable** : même preprocessing, même splits, CV interne

### Innovation
- Générération **automatique** des 8 fichiers de résultats
- Rapport texte **horodaté** et détaillé
- Diagrammes **haute résolution** (300 DPI)
- API de prédiction **robuste** et réutilisable

---

## Guide d'Utilisation

### Option 1 : Exécution sur Kaggle (Recommandé)
```
1. Accéder à kaggle.com → Notebooks → New Notebook
2. Importer/uploader le fichier hate-speech-model.ipynb
3. Vérifier l'accès au dataset : 
   - Add Data → Hate Speech and Offensive Language Dataset
4. Run All (cellules 1-24)
5. Télécharger les 8 fichiers depuis Outputs/
```

### Option 2 : Exécution Locale
```bash
# 1. Télécharger labeled_data.csv depuis Kaggle
# 2. Cloner le repo ou copier hate-speech-model.ipynb
# 3. Modifier cellule 4 avec chemin du CSV
# 4. Installer dépendances
=======
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
unset CUDA_PATH
>>>>>>> main
pip install -r requirements.txt

# 5. Lancer Jupyter
jupyter notebook Code/hate-speech-model.ipynb
```

<<<<<<< HEAD
### Option 3 : Utiliser le Modèle Pré-entraîné
```python
import pickle
import pandas as pd

# Charger le modèle sauvegardé
with open('hate_speech_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    tfidf = data['tfidf']
    label_map = data['label_map']

# Prédire
texte = "I hate this awful movie"
features = tfidf.transform([texte])
prediction = model.predict(features)[0]
print(f"Classe: {label_map[prediction]}")
```

---

## Exemples de Prédiction

```
Entrée: "I love spending time with my friends at the park"
Sortie: Neither (Confidence: 95.3%)
=======
Vérification minimale:

```bash
python -c "import sklearn, pandas, matplotlib; print('OK')"
```

## How-to: exécuter le projet

## 1) Utiliser `main.py` (CLI, recommandé)

Le `main.py` à la racine est le lanceur principal multi-runs.

### Commandes utiles

Run rapide:

```bash
python main.py --run-matrix default
```

Run complet:

```bash
python main.py --run-matrix exhaustive
```

Avec ajustement du malus DistilBERT CV proxy:

```bash
python main.py --run-matrix default --distilbert-proxy-penalty 0.02
```

### Ce que fait ce lanceur

- charge la matrice de runs (`default` ou `exhaustive`) depuis `Code/run_configs.py`;
- filtre les runs incompatibles avec l'environnement courant;
- exécute chaque run dans un subprocess isolé (meilleure stabilité mémoire);
- agrège et classe les runs;
- affiche un résumé du meilleur run;
- produit les artefacts dans `Outputs/`.

### Différence avec `Code/main.py`

- `main.py` (racine): orchestration multi-runs + comparaison globale.
- `Code/main.py`: pipeline unitaire (`run_pipeline`) pour un run unique (utile surtout en dev interne).

## 2) Utiliser `notebook_principal.ipynb`

Le notebook est l'interface interactive pour piloter les mêmes runs que la CLI.

### Étapes d'utilisation

1. Ouvrir `notebook_principal.ipynb`.
2. Exécuter la cellule d'import.
3. Ajuster les paramètres dans la section "Paramètres du notebook":
   - `RUN_MATRIX`;
   - `SELECTED_MODELS`;
   - profils DistilBERT/MLP/AdaBoost/GPU;
   - `DISTILBERT_PROXY_PENALTY`.
4. Exécuter la cellule d'orchestration (`run_all_configs`).
5. Lire les tableaux et le rapport synthèse.
6. Consulter les figures générées dans `Outputs/figures/`.

### Conseils pratiques notebook

- pour itérer vite: `RUN_MATRIX="default"` + peu de modèles;
- pour une analyse complète: `RUN_MATRIX="exhaustive"` + profils ciblés;
- si mémoire/VRAM limitée: éviter profils agressifs (`gpu_aggressive`, `vram_max`);
- si DistilBERT/cuML absent: runs incompatibles automatiquement ignorés.

## 3) Utiliser Docker

### Mode interactif notebook/shell

```bash
docker compose -f docker/docker-compose.simple.yml build
docker compose -f docker/docker-compose.simple.yml up notebook
```

Puis:
- URL: `http://localhost:8888`
- token par défaut: `techapp` (modifiable via `JUPYTER_TOKEN`)

Shell interactif dans le même environnement:

```bash
docker compose -f docker/docker-compose.simple.yml run --rm shell
```

### Mode pipeline non-interactif (run puis exit)

```bash
docker compose -f docker/docker-compose.pipeline.yml build
docker compose -f docker/docker-compose.pipeline.yml run --rm pipeline
```

Ce mode lance automatiquement:

```bash
python main.py --run-matrix default
```

avec limites mémoire/CPU définies dans `docker/docker-compose.pipeline.yml`.

### Variantes de build

Build alternatif Arch/Fedora:

```bash
docker build -f docker/Dockerfile.arch -t tech-app-devoir-ii:arch .
docker build -f docker/Dockerfile.fedora -t tech-app-devoir-ii:fedora .
```

Build latest plutôt que lock:

```bash
REQUIREMENTS_MODE=latest docker compose -f docker/docker-compose.simple.yml build
```

Activation RAPIDS/cuML (expérimental):

```bash
ENABLE_RAPIDS=1 docker compose -f docker/docker-compose.simple.yml build
```

## Paramètres importants (CLI + notebook)

- `RUN_MATRIX`: `default` (rapide) ou `exhaustive` (coûteux);
- `DISTILBERT_PROXY_PENALTY`: malus appliqué aux runs DistilBERT en CV proxy;
- `MODEL_PARAM_OVERRIDES["DistilBERT"]`: `epochs`, `batch_size`, `max_length`, etc.;
- `MODEL_GRID_OVERRIDES["<ModelName>"]`: surcharge de grille GridSearchCV;
- profils activables dans `Code/run_configs.py` (DistilBERT, MLP, AdaBoost, GPU);
- runs centralisés dans `Code/run_configs.py`, partagés par notebook + CLI.
>>>>>>> main

Entrée: "This movie is absolutely terrible, worst thing ever made"  
Sortie: Offensive Language (Confidence: 87.2%)

<<<<<<< HEAD
Entrée: "You're all stupid idiots who deserve nothing"
Sortie: Hate Speech (Confidence: 92.1%)
```
=======
- `Outputs/figures/` :
  - distribution des classes
  - valeurs manquantes
  - heatmap de corrélation
  - histogrammes / boxplots
  - matrices de confusion
  - compilation des matrices de confusion (tous les modèles)
  - comparaison des modèles
  - compilation synthèse comparative globale
  - couverture de statuts pour tous les modèles (trained/skipped/failed)
  - courbe d’apprentissage
  - importance des features du meilleur modèle (`feature_importance_best_model.png`)
  - heatmap de contribution des features entre modèles (`feature_importance_comparison_models.png`)
- `Outputs/reports/eda_summary.json`
- `Outputs/reports/metrics_report.json` (inclut `all_models`, statuts, erreurs éventuelles et métriques disponibles pour tous les modèles attendus)
- `Outputs/reports/metrics_report.md` (version lisible humain du report principal)
- `Outputs/reports/metrics_report_<run_name>.md` (version lisible par run)
- `Outputs/reports/runs_comparison_overview.md` (tableau lisible de comparaison inter-runs)
- `Outputs/reports/feature_importance_summary.json` (top termes influents par modèle entraîné)
- `Outputs/models/best_model.joblib`
  - si meilleur modèle deep learning: `Outputs/reports/best_model_deep_learning_note.json`
>>>>>>> main

---

<<<<<<< HEAD
## Troubleshooting

| Problème | Cause | Solution |
|---------|-------|---------|
| `/kaggle/input/ not found` | Dataset non lié | Ajouter le dataset dans Kaggle → Add Data |
| `ModuleNotFoundError` | Packages manquants | Réexécuter cellule 2 ou `pip install -r requirements.txt` |
| Manque de mémoire | Dataset trop volumineux | Réduire `max_features` dans TfidfVectorizer (cellule 10) |
| Résultats différents | Aléatoire non fixé | Normal ; `random_state=42` ne garantit pas 100% reproduction |
| Diagrammes pas visibles | Matplotlib non configuré | Ajouter `%matplotlib inline` au début (Jupyter) |

---
=======
- Naive Bayes
- Logistic Regression
- Linear SVC
- KNN
- Decision Tree
- Random Forest
- AdaBoost
- MLPClassifier
- DistilBERT (fine-tuning, si dépendances deep learning installées)
- LogisticRegressionGPU (cuML)
- LinearSVCGPU (cuML)
- KNNGPU (cuML)
- RandomForestGPU (cuML)

Chaque modèle est entraîné avec `GridSearchCV` et évalué avec:
- accuracy
- balanced accuracy
- précision macro
- rappel macro
- F1 macro
- F1 par classe (dont `hate_speech`)
- matrice de confusion
- validation croisée (k-fold) et IC95 (`mean ± std`) pour les modèles classiques

Sélection finale du meilleur modèle via score pondéré:
`w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean + w_hate * hate_recall_test`
avec une pénalité optionnelle si `hate_recall_test` est sous `HATE_RECALL_FLOOR`.
La `precision_macro` reste un indicateur diagnostique, mais n'est pas un critère principal de sélection.

Le pipeline produit aussi une matrice d'erreurs textuelles pour le meilleur modèle:
- `Outputs/reports/error_cases_best_model.json`
- `Outputs/reports/error_cases_best_model.md`
avec des exemples de faux négatifs/faux positifs sur `hate_speech`.
>>>>>>> main

## Références & Ressources

<<<<<<< HEAD
**Dataset** : [Hate Speech and Offensive Language Dataset - Kaggle](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

**Bibliothèques** :
- [scikit-learn](https://scikit-learn.org/) - ML AlgosSans
- [NLTK](https://www.nltk.org/) - NLP preprocessing
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [matplotlib/seaborn](https://matplotlib.org/) - Visualisation

**Concepts ML** :
- [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [RandomizedSearchCV](https://scikit-learn.org/stable/modules/model_selection.html)
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Métriques de classification](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## Auteur & Attribution

- **Projet** : Techniques d'Apprentissage (UQO, 2026)
- **Base initiale** : Code de [Elbeg Darkhanbaatar](https://kaggle.com/elbegdarkhanbaatar) (modifié et étoffé)
- **Améliorations** : Tuning hyperparamètres, export automatique, rapport texte, API robuste

### Source
Le code édite celui de l'utilisateur Kaggle Elbeg Darkhanbaatar, il a été modifié pour l'adapter aux consignes.

---

## 📄 Licence

Ce projet est à **usage éducatif**. Le dataset provient de Kaggle sous licence libre.

---

## Résumé

| Aspect | Détails |
|--------|---------|
| **Type** | Classification texte multiclasse |
| **Classes** | 3 (Hate Speech, Offensive Language, Neither) |
| **Modèles** | 4 (Logistic Reg, SVM, Naive Bayes, Random Forest) |
| **Accuracy** | 90-96% selon modèle |
| **Environment** | Kaggle Notebooks (conçu pour) |
| **Alternative** | Exécution locale possible (**en modifiant les chemins des fichiers in et out**)|
| **Résultats** | 8 fichiers (PNG + CSV + TXT) |
| **Temps d'exécution** | ~5-10 min (Kaggle) selon kernel |

---

**Dernière mise à jour** : Avril 2026  
**Statut** : Conforme aux exigences du projet
=======
Note GPU classiques (cuML):
- les modèles GPU sont intégrés dans la pipeline comme les autres via `model_zoo`;
- si les dépendances cuML/cupy sont absentes, les runs GPU-only sont ignorés dans l'orchestration multi-runs;
- dans le report, `feature_config.backend` indique le backend (`cpu_sklearn`, `gpu_cuml`, `gpu_torch`).

La figure `runs_comparison_overview.png` est volontairement zoomée sur l'intervalle `[0.6, 0.8]`
pour mieux visualiser les écarts fins entre runs.

Le notebook inclut aussi un interpréteur de résultats (`Code/result_interpreter.py`) qui imprime:
- le meilleur modèle et son score global;
- la répartition des statuts d'exécution de tous les modèles;
- un top-3 des modèles entraînés;
- les modèles à améliorer selon un seuil simple de `f1_macro`.

## Exécution multi-runs en subprocess

Les runs de `run_all_configs()` sont exécutés dans des subprocess Python isolés
via `Code/run_pipeline_subprocess.py`. Cette stratégie limite les pics mémoire
car chaque process libère ses ressources à la fin du run.

## Dépannage rapide

- Erreur Jupyter token: vérifier `JUPYTER_TOKEN` et le port exposé (`JUPYTER_PORT`);
- DistilBERT indisponible: vérifier installation de `torch`, `transformers`, `datasets`;
- modèles GPU absents: vérifier cuML/cupy + compatibilité CUDA/driver;
- exécution trop lente: commencer avec `--run-matrix default` et/ou réduire les profils actifs;
- mémoire saturée: réduire batch size DistilBERT, désactiver profils agressifs, garder l'exécution en subprocess.
>>>>>>> main
