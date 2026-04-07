# Rapport de métriques (lisible)

## Résumé global
- Modèle retenu: `LogisticRegression`
- Score de sélection: `0.6977`
- F1 macro test du meilleur: `0.7327`
- Échantillons: `24783`

## Statuts d'exécution
- trained: `10`
- skipped: `0`
- failed: `3`
- modèles attendus: `['NaiveBayes', 'LogisticRegression', 'LinearSVC', 'KNN', 'DecisionTree', 'RandomForest', 'AdaBoost', 'MLPClassifier', 'LogisticRegressionGPU', 'LinearSVCGPU', 'KNNGPU', 'RandomForestGPU', 'DistilBERT']`
- modèles entraînés: `['NaiveBayes', 'LogisticRegression', 'LinearSVC', 'KNN', 'DecisionTree', 'RandomForest', 'AdaBoost', 'MLPClassifier', 'KNNGPU', 'DistilBERT']`

## Méthode de sélection
- Formule: `selection_score = w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean + w_hate * hate_recall_test - penalty_if(hate_recall_test < hate_recall_floor)`
- Poids: validation=0.3000, test=0.3500, cv=0.2000, hate_recall=0.1500
- Seuil hate_recall: `0.4000`
- Pénalité hate_recall: `0.0300`
- Politique précision: `La précision macro est suivie comme métrique diagnostique, mais n'est pas utilisée comme critère principal de sélection.`
- Modèles avec CV proxy: `['DistilBERT']`

## Configuration du run
- max_samples: `None`
- distilbert_epochs: `1`
- include_distilbert: `True`
- algorithm_switches: `{'NaiveBayes': True, 'LogisticRegression': True, 'LinearSVC': True, 'KNN': True, 'DecisionTree': True, 'RandomForest': True, 'AdaBoost': True, 'MLPClassifier': True, 'LogisticRegressionGPU': True, 'LinearSVCGPU': True, 'KNNGPU': True, 'RandomForestGPU': True, 'DistilBERT': True}`
- test_size: `0.2`
- val_size: `0.1`
- cv_folds: `5`
- scoring: `f1_macro`
- model_param_overrides: `{}`
- model_grid_overrides: `{}`
- selection_weights: `[0.3, 0.35, 0.2, 0.15]`
- hate_recall_floor: `0.4`
- hate_recall_penalty: `0.03`
- random_state: `42`

## Détail par modèle

| Modèle | Status | Selection score | Balanced Acc | Val F1 | Test F1 | CV mean ± CI95 | Hate recall | Hate F1 | Pénalité appliquée | Erreur |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NaiveBayes | trained | 0.4162 | 0.4919 | 0.5191 | 0.5219 | 0.5205 ± 0.0099 | 0.0245 | 0.0476 | 0.0300 |  |
| LogisticRegression | trained | 0.6977 | 0.7748 | 0.7257 | 0.7327 | 0.7298 ± 0.0055 | 0.5175 | 0.4398 | 0.0000 |  |
| LinearSVC | trained | 0.6817 | 0.7508 | 0.7169 | 0.7338 | 0.7316 ± 0.0069 | 0.4231 | 0.4130 | 0.0000 |  |
| KNN | trained | 0.3209 | 0.3945 | 0.3668 | 0.3669 | 0.3682 ± 0.0334 | 0.2587 | 0.1152 | 0.0300 |  |
| DecisionTree | trained | 0.6522 | 0.7425 | 0.6953 | 0.6942 | 0.6912 ± 0.0071 | 0.4161 | 0.3371 | 0.0000 |  |
| RandomForest | trained | 0.6887 | 0.7612 | 0.7342 | 0.7251 | 0.7299 ± 0.0092 | 0.4580 | 0.4302 | 0.0000 |  |
| AdaBoost | trained | 0.4481 | 0.6172 | 0.5546 | 0.5382 | 0.5224 ± 0.0262 | 0.1259 | 0.1865 | 0.0300 |  |
| MLPClassifier | trained | 0.4937 | 0.6044 | 0.6018 | 0.6024 | 0.5959 ± 0.0162 | 0.0874 | 0.1515 | 0.0300 |  |
| LogisticRegressionGPU | failed | N/A | N/A | N/A | N/A | n/a | N/A | N/A | N/A | /opt/cuda/include/cuda_fp8.hpp(1498): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp8.hpp(1499): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e5m2(const double4 f) {
      ^

/opt/cuda/include/cuda_fp8.hpp(1559): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/opt/cuda/include/cuda_fp8.hpp(2127): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp8.hpp(2128): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e4m3(const double4 f) {
      ^

/opt/cuda/include/cuda_fp8.hpp(2188): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

/opt/cuda/include/cuda_fp8.hpp(2776): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp8.hpp(2777): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e8m0(const double4 f) {
      ^

/opt/cuda/include/cuda_fp8.hpp(2838): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

/opt/cuda/include/cuda_fp6.hpp(1032): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp6.hpp(1033): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e3m2(const double4 f) {
      ^

/opt/cuda/include/cuda_fp6.hpp(1093): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

/opt/cuda/include/cuda_fp6.hpp(1614): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp6.hpp(1615): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e2m3(const double4 f) {
      ^

/opt/cuda/include/cuda_fp6.hpp(1675): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

/opt/cuda/include/cuda_fp4.hpp(973): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp4.hpp(974): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP4__ __nv_fp4x4_e2m1(const double4 f) {
      ^

/opt/cuda/include/cuda_fp4.hpp(1036): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

12 errors detected in the compilation of "/tmp/tmpu0157bg3/f76abf3c16ef2526e059499e2786b5a72ae1484a.cubin.cu".
 |
| LinearSVCGPU | failed | N/A | N/A | N/A | N/A | n/a | N/A | N/A | N/A | /opt/cuda/include/cuda_fp8.hpp(1498): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp8.hpp(1499): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e5m2(const double4 f) {
      ^

/opt/cuda/include/cuda_fp8.hpp(1559): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/opt/cuda/include/cuda_fp8.hpp(2127): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp8.hpp(2128): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e4m3(const double4 f) {
      ^

/opt/cuda/include/cuda_fp8.hpp(2188): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

/opt/cuda/include/cuda_fp8.hpp(2776): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp8.hpp(2777): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e8m0(const double4 f) {
      ^

/opt/cuda/include/cuda_fp8.hpp(2838): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

/opt/cuda/include/cuda_fp6.hpp(1032): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp6.hpp(1033): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e3m2(const double4 f) {
      ^

/opt/cuda/include/cuda_fp6.hpp(1093): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

/opt/cuda/include/cuda_fp6.hpp(1614): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp6.hpp(1615): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e2m3(const double4 f) {
      ^

/opt/cuda/include/cuda_fp6.hpp(1675): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

/opt/cuda/include/cuda_fp4.hpp(973): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp4.hpp(974): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP4__ __nv_fp4x4_e2m1(const double4 f) {
      ^

/opt/cuda/include/cuda_fp4.hpp(1036): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

12 errors detected in the compilation of "/tmp/tmppyig3oy6/f76abf3c16ef2526e059499e2786b5a72ae1484a.cubin.cu".
 |
| KNNGPU | trained | 0.4215 | 0.5131 | 0.5299 | 0.5047 | 0.5139 ± 0.0137 | 0.0874 | 0.1168 | 0.0300 |  |
| RandomForestGPU | failed | N/A | N/A | N/A | N/A | n/a | N/A | N/A | N/A | /opt/cuda/include/cuda_fp8.hpp(1498): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp8.hpp(1499): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e5m2(const double4 f) {
      ^

/opt/cuda/include/cuda_fp8.hpp(1559): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/opt/cuda/include/cuda_fp8.hpp(2127): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp8.hpp(2128): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e4m3(const double4 f) {
      ^

/opt/cuda/include/cuda_fp8.hpp(2188): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

/opt/cuda/include/cuda_fp8.hpp(2776): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp8.hpp(2777): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e8m0(const double4 f) {
      ^

/opt/cuda/include/cuda_fp8.hpp(2838): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

/opt/cuda/include/cuda_fp6.hpp(1032): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp6.hpp(1033): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e3m2(const double4 f) {
      ^

/opt/cuda/include/cuda_fp6.hpp(1093): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

/opt/cuda/include/cuda_fp6.hpp(1614): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp6.hpp(1615): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e2m3(const double4 f) {
      ^

/opt/cuda/include/cuda_fp6.hpp(1675): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

/opt/cuda/include/cuda_fp4.hpp(973): error: this declaration has no storage class or type specifier
      __NV_SILENCE_DEPRECATION_BEGIN
      ^

/opt/cuda/include/cuda_fp4.hpp(974): error: expected a ";"
      explicit __CUDA_HOSTDEVICE_FP4__ __nv_fp4x4_e2m1(const double4 f) {
      ^

/opt/cuda/include/cuda_fp4.hpp(1036): warning #12-D: parsing restarts here after previous syntax error
  };
  ^

12 errors detected in the compilation of "/tmp/tmpg2psw65z/b2e9282a567a8f103e5223b9f017cff49a8aaa71.cubin.cu".
 |
| DistilBERT | trained | 0.6060 | 0.6905 | 0.7180 | 0.7044 | 0.7180 ± n/a | 0.2028 | 0.2822 | 0.0300 |  |

## Analyse d'erreurs textuelles
- Fichier JSON: `Outputs/reports/error_cases_best_model.json`
- Fichier Markdown: `Outputs/reports/error_cases_best_model.md`
- Résumé features par modèle: `Outputs/reports/feature_importance_summary.json`
- Heatmap comparative: `Outputs/figures/feature_importance_comparison_models.png`
