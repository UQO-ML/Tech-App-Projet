# Comparaison inter-runs

| Run | Best model | Selection | Adjusted | Test F1 | DistilBERT CV proxy | Penalty |
|---|---|---:|---:|---:|---:|---:|
| run_g_distilbert_safe_ep3 | DistilBERT | 0.7539 | 0.7439 | 0.7564 | True | 0.0100 |
| run_e_method_strict_classic | RandomForest | 0.7318 | 0.7318 | 0.7251 | False | 0.0000 |
| run_f_cv_heavy_classic | RandomForest | 0.7309 | 0.7309 | 0.7251 | False | 0.0000 |
| run_d_distilbert_focus | DistilBERT | 0.7400 | 0.7300 | 0.7543 | True | 0.0100 |
| run_c_classic_focus | LogisticRegression | 0.7297 | 0.7297 | 0.7327 | False | 0.0000 |
| run_a_data_balance | LogisticRegression | 0.7295 | 0.7195 | 0.7327 | True | 0.0100 |
| run_b_data_low_balance | RandomForest | 0.7212 | 0.7112 | 0.7250 | True | 0.0100 |
