# Comparaison inter-runs

| Run | Best model | Selection | Adjusted | Test F1 | DistilBERT CV proxy | Penalty |
|---|---|---:|---:|---:|---:|---:|
| run_e_method_strict_classic | LogisticRegression | 0.6977 | 0.6977 | 0.7327 | False | 0.0000 |
| run_g_distilbert_safe_ep3 | DistilBERT | 0.7081 | 0.6881 | 0.7669 | True | 0.0200 |
| run_a_data_balance | LogisticRegression | 0.6977 | 0.6777 | 0.7327 | True | 0.0200 |
