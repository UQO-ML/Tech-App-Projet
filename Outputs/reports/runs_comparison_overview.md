# Comparaison inter-runs

| Run | Best model | Selection | Adjusted | Test F1 | DistilBERT CV proxy | Penalty |
|---|---|---:|---:|---:|---:|---:|
| run_e_method_strict_classic | LogisticRegression | 0.6977 | 0.6977 | 0.7327 | False | 0.0000 |
| run_d_balanced | DistilBERT | 0.7073 | 0.6973 | 0.7618 | True | 0.0100 |
| run_a_data_balance | LogisticRegression | 0.6977 | 0.6877 | 0.7327 | True | 0.0100 |
| run_d_robust | DistilBERT | 0.6788 | 0.6688 | 0.7631 | True | 0.0100 |
