# Soft-Target Evaluation

- rows: 2188
- patient/record units: 21
- official Round5 alarm changed: no
- dominant-class metrics remain auxiliary; these metrics evaluate the full future-window distribution

## Observed Metrics
- future_soft_cross_entropy: 1.3355 (95% patient bootstrap CI 0.8657-1.8449)
- future_soft_kl: 1.0096 (95% patient bootstrap CI 0.5489-1.5073)
- future_soft_brier: 0.3800 (95% patient bootstrap CI 0.2328-0.5527)
- future_abnormal_mass_mae: 0.2664 (95% patient bootstrap CI 0.1921-0.3382)
- future_abnormal_mass_rmse: 0.3546 (95% patient bootstrap CI 0.2713-0.4257)
- future_abnormal_mass_bias: 0.0129 (95% patient bootstrap CI -0.1121-0.1308)
