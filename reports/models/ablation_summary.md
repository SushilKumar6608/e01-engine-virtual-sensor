# Feature ablation A-D and models M1, M2 on VALIDATION (simulated data)

RMSE against measured [K], 95 % episode-bootstrap interval. B-D predict the residual of cross-fitted M0b; A predicts the target directly.
Ramp rows: fuel-flow slope over 30 s in the top 10 % of train (threshold 0.00042 kg/s per s). The 10 s large-change rule of the feature 'since_change' hits only 60 validation rows in 1 episode, so it is not used for reporting.
Fitted on every 5th second of train, scalers on train only. Test and stress splits not touched.

| model | all | diesel_like | methanol | load>=0.85 | ramp (top 10 % |fuel slope|) | not ramp |
|---|---|---|---|---|---|---|
| M0 | 10.47 [9.77,11.18] | 11.34 [10.19,12.50] | 8.51 [7.79,9.26] | 7.78 [6.76,8.75] | 8.32 [7.82,8.84] | 10.67 [9.95,11.40] |
| M0b | 9.57 [9.16,10.07] | 9.77 [9.05,10.44] | 9.18 [8.43,9.90] | 11.95 [10.54,13.22] | 8.46 [7.89,8.95] | 9.69 [9.25,10.20] |
| M1-A | 5.19 [4.82,5.59] | 5.27 [4.83,5.69] | 5.01 [4.32,5.85] | 4.94 [4.34,5.70] | 5.50 [4.96,5.96] | 5.15 [4.77,5.56] |
| M2-A | 4.52 [4.25,4.79] | 4.56 [4.17,4.94] | 4.45 [3.98,4.94] | 3.85 [3.41,4.39] | 5.84 [5.16,6.38] | 4.35 [4.09,4.65] |
| M1-B | 3.91 [3.65,4.20] | 3.97 [3.62,4.33] | 3.80 [3.30,4.33] | 3.72 [3.34,4.11] | 4.85 [4.30,5.29] | 3.80 [3.53,4.10] |
| M2-B | 3.96 [3.75,4.26] | 4.03 [3.66,4.44] | 3.82 [3.51,4.20] | 3.36 [2.98,3.71] | 4.74 [4.28,5.17] | 3.86 [3.65,4.17] |
| M1-C | 3.61 [3.35,3.91] | 3.59 [3.24,3.96] | 3.65 [3.13,4.16] | 3.59 [3.20,3.98] | 3.73 [3.42,4.08] | 3.60 [3.34,3.91] |
| M2-C | 3.69 [3.47,3.97] | 3.64 [3.28,4.02] | 3.77 [3.39,4.15] | 3.23 [2.89,3.58] | 3.70 [3.42,4.01] | 3.69 [3.46,3.97] |
| M1-D | 3.61 [3.35,3.91] | 3.59 [3.24,3.96] | 3.65 [3.13,4.16] | 3.59 [3.20,3.98] | 3.73 [3.42,4.08] | 3.60 [3.34,3.91] |
| M2-D | 3.69 [3.47,3.97] | 3.64 [3.28,4.02] | 3.77 [3.39,4.15] | 3.23 [2.89,3.58] | 3.70 [3.42,4.01] | 3.69 [3.46,3.97] |

Permutation importance, M2 variant C (RMSE increase when the feature is shuffled, K):

heat_per_rev      4.856
n_rpm             2.957
n_rpm_mean10      1.164
m_fuel_mean60     0.929
nominal_est       0.633
m_fuel            0.614
n_rpm_slope30     0.459
cooler_dt         0.375
lambda_proxy      0.316
t_amb             0.267
n_tc_corr         0.240
m_fuel_slope30    0.166
p_amb             0.146
t_scav            0.118
n_tc              0.061
fuel_code         0.036
fuel_power        0.009
p_scav            0.006
m_fuel_mean10     0.004
since_change      0.000
boost_ratio      -0.000

Hyperparameter grid (validation RMSE, all rows): [{"variant": "A", "max_leaf_nodes": 15, "min_samples_leaf": 200, "val_rmse": 4.518598706839546}, {"variant": "A", "max_leaf_nodes": 31, "min_samples_leaf": 200, "val_rmse": 4.761567339915193}, {"variant": "A", "max_leaf_nodes": 15, "min_samples_leaf": 1000, "val_rmse": 4.616803511227394}, {"variant": "A", "max_leaf_nodes": 31, "min_samples_leaf": 1000, "val_rmse": 4.7133288681408265}, {"variant": "B", "max_leaf_nodes": 15, "min_samples_leaf": 200, "val_rmse": 3.9913941092052623}, {"variant": "B", "max_leaf_nodes": 31, "min_samples_leaf": 200, "val_rmse": 4.016474070142712}, {"variant": "B", "max_leaf_nodes": 15, "min_samples_leaf": 1000, "val_rmse": 3.957480796556759}, {"variant": "B", "max_leaf_nodes": 31, "min_samples_leaf": 1000, "val_rmse": 4.009878470274471}, {"variant": "C", "max_leaf_nodes": 15, "min_samples_leaf": 200, "val_rmse": 3.7233152602945085}, {"variant": "C", "max_leaf_nodes": 31, "min_samples_leaf": 200, "val_rmse": 3.7793710268826164}, {"variant": "C", "max_leaf_nodes": 15, "min_samples_leaf": 1000, "val_rmse": 3.6875058127225877}, {"variant": "C", "max_leaf_nodes": 31, "min_samples_leaf": 1000, "val_rmse": 3.737860891590089}, {"variant": "D", "max_leaf_nodes": 15, "min_samples_leaf": 200, "val_rmse": 3.7233152602945085}, {"variant": "D", "max_leaf_nodes": 31, "min_samples_leaf": 200, "val_rmse": 3.7793710268826164}, {"variant": "D", "max_leaf_nodes": 15, "min_samples_leaf": 1000, "val_rmse": 3.6875058127225877}, {"variant": "D", "max_leaf_nodes": 31, "min_samples_leaf": 1000, "val_rmse": 3.737860891590089}]
