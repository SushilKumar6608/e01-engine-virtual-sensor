# Baselines on VALIDATION (simulated data)

Dataset: dataset_v2. Fitted on train only. Test and stress splits not touched.

M0 delivery ratio: 1.0215
M0b a, b (best start): 0.9502, 0.0289; all starts (params, RMSE vs measured on fit rows): [([0.9502114647767563, 0.028899163871142032], 9.509493857709998), ([0.9502113249035858, 0.02889922079349306], 9.509493857708025), ([0.9502108475487308, 0.028899415052647967], 9.509493857751748)]

M0b episode-bootstrap (20 resamples) 95 % interval: a [0.9365, 0.9674], b [0.0219, 0.0345]
Noise floor (RMS measured - true): 1.50 K
Lag (true vs steady): RMS 1.02 K, p99 4.50 K
Rows: 270000; with measured exhaust value: 270000; dropouts: 0

## RMSE vs measured [K] (95 % episode-bootstrap interval)

| group | per_fuel_constant | load_lookup | M0 | M0b |
|---|---|---|---|---|
| all | 22.95 [21.41, 24.47] | 18.56 [17.27, 19.81] | 10.47 [9.72, 11.15] | 9.57 [9.16, 10.05] |
| diesel_like | 25.58 [23.67, 27.36] | 19.68 [18.23, 21.38] | 11.34 [10.20, 12.52] | 9.77 [9.07, 10.48] |
| methanol | 16.69 [14.61, 18.69] | 16.17 [14.31, 17.95] | 8.51 [7.79, 9.25] | 9.18 [8.46, 9.89] |
| [0.4, 0.55) | 30.83 [26.78, 35.37] | 22.76 [20.20, 25.16] | 17.69 [15.85, 19.72] | 12.56 [11.11, 14.10] |
| [0.55, 0.7) | 20.89 [19.04, 22.95] | 18.57 [17.00, 20.03] | 8.05 [7.29, 8.88] | 8.79 [8.06, 9.61] |
| [0.7, 0.85) | 21.07 [18.86, 23.09] | 17.27 [15.73, 18.77] | 7.92 [7.34, 8.52] | 7.77 [7.22, 8.31] |
| [0.85, 1.01) | 18.76 [16.26, 21.16] | 13.60 [11.72, 15.73] | 7.78 [6.77, 8.73] | 11.95 [10.57, 13.16] |

Pre-registered ordering (constant > lookup > M0 > M0b, per fuel): DOES NOT HOLD - run the defined checks, report as is, do not tune
