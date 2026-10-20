# M3 + split conformal + abstain rule on VALIDATION (simulated data; no test or stress split read)

alpha 0.1; conformal correction (global) 2.075 K; per fuel 1.802 (diesel-like), 2.305 (methanol)

## Coverage of the 90 % interval

| split | interval | reference | coverage | episodes <80 % | worst-decile episode |
|---|---|---|---|---|---|
| calibration | 90 % | t_exh_meas | 90.0 % | 13.3 % | 75.8 % |
| calibration | 90 % | t_exh_true | 92.5 % | 9.3 % | 84.1 % |
| validation | 90 % | t_exh_meas | 90.7 % | 17.3 % | 73.3 % |
| validation | 90 % | t_exh_true | 93.0 % | 10.7 % | 78.5 % |
| validation (per-fuel correction) | 90 % | t_exh_meas | 90.4 % | 18.0 % | 72.4 % |
| validation (per-fuel correction) | 90 % | t_exh_true | 92.6 % | 10.0 % | 81.0 % |
| validation (uncorrected quantiles) | 90 % | t_exh_meas | 72.2 % | 47.3 % | 34.9 % |
| validation (uncorrected quantiles) | 90 % | t_exh_true | 74.7 % | 44.0 % | 34.8 % |

## Coverage and width by group on validation (global correction, vs measured)

| group | coverage | mean width [K] | p95 width [K] |
|---|---|---|---|
| all | 90.7 % | 12.5 | 16.1 |
| diesel_like | 91.9 % | 12.8 | 16.7 |
| methanol | 88.5 % | 11.8 | 14.2 |
| load [0.4, 0.55) | 93.5 % | 13.9 | 18.2 |
| load [0.55, 0.7) | 90.2 % | 12.4 | 15.7 |
| load [0.7, 0.85) | 89.1 % | 12.0 | 14.9 |
| load [0.85, 1.01) | 93.6 % | 11.7 | 14.6 |

## Abstain rule on validation (in-distribution, so every abstain is a false abstain)

OOD threshold (99th percentile of calibration scores) 0.941; width tolerance 30.0 K (assumed)

| reason | share of rows |
|---|---|
| input_out_of_range | 0.00 % |
| input_stuck | 0.00 % |
| far_from_training | 0.82 % |
| interval_too_wide | 0.00 % |
| abstain | 0.82 % |

Coverage of rows answered: 90.7 % (all rows 90.7 %). Row-wise numbers; rows within an episode are not independent.
Corrupted copy (first 1000 rows p_scav = 9 bar): input_out_of_range flagged 100 % of them, 0.0 % of the rest.
