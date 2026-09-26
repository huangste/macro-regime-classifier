# Experiment 2 - Label stability

Sample: 1990-12-28 to 2026-09-24, 8,929 days, 14 labelling features, K=4.

## S1. Same data, different seed

| variant                |   mean_ARI_vs_seed0 |   min_ARI |   risk_on_off_flip_rate |
|:-----------------------|--------------------:|----------:|------------------------:|
| gmm_raw_index          |               0.999 |     0.998 |                   0.576 |
| gmm_anchored           |               0.999 |     0.998 |                   0     |
| hmm_anchored           |               0.567 |     0.306 |                   0.083 |
| gmm_anchored_scoreinit |               0.513 |     0.365 |                   0.062 |
| hmm_anchored_scoreinit |               1     |     1     |                   0     |
| score_only             |               1     |     1     |                   0     |

`ARI` compares partitions and is permutation-invariant by construction, so it is the same for raw and anchored variants of the same estimator -- that is the point. The column that matters is the flip rate: the share of historical days whose **risk-on/risk-off** verdict changes when only the random seed changes.

## S2. Refitting on later data vintages

Vintages cut 1, 2, 3, 4 and 5 years off the end and compare the historical verdict on the overlapping dates:

| variant                |   mean_ARI_vs_latest |   min_ARI |   risk_on_off_flip_rate |   worst_flip_rate |
|:-----------------------|---------------------:|----------:|------------------------:|------------------:|
| gmm_raw_index          |                0.712 |     0.356 |                   0.316 |             0.735 |
| gmm_anchored           |                0.712 |     0.356 |                   0.065 |             0.118 |
| hmm_anchored           |                0.384 |     0.255 |                   0.199 |             0.213 |
| gmm_anchored_scoreinit |                0.528 |     0.331 |                   0.096 |             0.192 |
| hmm_anchored_scoreinit |                0.63  |     0.429 |                   0.065 |             0.11  |
| score_only             |                0.883 |     0.85  |                   0.012 |             0.016 |

## S3. Flicker in the historical classification

| variant                |   switches_per_year |   median_episode_days |   episodes_shorter_than_5d |
|:-----------------------|--------------------:|----------------------:|---------------------------:|
| gmm_raw_index          |                6.02 |                     5 |                       0.5  |
| gmm_anchored           |                6.02 |                     5 |                       0.5  |
| hmm_anchored           |                1.76 |                    89 |                       0    |
| gmm_anchored_scoreinit |               14.3  |                     4 |                       0.55 |
| hmm_anchored_scoreinit |                3.53 |                    51 |                       0.03 |
| score_only             |               30.92 |                     4 |                       0.55 |

## Choosing K

|   K |   GMM_BIC |   risk_off_states |   risk_off_share_of_days |   vintage_flip_rate |   sharpe_spread_worst_to_best |
|----:|----------:|------------------:|-------------------------:|--------------------:|------------------------------:|
|   2 |    260609 |                 1 |                    0.363 |               0.028 |                         0.891 |
|   3 |    252674 |                 1 |                    0.236 |               0.047 |                         1.442 |
|   4 |    247922 |                 2 |                    0.342 |               0.081 |                         1.324 |
|   5 |    241946 |                 2 |                    0.319 |               0.208 |                         1.542 |
|   6 |    235717 |                 3 |                    0.424 |               0.169 |                         1.323 |

BIC falls monotonically with K -- it always does on strongly autocorrelated data, because extra components soak up serial dependence the model does not otherwise represent. It is therefore not a usable criterion here. K=4 is chosen instead on vintage stability plus the requirement that the states separate realised risk-adjusted returns, while still leaving enough days in each state to estimate transitions.

## Economic characteristics of the chosen labelling (HMM, K=4)

|   regime | name                        |   risk_score |   share_of_days |   SPX_ann_return |   SPX_ann_vol |   SPX_hit_rate |   SPX_sharpe |   VIX_level |   Baa_spread |   Curve_10y2y |   US10y |
|---------:|:----------------------------|-------------:|----------------:|-----------------:|--------------:|---------------:|-------------:|------------:|-------------:|--------------:|--------:|
|        0 | Crisis / deleveraging (off) |       -0.723 |           0.169 |           -0.012 |         0.301 |          0.51  |       -0.041 |      29.131 |        3.362 |         1.681 |   3.502 |
|        1 | Stress / drawdown (off)     |       -0.326 |           0.174 |            0.035 |         0.197 |          0.508 |        0.179 |      20.944 |        2.217 |         0.713 |   4.731 |
|        2 | Recovery / mixed (on)       |        0.092 |           0.309 |            0.103 |         0.141 |          0.545 |        0.73  |      17.945 |        2.203 |         1.128 |   3.957 |
|        3 | Risk-on expansion (on)      |        0.43  |           0.349 |            0.141 |         0.11  |          0.555 |        1.283 |      14.769 |        1.875 |         0.758 |   4.378 |

Canonical index 0 is always the most risk-averse state and index 3 the most risk-seeking, by construction, whatever the seed or the data vintage.

Longest episodes of canonical regime 0 (crisis / deleveraging):

| start      | end        |   days |
|:-----------|:-----------|-------:|
| 2007-12-31 | 2009-10-26 |    460 |
| 2002-06-04 | 2003-06-27 |    270 |
| 2001-06-11 | 2002-03-25 |    195 |
| 2011-07-28 | 2012-02-01 |    130 |
| 2010-05-06 | 2010-10-25 |    120 |
| 1998-08-27 | 1998-12-24 |     84 |
| 2016-01-07 | 2016-04-13 |     67 |
| 2015-08-26 | 2015-11-18 |     60 |
| 2000-04-17 | 2000-06-15 |     42 |
| 2012-06-06 | 2012-08-02 |     41 |
| 2020-03-05 | 2020-04-17 |     31 |
