# Experiment 1 - Reproduction and diagnosis of the original notebook

## Reproduction

* Raw panel after the notebook's `dropna()`: **6800 rows**, 2000-08-30 to 2026-09-24
* Feature matrix: **6737 rows x 54 columns**

The download starts at 1990 but the effective sample starts **2000-08-30**: `GC=F` and `CL=F` have no history before 2000, and `dropna()` on the concatenated panel silently truncates everything else to match. Thirty-six years of available S&P history became twenty-five.

* PCA retains **21 components** for 85% of variance (first three explain 37.0%)
* GMM(5, diag) component shares: {0: 0.233, 1: 0.277, 2: 0.249, 3: 0.033, 4: 0.208}

* Notebook's reported setup: H=21, stress mask = components [0, 2], base rate 48.3%
* Logistic CV log-loss **0.6367** vs unconditional baseline **0.6926**

## Diagnosis

### D1. The effective sample is roughly 300 observations, not 6,000

The target is the regime 21 trading days ahead, sampled every day, so consecutive rows share 20/21 of their horizon. The 6,716 rows carry on the order of **319 non-overlapping observations**, and the risk-off class contains roughly **154** of them. Every standard error in the notebook is understated by a factor of about sqrt(21) = 4.6.

### D2. Component indices carry no stable meaning

Refitting with `n_init=1` and different seeds (the notebook's `n_init=50` hides, but does not remove, this):

|   seed |   ARI_vs_seed0 |   index_of_highest_vol_component |
|-------:|---------------:|---------------------------------:|
|      0 |          1     |                                3 |
|      1 |          0.244 |                                0 |
|      2 |          0.557 |                                3 |
|      7 |          0.551 |                                4 |
|     42 |          0.552 |                                4 |

Refitting on earlier data vintages, which is what happens every time the notebook is re-run on a later date:

|   days_removed_from_end |   index_of_highest_vol_component |   ARI_vs_full_sample_fit |
|------------------------:|---------------------------------:|-------------------------:|
|                       0 |                                3 |                    0.882 |
|                      60 |                                1 |                    0.792 |
|                     125 |                                2 |                    0.871 |
|                     250 |                                4 |                    0.774 |

The index of the crisis component moves. The notebook hard-codes `np.isin(labels, [0, 2])` as the stress mask while the comment on the line above says `classify 1, 3 as crisis` and the plot legend calls component 3 `Crisis/Deleveraging`. Those three statements cannot all be true, which is the label-drift problem showing up in the code itself.

### D3. The scaler, the PCA and the regime labels all see the future

`StandardScaler`, `PCA` and the `GaussianMixture` are all fitted on the entire sample **before** `cross_val_score` splits it. Two distinct leaks follow, and they have to be measured separately.

**Leak 1 - the feature pipeline.** Holding the target fixed (so the comparison is like-for-like) and varying only whether the scaler and PCA are fitted on the training fold or on everything:

| feature pipeline                          |   mean_fold_log_loss |
|:------------------------------------------|---------------------:|
| scaler + PCA fitted on full sample        |               0.8082 |
| scaler + PCA fitted on training fold only |               0.6551 |

**Leak 2 - the target.** This one cannot be measured by swapping a line: the labels themselves are produced by a mixture fitted with hindsight, so a causal version has a different target and its log-losses are not comparable on the same scale. That is why experiment 3 evaluates skill *relative to a baseline computed under the same labelling*, and refits the labeller inside every walk-forward fold.

### D4. A persistence baseline that uses no features at all

The notebook compares its model only against the unconditional base rate. But regimes are extremely persistent, so the honest question is whether the features add anything beyond knowing today's state. Fitting P(risk-off in 21d | risk-off today) on the training fold only:

|                  |   mean fold log-loss |
|:-----------------|---------------------:|
| unconditional    |               0.7496 |
| persistence_only |               0.6531 |
| logistic_on_PCA  |               0.8082 |

Lower is better. The published model is beaten by a two-parameter persistence rule, and does worse than simply quoting the base rate.

### D5. The headline number depends on how many folds you ask for

The final cell of the notebook scans `n_splits` in {2, 3, 4, 5} across four model families and prints all sixteen results. Nothing corrects for that search, and the setting used earlier in the notebook (`n_splits=2`) happens to be the most flattering one:

|   n_splits |   mean_log_loss |   worst_fold |   best_fold |
|-----------:|----------------:|-------------:|------------:|
|          2 |          0.6367 |       0.6846 |      0.5887 |
|          3 |          0.6899 |       0.9118 |      0.5655 |
|          4 |          0.6701 |       0.8451 |      0.5431 |
|          5 |          0.7997 |       1.4941 |      0.5589 |

With two folds the model appears to beat the unconditional baseline of 0.6926. With five it does not. A result that reverses on a validation setting is not evidence of predictive information.
