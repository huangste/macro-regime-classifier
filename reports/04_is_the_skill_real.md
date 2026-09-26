# Experiment 4 - Is the measured skill real?

## M1. How much of the future feature vector is already known?

| feature           |   window_days |   overlap_h5 |   autocorr_h5 |   overlap_h10 |   autocorr_h10 |   overlap_h25 |   autocorr_h25 |
|:------------------|--------------:|-------------:|--------------:|--------------:|---------------:|--------------:|---------------:|
| SPX_rmom21        |            21 |        0.762 |         0.715 |         0.524 |          0.486 |         0     |         -0.033 |
| SPX_rmom63        |            63 |        0.921 |         0.876 |         0.841 |          0.772 |         0.603 |          0.505 |
| SPX_logvol21      |            21 |        0.762 |         0.938 |         0.524 |          0.85  |         0     |          0.604 |
| SPX_volratio      |            63 |        0.921 |         0.827 |         0.841 |          0.592 |         0.603 |         -0.046 |
| SPX_dd252         |           252 |        0.98  |         0.972 |         0.96  |          0.949 |         0.901 |          0.877 |
| VIX_log           |             1 |        0     |         0.922 |         0     |          0.876 |         0     |          0.763 |
| Baa_log           |             1 |        0     |         0.993 |         0     |          0.984 |         0     |          0.953 |
| Baa_chg63         |            63 |        0.921 |         0.953 |         0.841 |          0.885 |         0.603 |          0.622 |
| Slope_10Y2Y_level |             1 |        0     |         0.996 |         0     |          0.992 |         0     |          0.979 |
| DGS10_chg63       |            63 |        0.921 |         0.925 |         0.841 |          0.855 |         0.603 |          0.612 |
| RUT_rel_SPX_63    |            63 |        0.921 |         0.912 |         0.841 |          0.826 |         0.603 |          0.547 |
| Nikkei_rmom63     |            63 |        0.921 |         0.891 |         0.841 |          0.794 |         0.603 |          0.544 |
| USDJPY_rmom63     |            63 |        0.921 |         0.9   |         0.841 |          0.809 |         0.603 |          0.567 |
| SPX_bond_corr63   |            63 |        0.921 |         0.988 |         0.841 |          0.971 |         0.603 |          0.912 |

`overlap_hN` is the share of the feature's trailing window at `t+N` that was already observed at `t`; `autocorr_hN` is the realised lag-N autocorrelation. Both say the same thing: at h=5 the state in five days is largely arithmetic, and at h=25 the 63-day features still retain 60% of their window and correlations above 0.8.

Mean lag-5 autocorrelation across labelling features: **0.915**; lag-25: **0.600**.

## M2. Targets with no window overlap

`mkt_negative_return` is 1 when the S&P log return over `(t, t+h]` is negative. `mkt_high_vol` is 1 when realised volatility over the same window exceeds its training-sample median. Neither shares any data with the features at `t`. The benchmark is unchanged in spirit: the regime-conditional frequency estimated on the training fold, i.e. today's state and nothing else.

### `mkt_high_vol`, h = 5

Base rate 54.2%, 6,804 daily observations (~1,360 non-overlapping).

|   h | target       | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:-------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|   5 | mkt_high_vol | compact|logit_reg |     0.4682 |  0.1521 | 0.8555 |          0.2075 | -12.7514 |   0      |
|   5 | mkt_high_vol | compact|logit     |     0.4703 |  0.1524 | 0.855  |          0.2038 | -11.356  |   0      |
|   5 | mkt_high_vol | full|hgb          |     0.4845 |  0.1569 | 0.8458 |          0.1799 | -10.3793 |   0      |
|   5 | mkt_high_vol | compact|hgb       |     0.4859 |  0.1575 | 0.8438 |          0.1774 | -10.4764 |   0      |
|   5 | mkt_high_vol | full|logit_reg    |     0.4869 |  0.1564 | 0.8483 |          0.1758 |  -9.668  |   0      |
|   5 | mkt_high_vol | full|rf           |     0.4988 |  0.1622 | 0.8416 |          0.1556 | -13.3085 |   0      |
|   5 | mkt_high_vol | state|logit       |     0.5893 |  0.2014 | 0.7582 |          0.0025 |  -3.3789 |   0.0004 |
|   5 | mkt_high_vol | persist_markov    |     0.5907 |  0.2019 | 0.7554 |          0      | nan      | nan      |
|   5 | mkt_high_vol | uncond            |     0.6932 |  0.25   | 0.3959 |         -0.1735 |   9.6333 |   1      |

### `mkt_high_vol`, h = 25

Base rate 53.6%, 6,804 daily observations (~272 non-overlapping).

|   h | target       | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:-------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|  25 | mkt_high_vol | compact|logit_reg |     0.4616 |  0.1515 | 0.8575 |          0.2474 |  -6.2289 |   0      |
|  25 | mkt_high_vol | compact|logit     |     0.4815 |  0.1562 | 0.8544 |          0.2149 |  -4.6674 |   0      |
|  25 | mkt_high_vol | full|rf           |     0.485  |  0.1583 | 0.8414 |          0.2093 |  -7.2256 |   0      |
|  25 | mkt_high_vol | compact|hgb       |     0.4917 |  0.1583 | 0.8526 |          0.1983 |  -4.2833 |   0      |
|  25 | mkt_high_vol | full|hgb          |     0.4992 |  0.162  | 0.8482 |          0.186  |  -4.0328 |   0      |
|  25 | mkt_high_vol | full|logit_reg    |     0.5484 |  0.1828 | 0.8124 |          0.1059 |  -2.0435 |   0.0205 |
|  25 | mkt_high_vol | state|logit       |     0.6103 |  0.2077 | 0.7322 |          0.0049 |  -4.354  |   0      |
|  25 | mkt_high_vol | persist_markov    |     0.6133 |  0.2085 | 0.7282 |          0      | nan      | nan      |
|  25 | mkt_high_vol | uncond            |     0.6932 |  0.25   | 0.3734 |         -0.1302 |   2.8088 |   0.9975 |

### `mkt_negative_return`, h = 5

Base rate 42.8%, 6,804 daily observations (~1,360 non-overlapping).

|   h | target              | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:--------------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|   5 | mkt_negative_return | persist_markov    |     0.6827 |  0.2448 | 0.5222 |          0      | nan      | nan      |
|   5 | mkt_negative_return | state|logit       |     0.6828 |  0.2448 | 0.5217 |         -0      |   0.1294 |   0.5515 |
|   5 | mkt_negative_return | uncond            |     0.6838 |  0.2453 | 0.4855 |         -0.0016 |   0.8782 |   0.8101 |
|   5 | mkt_negative_return | full|rf           |     0.6849 |  0.2458 | 0.5167 |         -0.0031 |   1.0607 |   0.8556 |
|   5 | mkt_negative_return | compact|logit_reg |     0.6996 |  0.2521 | 0.5052 |         -0.0247 |   3.897  |   1      |
|   5 | mkt_negative_return | compact|logit     |     0.7059 |  0.2544 | 0.5039 |         -0.0339 |   4.4423 |   1      |
|   5 | mkt_negative_return | full|logit_reg    |     0.7153 |  0.2572 | 0.5282 |         -0.0477 |   4.4086 |   1      |
|   5 | mkt_negative_return | full|hgb          |     0.7223 |  0.2615 | 0.5037 |         -0.058  |   5.8688 |   1      |
|   5 | mkt_negative_return | compact|hgb       |     0.7296 |  0.2648 | 0.4875 |         -0.0686 |   7.0153 |   1      |

### `mkt_negative_return`, h = 25

Base rate 37.3%, 6,804 daily observations (~272 non-overlapping).

|   h | target              | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:--------------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|  25 | mkt_negative_return | uncond            |     0.6646 |  0.2358 | 0.4566 |          0.0066 |  -0.7744 |   0.2194 |
|  25 | mkt_negative_return | state|logit       |     0.6687 |  0.2376 | 0.4852 |          0.0005 |  -2.3657 |   0.009  |
|  25 | mkt_negative_return | persist_markov    |     0.669  |  0.2378 | 0.487  |          0      | nan      | nan      |
|  25 | mkt_negative_return | full|rf           |     0.6782 |  0.2418 | 0.479  |         -0.0138 |   1.1696 |   0.8789 |
|  25 | mkt_negative_return | compact|logit_reg |     0.7402 |  0.2615 | 0.4724 |         -0.1064 |   3.144  |   0.9992 |
|  25 | mkt_negative_return | compact|logit     |     0.7697 |  0.2672 | 0.4712 |         -0.1504 |   3.2867 |   0.9995 |
|  25 | mkt_negative_return | full|hgb          |     0.7867 |  0.2793 | 0.471  |         -0.1759 |   4.642  |   1      |
|  25 | mkt_negative_return | compact|hgb       |     0.8365 |  0.2917 | 0.4552 |         -0.2504 |   5.0977 |   1      |
|  25 | mkt_negative_return | full|logit_reg    |     0.8378 |  0.2799 | 0.5007 |         -0.2523 |   3.5743 |   0.9998 |

### Reading M2

* **mkt_high_vol** -- h=5: compact|logit_reg skill +20.7% (p=0.000), h=25: compact|logit_reg skill +24.7% (p=0.000)
* **mkt_negative_return** -- h=5: state|logit skill -0.0% (p=0.551), h=25: uncond skill +0.7% (p=0.219)

## M3. Does the skill survive on days when the regime changes?

|   h | subset         | model             |   share_of_days |   log_loss |   skill_vs_base |    auc |
|----:|:---------------|:------------------|----------------:|-----------:|----------------:|-------:|
|   5 | state persists | persist_markov    |          0.9536 |     0.0286 |          0      | 1      |
|   5 | state changes  | persist_markov    |          0.0464 |     3.3321 |          0      | 0      |
|   5 | state persists | hmm_analytic      |          0.9536 |     0.0311 |         -0.0888 | 1      |
|   5 | state changes  | hmm_analytic      |          0.0464 |     2.8143 |          0.1554 | 0      |
|   5 | state persists | compact|logit_reg |          0.9536 |     0.0632 |         -1.2126 | 0.9999 |
|   5 | state changes  | compact|logit_reg |          0.0464 |     2.1262 |          0.3619 | 0.0035 |
|   5 | state persists | full|hgb          |          0.9536 |     0.0483 |         -0.6916 | 0.9987 |
|   5 | state changes  | full|hgb          |          0.0464 |     1.8594 |          0.442  | 0.2274 |
|   5 | state persists | full|rf           |          0.9536 |     0.1899 |         -5.6499 | 0.9981 |
|   5 | state changes  | full|rf           |          0.0464 |     1.0343 |          0.6896 | 0.2009 |
|  25 | state persists | persist_markov    |          0.8643 |     0.1431 |          0      | 1      |
|  25 | state changes  | persist_markov    |          0.1357 |     2.1155 |          0      | 0      |
|  25 | state persists | hmm_analytic      |          0.8643 |     0.1263 |          0.1176 | 1      |
|  25 | state changes  | hmm_analytic      |          0.1357 |     1.9951 |          0.0569 | 0      |
|  25 | state persists | compact|logit_reg |          0.8643 |     0.2037 |         -0.4233 | 0.9844 |
|  25 | state changes  | compact|logit_reg |          0.1357 |     1.5211 |          0.2809 | 0.0916 |
|  25 | state persists | full|hgb          |          0.8643 |     0.2765 |         -0.9319 | 0.9592 |
|  25 | state changes  | full|hgb          |          0.1357 |     1.5763 |          0.2549 | 0.3018 |
|  25 | state persists | full|rf           |          0.8643 |     0.255  |         -0.7821 | 0.99   |
|  25 | state changes  | full|rf           |          0.1357 |     1.0547 |          0.5014 | 0.2496 |

The 'state changes' rows are the ones that matter: they are the days on which a forecast could have been useful. A model that is only good at saying 'the same as today' scores well overall and badly here.
