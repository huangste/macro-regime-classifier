# Experiment 3 - Out-of-sample forecasting benchmark

Sample 1990-12-28 to 2026-09-24, 8,929 trading days, 61 features. Expanding walk-forward, 8-year initial training window, 1-year test blocks, labeller and models refitted in every fold.

## Target: `point_in_time`

P(risk-off **on day t+h**). This decays towards the base rate as h grows: a single day far in the future is close to a draw from the stationary distribution.

### h = 5 trading days

Pooled out-of-sample: 6,804 daily observations, about 1,360 non-overlapping, base rate 38.3%. 27 folds from 1998-11-25 to 2026-03-25.

|   h | target        | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:--------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|   5 | point_in_time | full|hgb          |     0.1324 |  0.0372 | 0.9893 |          0.2724 |  -4.3301 |   0      |
|   5 | point_in_time | compact|hgb       |     0.1378 |  0.0385 | 0.9881 |          0.2426 |  -3.5787 |   0.0002 |
|   5 | point_in_time | compact|logit_reg |     0.159  |  0.0439 | 0.9826 |          0.1263 |  -2.2906 |   0.011  |
|   5 | point_in_time | hmm_analytic      |     0.1604 |  0.0405 | 0.9807 |          0.1189 |  -5.8939 |   0      |
|   5 | point_in_time | state|logit       |     0.1691 |  0.0418 | 0.9777 |          0.0709 |  -5.3842 |   0      |
|   5 | point_in_time | full|logit_reg    |     0.1745 |  0.0489 | 0.9814 |          0.041  |  -0.6391 |   0.2614 |
|   5 | point_in_time | persist_markov    |     0.182  |  0.0436 | 0.9711 |          0      | nan      | nan      |
|   5 | point_in_time | compact|logit     |     0.1938 |  0.0522 | 0.9786 |         -0.065  |   0.889  |   0.813  |
|   5 | point_in_time | full|rf           |     0.2291 |  0.0616 | 0.985  |         -0.2589 |   3.302  |   0.9995 |
|   5 | point_in_time | uncond            |     0.7125 |  0.2558 | 0.5176 |         -2.9151 |  20.5506 |   1      |

### h = 10 trading days

Pooled out-of-sample: 6,804 daily observations, about 680 non-overlapping, base rate 39.3%. 27 folds from 1998-11-25 to 2026-03-25.

|   h | target        | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:--------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|  10 | point_in_time | full|hgb          |     0.2197 |  0.062  | 0.9722 |          0.0872 |  -1.1827 |   0.1185 |
|  10 | point_in_time | hmm_analytic      |     0.2236 |  0.0594 | 0.963  |          0.0713 |  -3.6945 |   0.0001 |
|  10 | point_in_time | compact|logit_reg |     0.225  |  0.0674 | 0.9696 |          0.0653 |  -0.9116 |   0.181  |
|  10 | point_in_time | state|logit       |     0.2279 |  0.0599 | 0.9611 |          0.0533 |  -4.3277 |   0      |
|  10 | point_in_time | compact|hgb       |     0.2326 |  0.0664 | 0.9695 |          0.0338 |  -0.3948 |   0.3465 |
|  10 | point_in_time | persist_markov    |     0.2407 |  0.0624 | 0.9559 |          0      | nan      | nan      |
|  10 | point_in_time | full|logit_reg    |     0.2544 |  0.0755 | 0.964  |         -0.0567 |   0.6201 |   0.7324 |
|  10 | point_in_time | full|rf           |     0.2621 |  0.0743 | 0.9736 |         -0.0886 |   1.1055 |   0.8655 |
|  10 | point_in_time | compact|logit     |     0.375  |  0.1003 | 0.9313 |         -0.5576 |   3.1339 |   0.9991 |
|  10 | point_in_time | uncond            |     0.7129 |  0.2566 | 0.5447 |         -1.9613 |  13.3486 |   1      |

### h = 15 trading days

Pooled out-of-sample: 6,804 daily observations, about 453 non-overlapping, base rate 38.5%. 27 folds from 1998-11-25 to 2026-03-25.

|   h | target        | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:--------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|  15 | point_in_time | compact|logit_reg |     0.2464 |  0.0744 | 0.9621 |          0.1057 |  -1.3514 |   0.0883 |
|  15 | point_in_time | hmm_analytic      |     0.2564 |  0.0688 | 0.9503 |          0.0693 |  -2.5583 |   0.0053 |
|  15 | point_in_time | full|hgb          |     0.2569 |  0.0763 | 0.9619 |          0.0676 |  -0.7576 |   0.2244 |
|  15 | point_in_time | compact|hgb       |     0.2574 |  0.0751 | 0.959  |          0.0656 |  -0.6833 |   0.2472 |
|  15 | point_in_time | state|logit       |     0.264  |  0.0695 | 0.9476 |          0.0418 |  -3.2105 |   0.0007 |
|  15 | point_in_time | full|logit_reg    |     0.2699 |  0.0823 | 0.9552 |          0.0204 |  -0.2237 |   0.4115 |
|  15 | point_in_time | persist_markov    |     0.2755 |  0.0715 | 0.9421 |          0      | nan      | nan      |
|  15 | point_in_time | full|rf           |     0.2771 |  0.0802 | 0.968  |         -0.0058 |   0.071  |   0.5283 |
|  15 | point_in_time | compact|logit     |     0.3261 |  0.0975 | 0.9367 |         -0.1836 |   1.5295 |   0.9369 |
|  15 | point_in_time | uncond            |     0.6877 |  0.2457 | 0.5711 |         -1.4964 |  10.6895 |   1      |

### h = 20 trading days

Pooled out-of-sample: 6,804 daily observations, about 340 non-overlapping, base rate 38.0%. 27 folds from 1998-11-25 to 2026-03-25.

|   h | target        | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:--------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|  20 | point_in_time | compact|logit_reg |     0.2912 |  0.0907 | 0.9471 |          0.0931 |  -1.1102 |   0.1335 |
|  20 | point_in_time | hmm_analytic      |     0.2995 |  0.0822 | 0.9301 |          0.0672 |  -2.2056 |   0.0137 |
|  20 | point_in_time | full|rf           |     0.3043 |  0.0902 | 0.9577 |          0.0521 |  -0.6173 |   0.2685 |
|  20 | point_in_time | state|logit       |     0.309  |  0.0829 | 0.9295 |          0.0375 |  -2.8108 |   0.0025 |
|  20 | point_in_time | persist_markov    |     0.321  |  0.0844 | 0.9245 |          0      | nan      | nan      |
|  20 | point_in_time | full|logit_reg    |     0.3284 |  0.0994 | 0.9354 |         -0.0229 |   0.1987 |   0.5788 |
|  20 | point_in_time | compact|hgb       |     0.3426 |  0.0979 | 0.932  |         -0.0671 |   0.6045 |   0.7273 |
|  20 | point_in_time | full|hgb          |     0.3543 |  0.1028 | 0.9305 |         -0.1037 |   1.0191 |   0.8459 |
|  20 | point_in_time | compact|logit     |     0.4357 |  0.1204 | 0.8956 |         -0.3572 |   2.1564 |   0.9845 |
|  20 | point_in_time | uncond            |     0.714  |  0.2571 | 0.53   |         -1.2238 |   8.446  |   1      |

### h = 25 trading days

Pooled out-of-sample: 6,804 daily observations, about 272 non-overlapping, base rate 40.9%. 27 folds from 1998-11-25 to 2026-03-25.

|   h | target        | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:--------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|  25 | point_in_time | full|rf           |     0.3635 |  0.1123 | 0.9316 |          0.1148 |  -1.4727 |   0.0704 |
|  25 | point_in_time | hmm_analytic      |     0.3798 |  0.1114 | 0.9085 |          0.0752 |  -2.1903 |   0.0143 |
|  25 | point_in_time | compact|logit_reg |     0.3824 |  0.1225 | 0.9105 |          0.0688 |  -0.8757 |   0.1906 |
|  25 | point_in_time | state|logit       |     0.3957 |  0.1126 | 0.9049 |          0.0364 |  -2.598  |   0.0047 |
|  25 | point_in_time | persist_markov    |     0.4107 |  0.1145 | 0.9    |          0      | nan      | nan      |
|  25 | point_in_time | compact|hgb       |     0.4192 |  0.1257 | 0.9051 |         -0.0207 |   0.2364 |   0.5935 |
|  25 | point_in_time | full|hgb          |     0.4528 |  0.1383 | 0.8984 |         -0.1026 |   1.14   |   0.8729 |
|  25 | point_in_time | full|logit_reg    |     0.4616 |  0.1419 | 0.885  |         -0.124  |   1.0496 |   0.853  |
|  25 | point_in_time | compact|logit     |     0.5315 |  0.1633 | 0.8536 |         -0.2942 |   2.5991 |   0.9953 |
|  25 | point_in_time | uncond            |     0.7214 |  0.2615 | 0.5404 |         -0.7566 |   5.9626 |   1      |

## Target: `any_in_window`

P(**at least one** risk-off day in (t, t+h]). This rises with h and is the quantity a risk manager actually wants.

### h = 5 trading days

Pooled out-of-sample: 6,804 daily observations, about 1,360 non-overlapping, base rate 40.4%. 27 folds from 1998-11-25 to 2026-03-25.

|   h | target        | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:--------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|   5 | any_in_window | full|hgb          |     0.092  |  0.025  | 0.9937 |          0.2762 |  -3.7821 |   0.0001 |
|   5 | any_in_window | compact|hgb       |     0.0925 |  0.025  | 0.9938 |          0.2722 |  -3.5536 |   0.0002 |
|   5 | any_in_window | hmm_analytic      |     0.1031 |  0.0252 | 0.9887 |          0.1888 |  -6.0665 |   0      |
|   5 | any_in_window | state|logit       |     0.1122 |  0.0264 | 0.9868 |          0.1169 |  -5.2583 |   0      |
|   5 | any_in_window | compact|logit_reg |     0.1131 |  0.0282 | 0.9899 |          0.1097 |  -1.7709 |   0.0383 |
|   5 | any_in_window | persist_markov    |     0.1271 |  0.028  | 0.9812 |          0      | nan      | nan      |
|   5 | any_in_window | compact|logit     |     0.1318 |  0.0317 | 0.9866 |         -0.037  |   0.4769 |   0.6833 |
|   5 | any_in_window | full|logit_reg    |     0.1337 |  0.0349 | 0.9884 |         -0.052  |   0.7433 |   0.7713 |
|   5 | any_in_window | full|rf           |     0.2182 |  0.0567 | 0.9905 |         -0.7168 |   7.6167 |   1      |
|   5 | any_in_window | uncond            |     0.7228 |  0.2611 | 0.5295 |         -4.6874 |  25.8998 |   1      |

### h = 10 trading days

Pooled out-of-sample: 6,804 daily observations, about 680 non-overlapping, base rate 43.3%. 27 folds from 1998-11-25 to 2026-03-25.

|   h | target        | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:--------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|  10 | any_in_window | compact|hgb       |     0.141  |  0.0357 | 0.9851 |          0.1547 |  -1.5109 |   0.0654 |
|  10 | any_in_window | hmm_analytic      |     0.1462 |  0.0379 | 0.9788 |          0.1235 |  -4.1006 |   0      |
|  10 | any_in_window | compact|logit_reg |     0.1483 |  0.0392 | 0.984  |          0.1111 |  -1.4088 |   0.0794 |
|  10 | any_in_window | state|logit       |     0.1521 |  0.0385 | 0.9774 |          0.0879 |  -4.3335 |   0      |
|  10 | any_in_window | full|hgb          |     0.1639 |  0.0414 | 0.9826 |          0.0173 |  -0.1918 |   0.424  |
|  10 | any_in_window | persist_markov    |     0.1668 |  0.0402 | 0.9716 |          0      | nan      | nan      |
|  10 | any_in_window | full|logit_reg    |     0.1823 |  0.0502 | 0.9803 |         -0.0929 |   1.012  |   0.8442 |
|  10 | any_in_window | compact|logit     |     0.213  |  0.054  | 0.9717 |         -0.2767 |   1.9729 |   0.9757 |
|  10 | any_in_window | full|rf           |     0.2502 |  0.0683 | 0.9796 |         -0.4999 |   4.9975 |   1      |
|  10 | any_in_window | uncond            |     0.7311 |  0.2659 | 0.557  |         -3.3825 |  17.5488 |   1      |

### h = 15 trading days

Pooled out-of-sample: 6,804 daily observations, about 453 non-overlapping, base rate 44.1%. 27 folds from 1998-11-25 to 2026-03-25.

|   h | target        | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:--------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|  15 | any_in_window | compact|logit_reg |     0.1688 |  0.0463 | 0.9783 |          0.1212 |  -1.2901 |   0.0985 |
|  15 | any_in_window | hmm_analytic      |     0.1715 |  0.0457 | 0.9686 |          0.107  |  -2.6705 |   0.0038 |
|  15 | any_in_window | state|logit       |     0.1787 |  0.0458 | 0.967  |          0.0693 |  -3.5026 |   0.0002 |
|  15 | any_in_window | compact|hgb       |     0.1829 |  0.0514 | 0.9772 |          0.0476 |  -0.4512 |   0.3259 |
|  15 | any_in_window | persist_markov    |     0.192  |  0.0474 | 0.9622 |          0      | nan      | nan      |
|  15 | any_in_window | compact|logit     |     0.1944 |  0.0533 | 0.9713 |         -0.0125 |   0.1219 |   0.5485 |
|  15 | any_in_window | full|hgb          |     0.1955 |  0.0541 | 0.9772 |         -0.018  |   0.1786 |   0.5709 |
|  15 | any_in_window | full|logit_reg    |     0.2047 |  0.0591 | 0.974  |         -0.0661 |   0.6515 |   0.7426 |
|  15 | any_in_window | full|rf           |     0.262  |  0.0728 | 0.9758 |         -0.3643 |   3.2966 |   0.9995 |
|  15 | any_in_window | uncond            |     0.7058 |  0.2552 | 0.6011 |         -2.6752 |  13.954  |   1      |

### h = 20 trading days

Pooled out-of-sample: 6,804 daily observations, about 340 non-overlapping, base rate 44.8%. 27 folds from 1998-11-25 to 2026-03-25.

|   h | target        | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:--------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|  20 | any_in_window | compact|logit_reg |     0.2163 |  0.061  | 0.965  |          0.1052 |  -1.1669 |   0.1216 |
|  20 | any_in_window | hmm_analytic      |     0.2175 |  0.059  | 0.9541 |          0.1004 |  -2.3132 |   0.0104 |
|  20 | any_in_window | state|logit       |     0.226  |  0.0589 | 0.9538 |          0.0651 |  -2.7808 |   0.0027 |
|  20 | any_in_window | compact|hgb       |     0.2269 |  0.0598 | 0.9681 |          0.0613 |  -0.6309 |   0.264  |
|  20 | any_in_window | full|hgb          |     0.2412 |  0.0663 | 0.9682 |          0.0023 |  -0.0223 |   0.4911 |
|  20 | any_in_window | persist_markov    |     0.2417 |  0.0603 | 0.9493 |          0      | nan      | nan      |
|  20 | any_in_window | full|logit_reg    |     0.2748 |  0.0791 | 0.9555 |         -0.1369 |   1.2171 |   0.8882 |
|  20 | any_in_window | full|rf           |     0.2834 |  0.0811 | 0.9662 |         -0.1726 |   1.5237 |   0.9362 |
|  20 | any_in_window | compact|logit     |     0.2932 |  0.0751 | 0.9494 |         -0.2129 |   1.8692 |   0.9692 |
|  20 | any_in_window | uncond            |     0.7416 |  0.2716 | 0.5686 |         -2.0681 |  10.5943 |   1      |

### h = 25 trading days

Pooled out-of-sample: 6,804 daily observations, about 272 non-overlapping, base rate 49.7%. 27 folds from 1998-11-25 to 2026-03-25.

|   h | target        | model             |   log_loss |   brier |    auc |   skill_vs_base |     dm_t |     dm_p |
|----:|:--------------|:------------------|-----------:|--------:|-------:|----------------:|---------:|---------:|
|  25 | any_in_window | compact|logit_reg |     0.2612 |  0.0769 | 0.9538 |          0.1318 |  -1.414  |   0.0787 |
|  25 | any_in_window | hmm_analytic      |     0.2654 |  0.0777 | 0.9482 |          0.1179 |  -2.2335 |   0.0128 |
|  25 | any_in_window | state|logit       |     0.281  |  0.0778 | 0.9473 |          0.066  |  -2.7881 |   0.0027 |
|  25 | any_in_window | compact|hgb       |     0.2846 |  0.0776 | 0.9557 |          0.0538 |  -0.5591 |   0.2881 |
|  25 | any_in_window | full|hgb          |     0.2982 |  0.0792 | 0.9547 |          0.0087 |  -0.0837 |   0.4667 |
|  25 | any_in_window | persist_markov    |     0.3008 |  0.0796 | 0.9429 |          0      | nan      | nan      |
|  25 | any_in_window | full|rf           |     0.307  |  0.0889 | 0.9604 |         -0.0206 |   0.1813 |   0.5719 |
|  25 | any_in_window | full|logit_reg    |     0.3131 |  0.0923 | 0.9476 |         -0.0406 |   0.3504 |   0.637  |
|  25 | any_in_window | compact|logit     |     0.3363 |  0.0904 | 0.9395 |         -0.1178 |   1.2456 |   0.8936 |
|  25 | any_in_window | uncond            |     0.7397 |  0.2716 | 0.6063 |         -1.4588 |   8.3097 |   1      |

## Does anything beat the persistence benchmark?

| target        |   h | model             |   log_loss |   skill_vs_base |    dm_t |   dm_p |
|:--------------|----:|:------------------|-----------:|----------------:|--------:|-------:|
| any_in_window |   5 | full|hgb          |     0.092  |          0.2762 | -3.7821 | 0.0001 |
| any_in_window |  10 | compact|hgb       |     0.141  |          0.1547 | -1.5109 | 0.0654 |
| any_in_window |  15 | compact|logit_reg |     0.1688 |          0.1212 | -1.2901 | 0.0985 |
| any_in_window |  20 | compact|logit_reg |     0.2163 |          0.1052 | -1.1669 | 0.1216 |
| any_in_window |  25 | compact|logit_reg |     0.2612 |          0.1318 | -1.414  | 0.0787 |
| point_in_time |   5 | full|hgb          |     0.1324 |          0.2724 | -4.3301 | 0      |
| point_in_time |  10 | full|hgb          |     0.2197 |          0.0872 | -1.1827 | 0.1185 |
| point_in_time |  15 | compact|logit_reg |     0.2464 |          0.1057 | -1.3514 | 0.0883 |
| point_in_time |  20 | compact|logit_reg |     0.2912 |          0.0931 | -1.1102 | 0.1335 |
| point_in_time |  25 | full|rf           |     0.3635 |          0.1148 | -1.4727 | 0.0704 |

`skill_vs_base` is the fraction of the persistence benchmark's log loss removed; `dm_t` is negative when the model has lower loss and `dm_p` is the one-sided p-value. A result is only interesting if the skill is positive **and** `dm_p` is small.

2 of 10 horizon/target combinations show a statistically distinguishable improvement over persistence.

## Calibration

`persist_markov` at h=25, any-in-window:

| bin     |    n |   mean_predicted |   observed_freq |
|:--------|-----:|-----------------:|----------------:|
| 0.0-0.1 | 2983 |            0.058 |           0.099 |
| 0.1-0.2 | 1062 |            0.183 |           0.325 |
| 0.9-1.0 | 2759 |            0.992 |           0.994 |

`compact|logit` at h=25, any-in-window:

| bin     |    n |   mean_predicted |   observed_freq |
|:--------|-----:|-----------------:|----------------:|
| 0.0-0.1 | 2949 |            0.028 |           0.114 |
| 0.1-0.2 |  424 |            0.179 |           0.328 |
| 0.2-0.4 |  239 |            0.31  |           0.301 |
| 0.4-0.5 |  176 |            0.43  |           0.261 |
| 0.5-0.6 |  146 |            0.562 |           0.24  |
| 0.6-0.8 |  106 |            0.682 |           0.358 |
| 0.8-0.9 |   83 |            0.819 |           0.663 |
| 0.9-1.0 | 2681 |            0.987 |           0.993 |
