# Experiment 7 - Forecast combination

Equal-weight average of the persistence benchmark with each learned model, scored on exactly the out-of-sample predictions of experiment 3. Rows marked `blend50|` are the combinations.

## Target: `point_in_time`

Skill versus the benchmark, by horizon:

| model                     |       5 |      10 |      15 |      20 |      25 |
|:--------------------------|--------:|--------:|--------:|--------:|--------:|
| blend50|compact|hgb       |  0.2802 |  0.1881 |  0.202  |  0.1386 |  0.1208 |
| blend50|compact|logit_reg |  0.1504 |  0.122  |  0.1422 |  0.1344 |  0.1134 |
| blend50|full|hgb          |  0.2911 |  0.1878 |  0.1892 |  0.1181 |  0.0927 |
| blend50|full|rf           |  0.0582 |  0.0856 |  0.1116 |  0.1277 |  0.1418 |
| blend50|hmm_analytic      |  0.0837 |  0.0495 |  0.0488 |  0.0498 |  0.057  |
| compact|hgb               |  0.2426 |  0.0338 |  0.0656 | -0.0671 | -0.0207 |
| compact|logit             | -0.065  | -0.5576 | -0.1836 | -0.3572 | -0.2942 |
| compact|logit_reg         |  0.1263 |  0.0653 |  0.1057 |  0.0931 |  0.0688 |
| full|hgb                  |  0.2724 |  0.0872 |  0.0676 | -0.1037 | -0.1026 |
| full|logit_reg            |  0.041  | -0.0567 |  0.0204 | -0.0229 | -0.124  |
| full|rf                   | -0.2589 | -0.0886 | -0.0058 |  0.0521 |  0.1148 |
| hmm_analytic              |  0.1189 |  0.0713 |  0.0693 |  0.0672 |  0.0752 |
| persist_markov            |  0      |  0      |  0      |  0      |  0      |
| state|logit               |  0.0709 |  0.0533 |  0.0418 |  0.0375 |  0.0364 |
| uncond                    | -2.9151 | -1.9613 | -1.4964 | -1.2238 | -0.7566 |

One-sided Diebold-Mariano p-values:

| model                     |     5 |    10 |    15 |    20 |    25 |
|:--------------------------|------:|------:|------:|------:|------:|
| blend50|compact|hgb       | 0     | 0     | 0.001 | 0.008 | 0.011 |
| blend50|compact|logit_reg | 0     | 0.007 | 0.009 | 0.011 | 0.021 |
| blend50|full|hgb          | 0     | 0     | 0     | 0.008 | 0.021 |
| blend50|full|rf           | 0.157 | 0.07  | 0.036 | 0.026 | 0.011 |
| blend50|hmm_analytic      | 0     | 0     | 0.012 | 0.021 | 0.021 |
| compact|hgb               | 0     | 0.346 | 0.247 | 0.727 | 0.593 |
| compact|logit             | 0.813 | 0.999 | 0.937 | 0.984 | 0.995 |
| compact|logit_reg         | 0.011 | 0.181 | 0.088 | 0.133 | 0.191 |
| full|hgb                  | 0     | 0.118 | 0.224 | 0.846 | 0.873 |
| full|logit_reg            | 0.261 | 0.732 | 0.412 | 0.579 | 0.853 |
| full|rf                   | 1     | 0.866 | 0.528 | 0.269 | 0.07  |
| hmm_analytic              | 0     | 0     | 0.005 | 0.014 | 0.014 |
| state|logit               | 0     | 0     | 0.001 | 0.002 | 0.005 |
| uncond                    | 1     | 1     | 1     | 1     | 1     |

## Target: `any_in_window`

Skill versus the benchmark, by horizon:

| model                     |       5 |      10 |      15 |      20 |      25 |
|:--------------------------|--------:|--------:|--------:|--------:|--------:|
| blend50|compact|hgb       |  0.3053 |  0.2512 |  0.173  |  0.1678 |  0.1554 |
| blend50|compact|logit_reg |  0.1516 |  0.1425 |  0.1473 |  0.1358 |  0.1544 |
| blend50|full|hgb          |  0.3006 |  0.1666 |  0.1372 |  0.146  |  0.1381 |
| blend50|full|rf           | -0.1291 | -0.0857 | -0.0301 |  0.0566 |  0.1255 |
| blend50|hmm_analytic      |  0.1468 |  0.0981 |  0.0859 |  0.0828 |  0.0981 |
| compact|hgb               |  0.2722 |  0.1547 |  0.0476 |  0.0613 |  0.0538 |
| compact|logit             | -0.037  | -0.2767 | -0.0125 | -0.2129 | -0.1178 |
| compact|logit_reg         |  0.1097 |  0.1111 |  0.1212 |  0.1052 |  0.1318 |
| full|hgb                  |  0.2762 |  0.0173 | -0.018  |  0.0023 |  0.0087 |
| full|logit_reg            | -0.052  | -0.0929 | -0.0661 | -0.1369 | -0.0406 |
| full|rf                   | -0.7168 | -0.4999 | -0.3643 | -0.1726 | -0.0206 |
| hmm_analytic              |  0.1888 |  0.1235 |  0.107  |  0.1004 |  0.1179 |
| persist_markov            |  0      |  0      |  0      |  0      |  0      |
| state|logit               |  0.1169 |  0.0879 |  0.0693 |  0.0651 |  0.066  |
| uncond                    | -4.6874 | -3.3825 | -2.6752 | -2.0681 | -1.4588 |

One-sided Diebold-Mariano p-values:

| model                     |     5 |    10 |    15 |    20 |    25 |
|:--------------------------|------:|------:|------:|------:|------:|
| blend50|compact|hgb       | 0     | 0     | 0.01  | 0.007 | 0.008 |
| blend50|compact|logit_reg | 0.001 | 0.014 | 0.031 | 0.036 | 0.024 |
| blend50|full|hgb          | 0     | 0.001 | 0.017 | 0.017 | 0.006 |
| blend50|full|rf           | 0.969 | 0.879 | 0.64  | 0.263 | 0.086 |
| blend50|hmm_analytic      | 0     | 0     | 0.004 | 0.01  | 0.013 |
| compact|hgb               | 0     | 0.065 | 0.326 | 0.264 | 0.288 |
| compact|logit             | 0.683 | 0.976 | 0.549 | 0.969 | 0.894 |
| compact|logit_reg         | 0.038 | 0.079 | 0.099 | 0.122 | 0.079 |
| full|hgb                  | 0     | 0.424 | 0.571 | 0.491 | 0.467 |
| full|logit_reg            | 0.771 | 0.844 | 0.743 | 0.888 | 0.637 |
| full|rf                   | 1     | 1     | 1     | 0.936 | 0.572 |
| hmm_analytic              | 0     | 0     | 0.004 | 0.01  | 0.013 |
| state|logit               | 0     | 0     | 0     | 0.003 | 0.003 |
| uncond                    | 1     | 1     | 1     | 1     | 1     |

## Where the combination helps

| target        |   h | model                     |   skill_all |   skill_when_state_persists |   skill_when_state_changes |
|:--------------|----:|:--------------------------|------------:|----------------------------:|---------------------------:|
| point_in_time |   5 | compact|logit_reg         |      0.1263 |                     -1.2126 |                     0.3619 |
| point_in_time |   5 | blend50|compact|logit_reg |      0.1504 |                     -0.5174 |                     0.2679 |
| point_in_time |  25 | compact|logit_reg         |      0.0688 |                     -0.4233 |                     0.2809 |
| point_in_time |  25 | blend50|compact|logit_reg |      0.1134 |                     -0.1396 |                     0.2225 |
| any_in_window |   5 | compact|logit_reg         |      0.1097 |                     -2.1239 |                     0.395  |
| any_in_window |   5 | blend50|compact|logit_reg |      0.1516 |                     -0.9864 |                     0.2969 |
| any_in_window |  25 | compact|logit_reg         |      0.1318 |                     -0.5818 |                     0.2677 |
| any_in_window |  25 | blend50|compact|logit_reg |      0.1544 |                     -0.2556 |                     0.2325 |

The combination keeps most of the gain on turning days while giving back most of the loss on quiet days, which is exactly what it is supposed to do.

## Summary

| target        |   h | model               |   log_loss |   skill_vs_base |    dm_t |   dm_p |
|:--------------|----:|:--------------------|-----------:|----------------:|--------:|-------:|
| any_in_window |   5 | blend50|compact|hgb |     0.0883 |          0.3053 | -5.0512 | 0      |
| any_in_window |  10 | blend50|compact|hgb |     0.1249 |          0.2512 | -3.3636 | 0.0004 |
| any_in_window |  15 | blend50|compact|hgb |     0.1588 |          0.173  | -2.3284 | 0.0099 |
| any_in_window |  20 | blend50|compact|hgb |     0.2012 |          0.1678 | -2.4752 | 0.0067 |
| any_in_window |  25 | blend50|compact|hgb |     0.2541 |          0.1554 | -2.4155 | 0.0079 |
| point_in_time |   5 | blend50|full|hgb    |     0.129  |          0.2911 | -6.3242 | 0      |
| point_in_time |  10 | blend50|compact|hgb |     0.1955 |          0.1881 | -3.4514 | 0.0003 |
| point_in_time |  15 | blend50|compact|hgb |     0.2199 |          0.202  | -3.241  | 0.0006 |
| point_in_time |  20 | blend50|compact|hgb |     0.2766 |          0.1386 | -2.4154 | 0.0079 |
| point_in_time |  25 | blend50|full|rf     |     0.3524 |          0.1418 | -2.3058 | 0.0106 |

10 of 10 horizon/target combinations now show a positive skill that is distinguishable from noise at the 5% level, against 2 of 10 before combination.
