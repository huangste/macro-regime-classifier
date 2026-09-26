# Experiment 5 - Sensitivity and ablation

All arms: horizon 25 days, target `any_in_window`, expanding walk-forward purged by the horizon, benchmark `persist_markov`. Cells are **skill versus that benchmark** -- positive means the method removed some of the benchmark's log loss.

## A. Number of regimes

| arm   |   base_rate |   compact|logit_reg |   compact|hgb |   hmm_analytic |   state|logit |   uncond |
|:------|------------:|--------------------:|--------------:|---------------:|--------------:|---------:|
| K=2   |      0.5394 |              0.2395 |        0.093  |         0.0707 |        0.0576 |  -1.4818 |
| K=3   |      0.4322 |              0.2795 |        0.0806 |         0.0988 |        0.0642 |  -1.5189 |
| K=4   |      0.4972 |              0.1318 |        0.0538 |         0.1179 |        0.066  |  -1.4588 |
| K=5   |      0.4061 |              0.2678 |        0.1581 |         0.1899 |        0.1089 |  -1.2024 |

Base rates differ by K, so compare skill columns, not log losses.

## B. Labelling estimator

| arm       |   base_rate |   compact|logit_reg |   state|logit |   compact|hgb |   uncond |
|:----------|------------:|--------------------:|--------------:|--------------:|---------:|
| hmm (K=4) |      0.4972 |              0.1318 |        0.066  |        0.0538 |  -1.4588 |
| gmm (K=4) |      0.4361 |              0.1727 |        0.0814 |        0.033  |  -0.8035 |

The Gaussian mixture has no time dimension, so its state estimate flickers and its transition matrix is estimated from a much noisier label series. `hmm_analytic` has no mixture counterpart and is omitted from this table.

## C. Sample period

| arm                           |   base_rate |   compact|logit_reg |   hmm_analytic |   state|logit |   compact|hgb |   uncond |
|:------------------------------|------------:|--------------------:|---------------:|--------------:|--------------:|---------:|
| core_1990 (1990-)             |      0.4972 |              0.1318 |         0.1179 |        0.066  |        0.0538 |  -1.4588 |
| full_2000 (2000-, incl. gold) |      0.2086 |              0.2347 |         0.1054 |        0.0773 |        0.1077 |  -0.5191 |

## D. Class weighting

| model                  |   log_loss |   brier |    auc |   skill_vs_base |
|:-----------------------|-----------:|--------:|-------:|----------------:|
| compact|logit_reg      |     0.2612 |  0.0769 | 0.9538 |          0.1318 |
| persist_markov         |     0.3008 |  0.0796 | 0.9429 |          0      |
| compact|logit_balanced |     0.3414 |  0.0957 | 0.9375 |         -0.1348 |

`class_weight='balanced'` is what the original notebook used. It leaves ranking (AUC) roughly intact while badly damaging log loss and Brier score, because re-weighting the classes deliberately biases the predicted probabilities away from the base rate. If the output is a probability rather than a decision, it should not be on.

## E. Which features carry the signal?

| arm                   |   base_rate |   compact|logit_reg |   compact|hgb |
|:----------------------|------------:|--------------------:|--------------:|
| all features          |      0.4972 |              0.1318 |        0.0538 |
| only volatility       |      0.4972 |              0.1433 |        0.1136 |
| only credit           |      0.4972 |              0.1542 |        0.0706 |
| only momentum         |      0.4972 |              0.1075 |        0.0479 |
| only rates            |      0.4972 |              0.1004 |       -0.0005 |
| all except volatility |      0.4972 |              0.1012 |        0.062  |
| all except credit     |      0.4972 |              0.113  |        0.0332 |
| all except momentum   |      0.4972 |              0.1577 |        0.0572 |
| all except rates      |      0.4972 |              0.1824 |        0.0987 |

The labeller keeps its full feature set in every arm, so the regime definition is identical across rows and only the forecaster's inputs change.
