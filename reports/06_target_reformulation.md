# Experiment 6 - Reformulating the target

## R3. Forward volatility state, across all horizons

Target: realised S&P volatility over `(t, t+h]` above its training-fold median. Benchmark unchanged -- the regime-conditional frequency, i.e. today's state and nothing else.

|   h | model             |   log_loss |    auc |   skill_vs_base |     dm_t |   dm_p |
|----:|:------------------|-----------:|-------:|----------------:|---------:|-------:|
|   5 | compact|logit_reg |     0.4682 | 0.8555 |          0.2075 | -12.7514 |      0 |
|  10 | compact|logit_reg |     0.4278 | 0.8812 |          0.2713 |  -9.9857 |      0 |
|  15 | compact|logit_reg |     0.4395 | 0.8712 |          0.2565 |  -7.743  |      0 |
|  20 | compact|logit_reg |     0.4484 | 0.8649 |          0.2537 |  -6.6248 |      0 |
|  25 | compact|logit_reg |     0.4616 | 0.8575 |          0.2474 |  -6.2289 |      0 |

Full table:

| model             |      5 |     10 |     15 |     20 |     25 |
|:------------------|-------:|-------:|-------:|-------:|-------:|
| compact|hgb       |  0.177 |  0.237 |  0.183 |  0.187 |  0.198 |
| compact|logit     |  0.204 |  0.263 |  0.24  |  0.235 |  0.215 |
| compact|logit_reg |  0.207 |  0.271 |  0.257 |  0.254 |  0.247 |
| full|hgb          |  0.18  |  0.227 |  0.183 |  0.181 |  0.186 |
| full|logit_reg    |  0.176 |  0.224 |  0.167 |  0.145 |  0.106 |
| full|rf           |  0.156 |  0.204 |  0.193 |  0.198 |  0.209 |
| persist_markov    |  0     |  0     |  0     |  0     |  0     |
| state|logit       |  0.002 |  0.003 |  0.003 |  0.003 |  0.005 |
| uncond            | -0.173 | -0.181 | -0.173 | -0.154 | -0.13  |

## R1 vs R2. Same event, different loss

| model                 |      5 |     10 |     15 |     20 |     25 |
|:----------------------|-------:|-------:|-------:|-------:|-------:|
| binary_reference      | 0.159  | 0.225  | 0.2464 | 0.2912 | 0.3824 |
| continuous_ridge_a1   | 0.4422 | 0.4419 | 0.3956 | 0.4374 | 0.4798 |
| continuous_ridge_a10  | 0.4414 | 0.4413 | 0.395  | 0.4367 | 0.4791 |
| continuous_ridge_a100 | 0.4337 | 0.4357 | 0.3899 | 0.4302 | 0.4726 |

Log loss, lower is better. `binary_reference` is the regularised logistic classifier from experiment 3 on the same design matrix; the `continuous_ridge` rows regress the forward risk-appetite score and convert to a probability through a Gaussian predictive distribution.

| model                 |       5 |      10 |      15 |      20 |      25 |
|:----------------------|--------:|--------:|--------:|--------:|--------:|
| binary_reference      |  0      |  0      |  0      |  0      |  0      |
| continuous_ridge_a1   | -1.7813 | -0.9637 | -0.6055 | -0.5024 | -0.2548 |
| continuous_ridge_a10  | -1.7759 | -0.9611 | -0.603  | -0.4998 | -0.2528 |
| continuous_ridge_a100 | -1.7279 | -0.9365 | -0.5823 | -0.4776 | -0.2359 |

Skill relative to the binary classifier.
