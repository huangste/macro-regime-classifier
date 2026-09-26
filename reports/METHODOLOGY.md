# Methodology

## Two products, deliberately kept apart

**The historical classification** answers "what state was the market in?"  It
is fitted on the whole sample. That is not a bug: describing 2008 with the
benefit of hindsight is a smoothing problem, and using the full sample gives
the sharpest description. Everywhere it appears it is labelled as such.

**The forward forecast** answers "what state will the market be in?"  Nothing
in its construction may touch data after the forecast origin. The regime
definition, the feature scaling, the transition matrix and the classifier are
all refitted inside each walk-forward fold, and the number quoted for today is
produced by exactly the procedure that was scored out of sample.

Conflating these two is the single most common way a regime model comes to
look better than it is.

---

## Data

| block | series | source | from |
|---|---|---|---|
| Equity | S&P 500, Nasdaq 100, Nikkei 225, Russell 2000 | Yahoo | 1990 |
| Volatility | VIX | FRED `VIXCLS` | 1990 |
| Credit | Moody's Baa less 10y Treasury, Baa−Aaa | FRED `BAA10Y`, `DBAA`, `DAAA` | 1986 |
| Rates | 3m, 2y, 10y Treasury and two curve slopes | FRED | 1981 |
| FX | USD index, USD/JPY | Yahoo, FRED `DEXJPUS` | 1980 |
| Commodity | WTI crude (gold in the 2000 sample) | FRED `DCOILWTICO` | 1986 |

Everything is reindexed onto the S&P trading calendar and forward-filled
across foreign holidays with a five-session limit. The original notebook
concatenated series with different calendars and called `dropna()`, which
deleted rows silently, and truncated the whole panel to the 2000 start of the
Yahoo gold and oil futures. Sourcing oil from FRED and making gold optional
recovers ten years: **9,244 sessions from 1990** instead of 6,800 from 2000.

The original also downloaded HYG and then never put it in the panel, so the
model had no credit variable at all. Moody's Baa spread is used instead
because it is daily from 1986 rather than 2007.

**Known gap, April–July 2020.** WTI crude settled at −$37.63 on 20 April
2020. Oil features are built from log prices, so that single print makes them
undefined; the missing value then propagates through every 63-day window, and
the feature step drops 64 sessions (20 April – 20 July 2020) from the *whole*
sample — including the regime classification, which does not use oil at all.
It shows as a straight line through mid-2020 in the risk-appetite chart. This
is a defect, not a modelling choice, and has not yet been fixed.

---

## Features

Returns are divided by trailing realised volatility rather than used raw, so
that a −8% month means the same thing in 2017 as in 2008; volatility enters in
logs, which is far closer to Gaussian and matters for a Gaussian HMM. Added to
these are explicit stress variables the original lacked: VIX and its premium
over realised volatility, credit spreads and their changes, drawdown from a
rolling 252-day high, the stock/bond correlation, and small-cap relative
strength.

Every feature at row `t` uses only data observed at the close of `t`. This is
enforced by a test that corrupts the tail of the price panel and asserts that
no earlier feature value moves.

---

## Regime definition, and why the labels are now stable

An unsupervised mixture has no notion of which component is "state 0".
Component indices are an artefact of initialisation, so refitting after one
extra day of data can permute every label. Measured on the original setup,
**58% of historical days changed their risk-on/risk-off verdict when only the
random seed changed**, and the index holding the crisis component moved
between 0, 1, 2, 3 and 4 across seeds and data vintages.

Three changes fix this.

1. **Labels are never taken from the estimator's indexing.** Each state is
   scored on a risk-appetite functional whose signs are fixed in advance —
   equity trend and breadth positive, volatility negative, credit spreads
   negative, yen carry positive — computed from that state's mean in
   standardised feature space. States are then re-indexed in ascending score
   order. Canonical state 0 is always the most risk-averse and K−1 the most
   risk-seeking, for any seed, any K and any vintage. Variables whose risk
   interpretation is genuinely ambiguous (curve slope, the level change in
   yields, the stock/bond correlation) are excluded from the score and used
   only to describe states afterwards.

2. **The risk-on/risk-off split is derived, not hand-maintained.** States are
   already ordered, so the split is a prefix; it is placed at the widest gap
   in the ordered scores among prefixes covering between 15% and 55% of days.
   No lookup table to go stale.

3. **EM is started deterministically.** The Gaussian HMM likelihood on
   strongly autocorrelated data is badly multi-modal, and random restarts land
   in different partitions on different vintages — a source of instability
   that anchoring alone cannot repair. Seeding EM from the score partition
   removes the lottery entirely.

Measured effect:

| | seed flip rate | vintage flip rate | switches/year |
|---|---|---|---|
| original (mixture indices) | 57.6% | 31.6% | 6.0 |
| anchored mixture | 0% | 6.5% | 6.0 |
| **anchored HMM, score-initialised** | **0%** | **6.5%** | **3.5** |
| raw score quantiles | 0% | 1.2% | 30.9 |

The HMM also produces regimes rather than daily flicker: median episode length
51 sessions against 5 for the mixture, with 3% of episodes shorter than a week
against 50%.

K is chosen at 4 on vintage stability and on the requirement that states
separate realised risk-adjusted returns while leaving enough days in each to
estimate transitions. BIC is not used: it falls monotonically in K on data
this autocorrelated, because extra components absorb serial dependence the
model does not otherwise represent.

---

## Risk-appetite index

The index is built in this project from the market data above; it is not an
imported or published series. It exists to give the regimes an economic
order, and is also shown as a readable one-number summary of conditions.

### Construction

Eleven daily features in four blocks, each with a sign fixed in advance
(+ means a higher value is more risk-seeking):

| block | feature | sign | definition |
|---|---|---|---|
| Equity | `SPX_rmom21`, `SPX_rmom63` | + | S&P log return over 21 / 63 days, divided by 21-day realised vol × √h |
| | `SPX_dd252` | + | log(S&P ÷ its 252-day high); zero at a new high |
| | `RUT_rel_SPX_63` | + | 63-day change in log(Russell 2000 ÷ S&P): small caps leading |
| | `Nikkei_rmom63` | + | Nikkei 63-day volatility-adjusted return |
| Volatility | `SPX_logvol21` | − | log of annualised 21-day realised volatility |
| | `SPX_volratio` | − | log(21-day ÷ 63-day realised vol): volatility accelerating |
| | `VIX_log` | − | log VIX |
| Credit | `Baa_log` | − | log of Moody's Baa-minus-10y-Treasury spread |
| | `Baa_chg63` | − | 63-day change in that spread |
| Carry | `USDJPY_rmom63` | + | USD/JPY 63-day volatility-adjusted return: yen weakening, carry on |

1. Standardise each feature with the mean and standard deviation of the
   sample the model is fitted on.
2. Multiply by its sign and average within the block.
3. Average the four blocks with equal weight:

$$\text{RAI}_t = \tfrac14\left(\bar z^{\,\text{equity}}_t + \bar z^{\,\text{volatility}}_t + \bar z^{\,\text{credit}}_t + \bar z^{\,\text{carry}}_t\right)$$

Weighting by block rather than by feature stops equity dominating merely
because it contributes five columns. Zero means average conditions over the
estimation sample — an average that includes 1998, 2008 and 2020, so zero is
not a neutral-looking market but a mildly benign one.

The curve slope, the change in the 10-year yield and the stock/bond
correlation are deliberately **left out**: whether they signal risk appetite
depends on the macro backdrop (a steep curve can mean early expansion or
post-crisis easing), so they describe regimes but do not score them.

The signs and weights are a judgement, not an estimate. That is deliberate —
an index fitted to the data could be tuned to produce whatever regime ordering
looked best — but it means a different reasonable weighting would give a
somewhat different line. The index has not been validated against an
external measure such as the Chicago Fed NFCI or the St. Louis Fed Financial
Stress Index.

### What it is used for

- **Ordering the regimes.** Each state's average index value fixes its
  canonical position: 0 is the most risk-averse, K−1 the most risk-seeking.
- **Seeding the HMM.** Estimation starts from quantile buckets of the index
  rather than a random draw, which is what makes the labels seed-invariant.
- **Display.** On the header and the Characteristics tab. The historical line
  there is scaled with the full sample and so is descriptive; today's value
  uses only data up to today. The Characteristics chart is a 5-day average;
  the header shows the latest day.

### Why the index can be low while the regime is risk-on

This is expected, and it happens often. Over the full sample, the index was
below zero on roughly a quarter of the days the real-time regime was risk-on,
and above zero on roughly one risk-off day in eight. The panel below computes
these figures for whichever model is selected in the sidebar. There are five
reasons, and they usually act together.

**1. The two answer different questions.** The index is a fixed linear
summary designed to *order* states. The regime is chosen by the HMM's
likelihood over all fourteen labelling features with their full covariance,
and is designed to *classify* days. Nothing forces a day's index value to sit
nearest the average index value of the state it is assigned to; historically
the two agree on the exact state only about half the time, and on
risk-on versus risk-off about five times in six.

**2. They weight the same features very differently.** The index weights by
economic block, so the single carry feature, USD/JPY momentum, gets a quarter
of the index on its own. The HMM effectively weights each feature by how well
it separates the regimes relative to its noise within a regime. Drawdown,
credit spreads, VIX and realised volatility separate regimes strongly;
small-cap relative strength, the volatility ratio and yen carry barely do.
When those weakly separating features move sharply, the index moves and the
regime does not.

**3. The HMM sees features the index ignores.** The curve, the yield change
and the stock/bond correlation enter the likelihood but not the index.

**4. The regime has memory; the index does not.** The real-time regime
probability combines today's data with yesterday's regime and a daily
probability of staying put of roughly 99%. One soft reading cannot flip it;
it takes sustained evidence. The index is recomputed from scratch each day.

**5. Regimes are ranges, not thresholds.** Each regime covers a spread of
index values and the spreads overlap. The lower quarter of the Risk-on
expansion regime reaches down to around +0.2 and its tail goes below zero.
In the four-state model, near-average conditions typically fall in *Recovery
/ mixed*, which is itself on the risk-on side of the split — so an index near
zero is fully consistent with a risk-on verdict.

### How to read a divergence

A falling index inside a risk-on regime usually means the market is
**narrowing**: the variables that best identify regimes (volatility, credit,
drawdown) still look benign, while secondary ones (breadth, carry, foreign
equities) have turned. It is a reason to watch, not a regime change.

Whether such a divergence predicts a turn has **not** been tested as a signal
in its own right. The forecasting models already see every one of these
features, so any information in the divergence is already reflected in their
out-of-sample record, and that record shows turns are hard to time from
market data alone.

<!-- LIVE_RAI_PANEL -->

---

## What is actually being forecast at each horizon

Two different questions, reported separately:

- **Point in time** — P(risk-off **on day t+h**). This *decays towards the
  unconditional base rate* as h grows, because a single distant day approaches
  a draw from the stationary distribution.
- **Any in window** — P(**at least one** risk-off day in `(t, t+h]`). This
  *rises* with h by construction and is the quantity a risk manager wants.

A dashboard that shows one curve and lets the reader assume the other is how
horizon charts come to look nonsensical.

The target is the *filtered* state — what a real-time nowcast would have said
on day t+h using data up to t+h — not the smoothed state, which conditions on
the whole sample. On real data the two disagree on 3.3% of days.

---

## Evaluation

- **Expanding walk-forward**, 8-year initial window, 1-year test blocks, 27
  folds, out-of-sample from November 1998 to March 2026.
- **Purging**: training stops `h` sessions before each test block, so no
  training label is drawn from the period being scored.
- **Benchmarks that already exploit persistence.** Beating the unconditional
  base rate is nearly free when regimes are this persistent. The benchmark is
  a Markov forecast from the training transition matrix using today's state
  and nothing else. Every learned model is additionally *given* the current
  state posterior as an input, so it nests the benchmark and any gain is
  incremental.
- **Honest standard errors.** Daily sampling of an h-day horizon makes loss
  differentials heavily autocorrelated, so significance uses a
  Diebold–Mariano statistic with a Newey–West variance at bandwidth 2h. With
  overlapping windows the effective sample is roughly n/h: about **272
  independent observations at h=25**, not 6,800.

---

## What was found

**Regime states are hard to forecast beyond persistence.** On its own, the
best learned model beats the Markov benchmark by 7–13% of log loss at h≥10,
but only 2 of 10 horizon/target cells reach significance, and both are at h=5.

**Much of the short-horizon skill is arithmetic, not foresight.** The regime
at t+h is a function of trailing-window features at t+h; at h=5, 92% of a
63-day window is already observed at t, and the mean lag-5 autocorrelation of
the labelling features is 0.915. A model can forecast the future state
without knowing anything about the future market.

**Decomposing by what happened is more informative than the headline.** On the
86% of days when the state persists, the benchmark is close to unbeatable and
every learned model is worse than it. On the 14% of days when the state turns,
the benchmark is catastrophic (log loss above 2) and the learned models cut it
by 25–50%. The models carry information about turns and noise about
continuation.

**Forecast combination is what makes that usable.** An equal-weight average of
the benchmark with a learned model keeps most of the gain on turning days
while giving back most of the loss on quiet ones. After combination, **all 10
horizon/target cells show positive skill significant at the 5% level**, with
`blend50|compact|logit_reg` delivering a stable 14–15% across every horizon.
The weight is fixed at one half, not estimated: equal weighting cannot overfit
and is the standard robust default.

**Volatility is genuinely forecastable; direction is not.** On targets with no
window overlap at all, forward realised volatility above its training median
is predicted with **21–27% skill at every horizon from 5 to 25 days**
(t-statistics from −6.2 to −12.8, AUC 0.86–0.88), while the sign of the
forward S&P return shows **no skill at any horizon** (best result +0.7%,
p = 0.22, and the winner there is the unconditional base rate). This is the
cleanest evidence in the project that the feature set carries real
information — and that the information is about risk, not about returns. It
needs no combination step to be visible.

**Reformulating the regime target continuously did not work.** Regressing the
forward risk-appetite score with ridge and reading the probability off a
Gaussian predictive distribution — same features, same horizon, same event,
only a different loss — was *much worse* than classifying the indicator
directly, by 24% of log loss at h=25 and by far more at short horizons. A
constant residual standard deviation is a poor description of a score whose
dispersion is itself state-dependent, and the classifier that targets the
probability directly does not have to get the conditional variance right.

**Volatility and credit carry the signal.** Ablations show a forecaster given
only volatility features, or only credit features, does at least as well as
one given everything; rate features subtract. (Choosing the best ablation arm
on out-of-sample results would itself be selection bias, so the pre-specified
compact design remains the headline.)

**The original's class weighting was actively harmful.** `class_weight=
"balanced"` leaves ranking almost unchanged (AUC 0.938 against 0.954) while
turning +13% skill into −13%, because re-weighting deliberately biases
predicted probabilities away from the base rate. It is right for a decision
rule and wrong for a probability.

---

## Limitations

- **Effective sample.** About 272 independent observations at h=25 over 27
  years. Skill estimates of 10–15% have wide confidence intervals, and nothing
  here should be read as a precise number.
- **Multiplicity.** Roughly fourteen model specifications were evaluated
  across ten cells. A strict Bonferroni threshold (p < 0.004) would leave only
  the shortest horizons significant. The case rests on the *consistency* of
  sign and magnitude across all horizons, both target definitions, four
  choices of K, two labelling estimators and two samples — not on any single
  p-value.
- **The combination was motivated by an out-of-sample diagnostic.** The idea
  of blending came from observing the turn/continuation decomposition on
  out-of-sample data. The weight is not estimated and the blend is evaluated
  on the same folds, so its reported skill carries some researcher degree of
  freedom.
- **The target is model-defined.** "Risk-off" is an output of the HMM, not an
  observable. The volatility target is the model-free control, and it is the
  result to trust most.
- **Regimes are identified with a lag.** The labelling features include
  63- and 252-day windows, so the classification recognises a crisis some
  weeks after it starts. This is inherent to trailing-window features.
- **No transaction costs, no asset allocation, no backtest of a strategy.**
  Probability forecasts are evaluated as probability forecasts.
- **Structural change.** The sample spans 1990–2026, across which the
  volatility, credit and rate environments differ substantially. The expanding
  window means early folds are estimated on much less data than late ones.
- **A control not built.** The decisive test of the mechanical channel would
  be a simulation control that resamples future returns, recomputes features
  and reports the state distribution under "no news". It is not implemented
  here: the simpler flat-path version is ill-posed at h≥21 because the 21-day
  volatility window empties entirely.
