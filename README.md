# High-Frequency Tick Data Wrangler: A Statistical Mechanics Approach to Market Microstructure

A Python pipeline that ingests millions of raw cryptocurrency trades (ticks) from Binance's public data repository and applies market-microstructure theory to them: event-based resampling, normality testing, realized-volatility signature analysis, and an Order Flow Imbalance (OFI) model of short-term returns.

The project treats financial markets as complex, non-equilibrium statistical systems and uses that analogy to motivate each step of the analysis.

## The Physics of Financial Markets

Classical financial models (like Black-Scholes) assume price changes are Independent and Identically Distributed (IID), essentially modeling trades as random, non-interacting particles in an ideal gas. In reality, markets exhibit herd behavior, volatility clustering, and heavy-tailed risk events.

| Quantitative Finance Concept | Statistical Physics Analogy | Implementation in Pipeline |
| :--- | :--- | :--- |
| **High-Frequency Ticks** | Microscopic particle collisions | Ingesting raw, individual exchange matching events. |
| **Dollar / Volume Bars** | Arc-length (intrinsic-time) parameterization | Changing the independent variable from clock time *t* to cumulative economic throughput to stabilize variance. |
| **Fat Tails & Kurtosis** | Non-equilibrium states | Testing empirically whether clock-time sampling yields heavy-tailed returns while event-based sampling recovers a near-Gaussian distribution. |
| **Microstructure Noise** | Thermal jitter at microscopic scales | Using a Volatility Signature Plot to find the sampling frequency at which microstructure effects stop biasing realized volatility. |
| **Order Flow Imbalance (OFI)** | Net applied force | Measuring net aggressive buying/selling pressure and testing whether it predicts the next price move. |

---

## Project Structure

```
.
├── data/                              # Downloaded Binance archives are cached here (git-ignored)
├── notebooks/
│   └── microstructure_analysis.ipynb  # End-to-end analysis with plots
├── src/
│   ├── tick_wrangler.py               # Download, load, memory optimization, tick features
│   ├── microstructure.py              # Time/volume/dollar bars, normality tests, volatility signature
│   └── modeling.py                    # OFI feature + Ridge regression
├── tests/
│   └── test_pipeline.py               # Offline tests on synthetic tick data
└── requirements.txt
```

## Installation

Requires Python 3.10+.

```bash
git clone <repo-url>
cd High-Frequency-Tick-Data-Wrangler-A-Statistical-Mechanics-Approach-to-Market-Microstructure
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Notebook

```bash
cd notebooks
jupyter notebook microstructure_analysis.ipynb
```

The first run downloads about 11 MB (1.45 M trades) for `BTCUSDT` on `2026-01-01` into `data/`. Change the `symbol` and `date` in the first code cell to analyze any other Binance spot pair or day.

### As a library

Run from the repository root:

```python
from src.tick_wrangler import HighFreqTickWrangler
from src.microstructure import (
    create_time_bars, create_dollar_bars, matching_time_freq,
    analyze_bar_normality, calculate_volatility_signature, find_noise_cutoff,
)
from src.modeling import predictive_ofi_modeling

w = HighFreqTickWrangler(symbol="BTCUSDT", date="2026-01-01")
w.download_binance_trades()       # cached in data/, SHA256-verified
w.load_and_optimize()
w.engineer_features()

dollar_bars = create_dollar_bars(w.df, dollar_target=2_500_000)
time_bars = create_time_bars(w.df, freq=matching_time_freq(w.df, len(dollar_bars)))
print(analyze_bar_normality({"Time Bars": time_bars, "Dollar Bars": dollar_bars}))

rv = calculate_volatility_signature(w.df)
print("Noise cutoff:", rv["Label"].iloc[find_noise_cutoff(rv)])

predictive_ofi_modeling(w.df, freq="10s")
```

### Tests

```bash
pytest
```

The tests use synthetic tick data and need no network access.

---

## Core Modules & Architecture

### 1. Data Ingestion & Memory Optimization (`tick_wrangler.py`)
* Downloads daily trade archives from `data.binance.vision`, verifies them against Binance's published SHA256 checksum, and caches them in `data/`. Pandas reads the zip directly, so no extraction step is needed.
* Handles upstream **schema drift**: Binance switched spot timestamps from milliseconds to microseconds in 2025. The unit is inferred from the timestamp's magnitude rather than the file date, and an optional header row is detected automatically.
* **Memory optimization**: loads only the 4 needed columns and stores quantity as `float32`. Price is kept in `float64` because float32 has only ~7 significant digits and would corrupt quotes (87648.22 → 87648.21875). On the 1.45 M-trade sample day, memory drops from 58.1 MB (naive full load) to 29.0 MB.
* **Vectorized features**: trade direction (aggressor side), signed volume, Cumulative Volume Delta (CVD), and a rolling 1,000-tick VWAP. All running sums are computed in `float64`. A float32 cumulative sum drifts by about 1% over a day of ticks and shifts bar boundaries.

### 2. Event-Based Resampling & Normality (`microstructure.py`)
Log-returns `r_t = ln(P_t / P_{t-1})` are computed in three coordinate systems: **Time Bars**, **Volume Bars**, and **Dollar Bars**. Event bars are indexed by the timestamp of their last tick.

Aggregating more trades per bar makes returns more Gaussian on its own (Central Limit Theorem). To isolate the effect of the *sampling clock*, `matching_time_freq` builds a resolution-matched control: time bars with the same bar count as the dollar bars.

Results for BTCUSDT on 2026-01-01:

| Bar Schema | Bars | Skewness | Excess Kurtosis | Jarque-Bera | p-value |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Time Bars (1 min) | 1,439 | 0.70 | 11.03 | 7405.0 | < 1e-300 |
| Time Bars (391 s, resolution-matched) | 220 | 0.55 | 1.57 | 33.7 | 4.9e-08 |
| Volume Bars (50 BTC) | 125 | 0.17 | 0.17 | 0.8 | 0.68 |
| Dollar Bars ($2.5 M) | 221 | 0.21 | 0.07 | 1.7 | 0.43 |

Even at matched resolution, time-bar returns reject normality, while dollar- and volume-bar returns are consistent with a Gaussian. The notebook also compares autocorrelations of squared returns: the time bars show significant volatility clustering at short lags, and the dollar bars largely do not.

*Caveat:* this is a single day, and with roughly 100–200 bars the Jarque-Bera test has limited power. A failure to reject normality is evidence, not proof.

### 3. Microstructure Noise & Realized Volatility (`microstructure.py`)
Annualized realized volatility is computed across sampling frequencies from 1 s to 30 min:

`RV_annual = sqrt(365 * (1/D) * sum(r_i^2))`, where *D* is the number of days in the sample.

* The **Volatility Signature Plot** shows how microstructure effects bias RV at high frequencies. Bid-ask bounce *inflates* RV. Stale prices and split orders (positively autocorrelated tick returns) *deflate* it.
* `find_noise_cutoff` estimates the latent-volatility plateau as the median RV of the slowest frequencies. It returns the first frequency from which RV stays within ±10% of that plateau.
* For BTCUSDT the curve **rises** from 12.6% (1 s) toward roughly 15% (5 min and slower). With a $0.01 tick on a ~$87k asset, bid-ask bounce is negligible, and the dominant high-frequency distortion is positive autocorrelation. The detected cutoff is **5 min**. On a single day the low-frequency estimates rest on few returns (47 at 30 min), so the plateau itself is noisy.

### 4. Predictive Modeling: Order Flow Imbalance (`modeling.py`)
* OFI over each interval is the sum of signed trade volume: `OFI_t = sum(Volume_i * Direction_i)`. This is a trade-based proxy; true order-book OFI requires quote data.
* The target is the **next** interval's log return in basis points (`shift(-1)`), so the feature at *t* only predicts *t+1*.
* Leakage controls: the train/test split is chronological (first 80% / last 20%), and the outlier-trimming quantiles (1% / 99% of OFI) are learned on the training set only.
* Model: Ridge regression (L2, `alpha=1.0`).

Out-of-sample results on 10 s bars for 2026-01-01: **R² = 0.004**, with a directional hit rate of **56%** on non-zero returns. Concurrent OFI is known to explain a large share of same-interval price changes. Its predictive power for the *next* interval is small, as expected in a liquid market. Treat this as a measurement, not a trading strategy: it ignores transaction costs and covers a single day.

---

## Data Source

Binance public market data: <https://data.binance.vision> (`spot/daily/trades`). Each row is one trade: `trade_id, price, qty, quote_qty, timestamp, is_buyer_maker, is_best_match`. When `is_buyer_maker` is `True`, the seller was the aggressor, so the trade is labeled a sell (−1).

## Limitations & Possible Extensions

* All results above come from a single trading day. Running over many days or symbols would make them robust.
* Bar thresholds (50 BTC, $2.5 M) are fixed; they could be set adaptively, for example from a target number of bars per day.
* OFI is trade-based. Level-1 order-book data would allow the Cont–Kukanov–Stoikov OFI definition.
