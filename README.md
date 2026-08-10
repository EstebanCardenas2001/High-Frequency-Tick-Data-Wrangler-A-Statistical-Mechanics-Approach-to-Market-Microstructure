# High-Frequency Tick Data Wrangler: A Statistical Mechanics Approach to Market Microstructure
Processes millions of raw, high-frequency cryptocurrency trades (ticks) and applies advanced market microstructure theory to extract predictive, low-noise signals. 

By treating financial markets as complex, non-equilibrium thermodynamic systems, this project bridges the gap between theoretical physics and quantitative finance.

## The Physics of Financial Markets

Classical financial models (like Black-Scholes) assume price changes are Independent and Identically Distributed (IID)—essentially modeling trades as random, non-interacting particles in an ideal gas. In reality, markets exhibit herd behavior, momentum, and heavy-tailed risk events. 

This project maps financial data engineering directly to statistical mechanics to mathematically correct these assumptions.

| Quantitative Finance Concept | Statistical Physics Equivalent | Implementation in Pipeline |
| :--- | :--- | :--- |
| **High-Frequency Ticks** | Microscopic Particle Collisions | Ingesting raw, unstructured order-book matching events. |
| **Dollar/Volume Bars** | Arc Length Parameterization | Changing the independent variable from chronological time (t) to economic energy/throughput (E) to stabilize variance. |
| **Fat Tails & Kurtosis** | Non-Equilibrium States | Proving empirically that temporal sampling yields unstable jump dynamics, while event-based sampling recovers a quasi-Gaussian state. |
| **Microstructure Noise** | Brownian Motion (Thermal Jitter) | Using a Volatility Signature Plot to act as a low-pass filter, canceling out bid-ask bounce to isolate macroscopic latent volatility. |
| **Order Flow Imbalance (OFI)** | Net Applied Force (F=ma) | Calculating the net aggressive buying/selling pressure to predict the kinematic trajectory of the asset's price. |

---

##  Core Modules & Architecture

### 1. Memory Optimization & Data Ingestion
Processes millions of rows from Binance's public data endpoints. 
* Handles upstream **schema drift** (dynamically switching between millisecond and microsecond Unix timestamps).
* Employs **memory downcasting** (forcing 64-bit floats to 32-bit floats and booleans), reducing Pandas RAM consumption by over 60%.
* Utilizes **vectorized NumPy operations** to calculate Cumulative Volume Delta (CVD) and rolling Volume Weighted Average Price (VWAP) at C-level speeds.

### 2. Event-Based Resampling (Normality Recovery)
Calculates log-returns across different coordinate systems:
`r_t = ln(P_t / P_{t-1})`
* Generates Time Bars, Volume Bars, and Dollar Bars.
* Utilizes the **Jarque-Bera Test** and **Probability Density Functions (PDF)** to mathematically prove that Dollar Bars absorb periods of high volatility, stripping excess kurtosis and recovering the Gaussian normality required for stable machine learning models.

### 3. Microstructure Noise & Realized Volatility
Calculates annualized realized volatility across varying frequencies:
`RV_annual = sqrt(365 * sum(r_i^2))`
* Constructs a **Volatility Signature Plot** to visualize the decay of microstructure noise (bid-ask friction).
* Algorithmically identifies the optimal sampling frequency where "thermal jitter" disappears, isolating the true latent signal.

### 4. Predictive Modeling (Order Flow Imbalance)
Transitions from descriptive statistics to predictive machine learning. 
* Calculates Order Flow Imbalance (OFI) as the summation of signed volume:
`OFI_t = sum(Volume_i * Direction_i)`
* Employs a **Ridge Regression model** (L2 Regularization) to map the linear relationship between net market force and short-term price impact.
* Implements rigorous target shifting (`shift(-1)`) to prevent data leakage and guarantee purely out-of-sample predictions.


