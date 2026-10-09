import numpy as np
import pandas as pd
import scipy.stats as stats

SECONDS_PER_DAY = 86_400


def _add_log_returns(bars: pd.DataFrame) -> pd.DataFrame:
    # Compute log returns: r_t = ln(P_t / P_{t-1})
    bars['log_return'] = np.log(bars['price'] / bars['price'].shift(1))
    return bars.dropna(subset=['log_return'])


def _event_bars(df: pd.DataFrame, measure: np.ndarray, threshold: float) -> pd.DataFrame:
    """
    Groups ticks into bars that close each time the cumulative `measure` crosses a multiple
    of `threshold`. Bars are indexed by the timestamp of their last tick.
    """
    # float64 cumsum: a float32 running total over ~10^6 ticks drifts by >1%
    group = np.floor_divide(np.cumsum(measure, dtype='float64'), threshold).astype(np.int64)

    grouped = df.groupby(group, sort=True)
    bars = grouped.agg(price=('price', 'last'), qty=('qty', 'sum'))
    bars['n_ticks'] = grouped.size()
    bars.index = df.index.to_series().groupby(group, sort=True).last().values
    bars.index.name = 'timestamp'
    return _add_log_returns(bars)


def create_time_bars(df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """Resamples high-frequency ticks into standard Time Bars."""
    print(f"Sampling Time Bars ({freq})...")
    bars = df.resample(freq).agg({'price': 'last', 'qty': 'sum'}).dropna(subset=['price'])
    return _add_log_returns(bars)


def create_volume_bars(df: pd.DataFrame, volume_target: float = 100.0) -> pd.DataFrame:
    """Samples a new bar every time a target amount of asset volume is traded."""
    print(f"Sampling Volume Bars (Target: {volume_target} units)...")
    return _event_bars(df, df['qty'].to_numpy(dtype='float64'), volume_target)


def create_dollar_bars(df: pd.DataFrame, dollar_target: float = 5_000_000.0) -> pd.DataFrame:
    """Samples a new bar every time a target dollar (quote) value is exchanged."""
    print(f"Sampling Dollar Bars (Target: ${dollar_target:,.0f})...")
    dollar_value = df['price'].to_numpy(dtype='float64') * df['qty'].to_numpy(dtype='float64')
    return _event_bars(df, dollar_value, dollar_target)


def matching_time_freq(df: pd.DataFrame, n_bars: int) -> str:
    """
    Returns the time-bar frequency (whole seconds) that yields roughly `n_bars` bars over the
    sample. Used to compare time bars and event bars at the same sampling resolution.
    """
    span = (df.index[-1] - df.index[0]).total_seconds()
    return f"{max(1, round(span / n_bars))}s"


def analyze_bar_normality(bars: dict) -> pd.DataFrame:
    """
    Computes summary metrics (Skewness, Excess Kurtosis, Jarque-Bera statistic)
    for the return distribution of each bar schema.
    :param bars: mapping {label: bar DataFrame with a 'log_return' column}
    """
    metrics = []
    for name, bar_df in bars.items():
        returns = bar_df['log_return'].dropna().to_numpy()

        # Jarque-Bera test for normality (asymptotic; low power on a few hundred bars)
        jb_stat, p_val = stats.jarque_bera(returns)

        metrics.append({
            "Bar Schema": name,
            "Total Bars": len(returns),
            "Mean Return": np.mean(returns),
            "Std Dev": np.std(returns, ddof=1),
            "Skewness": stats.skew(returns),
            "Excess Kurtosis": stats.kurtosis(returns),  # Normal distribution = 0
            "Jarque-Bera Stat": jb_stat,
            "p-value": p_val,
        })

    return pd.DataFrame(metrics).set_index("Bar Schema")


DEFAULT_SIGNATURE_FREQS = {
    '1s': '1 Sec', '5s': '5 Sec', '15s': '15 Sec', '30s': '30 Sec',
    '1min': '1 Min', '2min': '2 Min', '5min': '5 Min', '10min': '10 Min',
    '15min': '15 Min', '30min': '30 Min',
}


def calculate_volatility_signature(df: pd.DataFrame, frequencies: dict = None) -> pd.DataFrame:
    """
    Calculates Annualized Realized Volatility across varying sampling frequencies.
    Assumes crypto markets (365 trading days). Works for samples spanning any number of days.
    """
    frequencies = frequencies or DEFAULT_SIGNATURE_FREQS
    n_days = max((df.index[-1] - df.index[0]).total_seconds() / SECONDS_PER_DAY, 1e-9)

    results = []
    print("Calculating Realized Volatility across frequencies...")
    for freq_code, label in frequencies.items():
        bars = df.resample(freq_code).agg({'price': 'last'}).dropna()
        returns = np.log(bars['price'] / bars['price'].shift(1)).dropna().to_numpy()

        # RV_annual = sqrt(365 * mean daily sum(r^2))
        daily_rv = np.sum(returns**2) / n_days
        results.append({
            'Freq_Code': freq_code,
            'Label': label,
            'N_Returns': len(returns),
            'Annualized_RV': np.sqrt(daily_rv * 365),
        })

    return pd.DataFrame(results)


def find_noise_cutoff(rv_df: pd.DataFrame, tolerance: float = 0.10, n_plateau: int = 3) -> int:
    """
    Locates the sampling frequency where microstructure noise has decayed.
    The latent-volatility plateau is estimated as the median RV of the `n_plateau` slowest
    frequencies; the cutoff is the first row from which every RV stays within
    `tolerance` of that plateau. Returns the positional index into `rv_df`.
    """
    rv = rv_df['Annualized_RV'].to_numpy()
    plateau = np.median(rv[-n_plateau:])
    inside = np.abs(rv / plateau - 1.0) <= tolerance

    # First index i such that inside[i:] are all True
    cutoff = len(rv) - 1
    for i in range(len(rv) - 1, -1, -1):
        if not inside[i]:
            break
        cutoff = i
    return cutoff
