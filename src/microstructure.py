import pandas as pd
import numpy as np
import scipy.stats as stats

def create_time_bars(df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """Resamples high-frequency ticks into standard Time Bars."""
    print(f"Sampling Time Bars ({freq})...")
    bars = df.resample(freq).agg({
        'price': 'last',
        'qty': 'sum'
    }).dropna()
    
    # Compute log returns: r_t = ln(P_t / P_{t-1})
    bars['log_return'] = np.log(bars['price'] / bars['price'].shift(1))
    return bars.dropna()

def create_volume_bars(df: pd.DataFrame, volume_target: float = 100.0) -> pd.DataFrame:
    """Samples a new bar every time a target amount of asset volume is traded."""
    print(f"Sampling Volume Bars (Target: {volume_target} units)...")
    
    cumulative_volume = df['qty'].cumsum()
    vol_group = (cumulative_volume // volume_target).astype(int)
    
    bars = df.groupby(vol_group).agg({
        'price': 'last',
        'qty': 'sum'
    })
    
    bars['log_return'] = np.log(bars['price'] / bars['price'].shift(1))
    return bars.dropna()

def create_dollar_bars(df: pd.DataFrame, dollar_target: float = 5_000_000.0) -> pd.DataFrame:
    """Samples a new bar every time a target dollar value is exchanged."""
    print(f"Sampling Dollar Bars (Target: ${dollar_target:,.0f})...")
    
    dollar_value = df['price'] * df['qty']
    cumulative_dollars = dollar_value.cumsum()
    dollar_group = (cumulative_dollars // dollar_target).astype(int)
    
    bars = df.groupby(dollar_group).agg({
        'price': 'last',
        'qty': 'sum'
    })
    
    bars['log_return'] = np.log(bars['price'] / bars['price'].shift(1))
    return bars.dropna()
def analyze_bar_normality(time_bars: pd.DataFrame, vol_bars: pd.DataFrame, dollar_bars: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary metrics (Kurtosis, Skewness, Jarque-Bera Statistic) 
    for return distributions across sampling techniques.
    """
    datasets = [
        ("Time Bars (1m)", time_bars),
        ("Volume Bars", vol_bars),
        ("Dollar Bars", dollar_bars)
    ]
    
    metrics = []
    
    for name, bar_df in datasets:
        returns = bar_df['log_return'].values
        
        # Jarque-Bera test for normality
        jb_stat, p_val = stats.jarque_bera(returns)
        kurt = stats.kurtosis(returns) # Excess kurtosis (Normal distribution = 0)
        skew = stats.skew(returns)
        
        metrics.append({
            "Bar Schema": name,
            "Total Bars": len(returns),
            "Mean Return": f"{np.mean(returns):.6f}",
            "Std Dev": f"{np.std(returns):.6f}",
            "Skewness": round(skew, 3),
            "Excess Kurtosis": round(kurt, 2),
            "Jarque-Bera Stat": round(jb_stat, 2),
            "p-value": f"{p_val:.2e}"
        })
        
    summary_df = pd.DataFrame(metrics)
    return summary_df
def calculate_volatility_signature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Annualized Realized Volatility across varying time frequencies.
    Assumes crypto markets (365 trading days).
    """
    # Pandas offset aliases for sampling
    frequencies = {
        '1s': '1 Sec', '5s': '5 Sec', '15s': '15 Sec', '30s': '30 Sec',
        '1min': '1 Min', '2min': '2 Min', '5min': '5 Min', '10min': '10 Min', 
        '15min': '15 Min', '30min': '30 Min'
    }
    
    results = []
    
    print("Calculating Realized Volatility across frequencies...")
    for freq_code, label in frequencies.items():
        # Resample to the target frequency
        bars = df.resample(freq_code).agg({'price': 'last'}).dropna()
        
        # Calculate log returns
        bars['log_return'] = np.log(bars['price'] / bars['price'].shift(1))
        returns = bars['log_return'].dropna().values
        
        # Calculate Annualized Realized Volatility
        # Math: sqrt( sum(r^2) * 365 ) -> assuming 1 day of data is passed
        daily_variance = np.sum(returns**2)
        annualized_vol = np.sqrt(daily_variance * 365)
        
        results.append({
            'Freq_Code': freq_code,
            'Label': label,
            'Annualized_RV': annualized_vol
        })
        
    return pd.DataFrame(results)