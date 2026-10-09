import zipfile

import numpy as np
import pandas as pd
import pytest

from src.microstructure import (
    analyze_bar_normality,
    calculate_volatility_signature,
    create_dollar_bars,
    create_time_bars,
    create_volume_bars,
    find_noise_cutoff,
    matching_time_freq,
)
from src.modeling import predictive_ofi_modeling
from src.tick_wrangler import HighFreqTickWrangler, infer_timestamp_unit


def make_ticks(n=50_000, seed=0):
    """Synthetic one-day tick tape with a random-walk price."""
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(0, 86_400 * 10**6, n)) + 1_767_225_600 * 10**6
    price = 87_000 * np.exp(np.cumsum(rng.normal(0, 1e-5, n)))
    qty = rng.exponential(0.05, n)
    return pd.DataFrame({
        'trade_id': np.arange(n),
        'price': np.round(price, 2),
        'qty': np.round(qty, 5),
        'quote_qty': np.round(price * qty, 8),
        'timestamp': ts,
        'is_buyer_maker': rng.random(n) < 0.5,
        'is_best_match': True,
    })


def write_zip(tmp_path, ticks, symbol='BTCUSDT', date='2026-01-01', header=False):
    name = f"{symbol}-trades-{date}"
    with zipfile.ZipFile(tmp_path / f"{name}.zip", 'w') as z:
        z.writestr(f"{name}.csv", ticks.to_csv(index=False, header=header))


@pytest.fixture
def wrangler(tmp_path):
    write_zip(tmp_path, make_ticks())
    w = HighFreqTickWrangler('btcusdt', '2026-01-01', data_dir=tmp_path)
    assert w.download_binance_trades()  # cached archive -> no network
    w.load_and_optimize()
    w.engineer_features()
    return w


@pytest.mark.parametrize("raw, unit", [
    (1_767_225_600, 's'),
    (1_704_067_200_000, 'ms'),
    (1_767_225_600_039_409, 'us'),
    (1_767_225_600_039_409_000, 'ns'),
])
def test_infer_timestamp_unit(raw, unit):
    assert infer_timestamp_unit(raw) == unit


@pytest.mark.parametrize("header", [False, True])
def test_load_handles_optional_header(tmp_path, header):
    ticks = make_ticks(1_000)
    write_zip(tmp_path, ticks, header=header)
    w = HighFreqTickWrangler('BTCUSDT', '2026-01-01', data_dir=tmp_path)
    w.load_and_optimize()
    assert len(w.df) == len(ticks)
    assert w.df.index[0].year == 2026
    # Prices keep cent precision (float32 would not)
    np.testing.assert_array_equal(w.df['price'].to_numpy(), ticks['price'].to_numpy())


def test_features(wrangler):
    df = wrangler.df
    assert set(df['trade_direction'].unique()) <= {-1, 1}
    assert (df.loc[df['is_buyer_maker'], 'trade_direction'] == -1).all()
    np.testing.assert_allclose(df['cvd'].iloc[-1], df['signed_volume'].sum())
    assert df['rolling_vwap'].between(df['price'].min(), df['price'].max()).all()


def test_event_bars_conserve_volume(wrangler):
    df = wrangler.df
    total = df['qty'].astype('float64').sum()
    vb = create_volume_bars(df, volume_target=total / 50)
    db = create_dollar_bars(df, dollar_target=(df['price'] * df['qty']).sum() / 50)
    # ~50 bars each (the first bar is consumed by the return calculation)
    assert 45 <= len(vb) <= 51 and 45 <= len(db) <= 51
    assert isinstance(vb.index, pd.DatetimeIndex) and vb.index.is_monotonic_increasing
    assert vb['n_ticks'].sum() <= len(df)
    assert not vb['log_return'].isna().any()


def test_normality_table_and_matching_freq(wrangler):
    df = wrangler.df
    db = create_dollar_bars(df, dollar_target=(df['price'] * df['qty']).sum() / 200)
    freq = matching_time_freq(df, len(db))
    tb = create_time_bars(df, freq)
    assert abs(len(tb) - len(db)) / len(db) < 0.1

    table = analyze_bar_normality({'time': tb, 'dollar': db})
    assert list(table.index) == ['time', 'dollar']
    assert table['p-value'].between(0, 1).all()
    assert pd.api.types.is_float_dtype(table['Excess Kurtosis'])


def test_volatility_signature_and_cutoff(wrangler):
    rv = calculate_volatility_signature(wrangler.df)
    assert (rv['Annualized_RV'] > 0).all()
    assert 0 <= find_noise_cutoff(rv) < len(rv)

    flat = pd.DataFrame({'Label': list('abcde'), 'Annualized_RV': [0.30, 0.22, 0.15, 0.15, 0.15]})
    assert find_noise_cutoff(flat, tolerance=0.10) == 2


def test_ofi_model_is_chronological(wrangler):
    clean_bars, model, X_test, y_test, y_pred = predictive_ofi_modeling(wrangler.df, freq='60s')
    assert len(X_test) == len(y_test) == len(y_pred)
    # Test rows are the tail of the sample (no shuffling)
    test_index = clean_bars.index[-len(y_test):]
    assert test_index.min() > clean_bars.index[:-len(y_test)].max()
