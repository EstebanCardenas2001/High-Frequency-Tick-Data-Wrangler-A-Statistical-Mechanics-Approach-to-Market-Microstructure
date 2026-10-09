import hashlib
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BINANCE_BASE_URL = "https://data.binance.vision/data/spot/daily/trades"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Binance spot trade files are header-less with this column order.
TRADE_COLUMNS = ['trade_id', 'price', 'qty', 'quote_qty', 'timestamp', 'is_buyer_maker', 'is_best_match']


def infer_timestamp_unit(raw_ts) -> str:
    """
    Infers the Unix epoch unit from the magnitude of a raw integer timestamp.
    Binance switched spot data from milliseconds to microseconds on 2025-01-01;
    reading the magnitude is robust to that schema drift regardless of the file date.
    """
    ts = int(raw_ts)
    if ts >= 10**17:
        return 'ns'
    if ts >= 10**14:
        return 'us'
    if ts >= 10**11:
        return 'ms'
    return 's'


class HighFreqTickWrangler:
    def __init__(self, symbol: str, date: str, data_dir=DEFAULT_DATA_DIR):
        """
        Initializes the wrangler.
        :param symbol: Trading pair, e.g., 'BTCUSDT'
        :param date: Date in 'YYYY-MM-DD' format
        :param data_dir: Directory where raw archives are cached (defaults to the repo's data/ folder)
        """
        self.symbol = symbol.upper()
        self.date = date
        self.filename = f"{self.symbol}-trades-{self.date}"
        self.data_dir = Path(data_dir)
        self.zip_path = self.data_dir / f"{self.filename}.zip"

        self.df = None

    def download_binance_trades(self, verify_checksum: bool = True) -> bool:
        """Downloads the raw tick archive from Binance's public data repository (cached in data_dir)."""
        if self.zip_path.exists():
            print(f"Archive already cached at {self.zip_path}.")
            return True

        self.data_dir.mkdir(parents=True, exist_ok=True)
        url = f"{BINANCE_BASE_URL}/{self.symbol}/{self.filename}.zip"
        tmp_path = self.zip_path.with_suffix('.zip.part')

        print(f"Downloading tick data for {self.symbol} on {self.date}...")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            if verify_checksum:
                with urllib.request.urlopen(f"{url}.CHECKSUM") as resp:
                    expected = resp.read().decode().split()[0]
                with open(tmp_path, 'rb') as f:
                    actual = hashlib.sha256(f.read()).hexdigest()
                if actual != expected:
                    raise ValueError(f"SHA256 mismatch (expected {expected}, got {actual})")
            os.replace(tmp_path, self.zip_path)
        except Exception as e:
            print(f"Error downloading data: {e}")
            tmp_path.unlink(missing_ok=True)
            return False

        print("Download complete.")
        return True

    def load_and_optimize(self):
        """Loads the archive and shrinks its memory footprint without sacrificing numerical precision."""
        print("Loading data into memory...")

        # Some Binance dumps ship a header row; sniff the first line to decide.
        first_row = pd.read_csv(self.zip_path, header=None, nrows=1)
        has_header = not str(first_row.iloc[0, 0]).isdigit()

        # Price stays float64: float32 only has ~7 significant digits, which corrupts
        # quotes such as 87648.22 (-> 87648.21875). Quantity is safe in float32 for storage;
        # every accumulation below is done in float64 to avoid summation drift.
        dtypes = {
            'price': 'float64',
            'qty': 'float32',
            'timestamp': 'int64',
            'is_buyer_maker': 'bool',
        }

        self.df = pd.read_csv(
            self.zip_path,
            header=0 if has_header else None,
            names=TRADE_COLUMNS,
            dtype=dtypes,
            usecols=['price', 'qty', 'timestamp', 'is_buyer_maker'],
        )

        time_unit = infer_timestamp_unit(self.df['timestamp'].iloc[0])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit=time_unit)
        self.df.set_index('timestamp', inplace=True)
        # Stable sort keeps the exchange's matching order for trades sharing a timestamp
        self.df.sort_index(inplace=True, kind='stable')

        print(f"Loaded {len(self.df):,} rows (timestamps in '{time_unit}'). "
              f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    def engineer_features(self, vwap_window: int = 1000):
        """Uses NumPy vectorization to calculate high-frequency features."""
        if self.df is None:
            raise RuntimeError("No data loaded. Call load_and_optimize() first.")

        print("Engineering features using vectorized operations...")
        start_time = time.time()

        # Trade Direction (aggressor side): buyer is maker -> seller aggressed -> -1, otherwise +1
        self.df['trade_direction'] = np.where(self.df['is_buyer_maker'], -1, 1).astype('int8')

        # Signed Volume (Volume * Direction), float64 so downstream sums do not drift
        qty = self.df['qty'].to_numpy(dtype='float64')
        self.df['signed_volume'] = qty * self.df['trade_direction'].to_numpy()

        # Cumulative Volume Delta (CVD) - running net aggressive volume
        self.df['cvd'] = np.cumsum(self.df['signed_volume'].to_numpy())

        # Volume Weighted Average Price over the last `vwap_window` ticks
        prices = self.df['price'].to_numpy(dtype='float64')
        rolling_vol = pd.Series(qty).rolling(window=vwap_window, min_periods=1).sum().to_numpy()
        rolling_vol_price = pd.Series(prices * qty).rolling(window=vwap_window, min_periods=1).sum().to_numpy()
        self.df['rolling_vwap'] = rolling_vol_price / rolling_vol

        print(f"Feature engineering completed in {time.time() - start_time:.4f} seconds.")

    def get_summary(self, n: int = 10) -> pd.DataFrame:
        return self.df.head(n)
