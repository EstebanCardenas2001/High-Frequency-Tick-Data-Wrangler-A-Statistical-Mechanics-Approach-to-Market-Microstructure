import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
import time
import ssl

class HighFreqTickWrangler:
    def __init__(self, symbol: str, date: str):
        """
        Initializes the wrangler.
        :param symbol: Trading pair, e.g., 'BTCUSDT'
        :param date: Date in 'YYYY-MM-DD' format
        """
        self.symbol = symbol.upper()
        self.date = date
        self.filename = f"{self.symbol}-trades-{self.date}"

        self.df = None

    def download_binance_trades(self):
        """Downloads and extracts raw tick data from Binance's public data repository."""
        url = f"https://data.binance.vision/data/spot/daily/trades/{self.symbol}/{self.filename}.zip"
        zip_path = f"{self.filename}.zip"
        csv_path = f"{self.filename}.csv"

        if not os.path.exists(csv_path):
            print(f"Downloading tick data for {self.symbol} on {self.date}...")
            try:
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall()
                os.remove(zip_path) # Clean up the zip file
                print("Download and extraction complete.")
            except Exception as e:
                print(f"Error downloading data: {e}")
                return False
        else:
            print("CSV already exists locally.")
        return True

    def load_and_optimize(self):
        """Loads the CSV and heavily optimizes memory footprint."""
        print("Loading data into memory...")
        
        # Binance trade CSVs do not have headers by default.
        columns = ['trade_id', 'price', 'qty', 'quote_qty', 'timestamp', 'is_buyer_maker', 'is_best_match']
        
        # We specify dtypes on load to prevent Pandas from defaulting to memory-heavy float64/int64
        dtypes = {
            'price': 'float32',
            'qty': 'float32',
            'quote_qty': 'float32',
            'is_buyer_maker': 'bool',
        }
        
        self.df = pd.read_csv(f"{self.filename}.csv", names=columns, dtype=dtypes, usecols=['price', 'qty', 'timestamp', 'is_buyer_maker'])
        
        # Convert timestamp to a datetime object (Binance uses millisecond Unix time)
        # Handle Binance schema drift: 2025+ uses microseconds, pre-2025 uses milliseconds
        year_requested = int(self.date[:4])
        time_unit = 'us' if year_requested >= 2025 else 'ms'
        
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit=time_unit)
        self.df.set_index('timestamp', inplace=True)
        self.df.sort_index(inplace=True)
        
        print(f"Loaded {len(self.df):,} rows. Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    def engineer_features(self):
        """Uses Numpy vectorization to calculate high-frequency features instantly."""
        print("Engineering features using vectorized operations...")
        start_time = time.time()

        # Trade Direction Mapping (Buyer Maker = Sell Order, otherwise Buy Order)
        # Using numpy.where is infinitely faster than pandas apply()
        self.df['trade_direction'] = np.where(self.df['is_buyer_maker'], -1, 1)

        # Signed Volume (Volume * Direction)
        self.df['signed_volume'] = self.df['qty'] * self.df['trade_direction']

        # Rolling Cumulative Volume Delta (CVD) - A key metric for algorithmic traders
        self.df['cvd'] = self.df['signed_volume'].cumsum()

        # Volume Weighted Average Price (VWAP) over a rolling window (e.g., last 1000 ticks)
        # Using NumPy arrays to bypass Pandas overhead for massive speedups
        prices = self.df['price'].values
        volumes = self.df['qty'].values
        
        rolling_window = 1000
        
        # Calculate rolling VWAP
        rolling_vol = pd.Series(volumes).rolling(window=rolling_window, min_periods=1).sum().values
        rolling_vol_price = pd.Series(prices * volumes).rolling(window=rolling_window, min_periods=1).sum().values
        
        self.df['rolling_vwap'] = rolling_vol_price / rolling_vol

        print(f"Feature engineering completed in {time.time() - start_time:.4f} seconds.")
        
    def get_summary(self):
        return self.df.head(10)