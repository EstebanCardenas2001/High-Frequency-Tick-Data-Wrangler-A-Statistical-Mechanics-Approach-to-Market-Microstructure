import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def predictive_ofi_modeling(df: pd.DataFrame, freq: str = '10s', test_size: float = 0.2,
                            trim_quantile: float = 0.01, alpha: float = 1.0):
    """
    Calculates Order Flow Imbalance (OFI) over `freq` intervals and fits a Ridge regression
    of the NEXT interval's log return (in bps) on the current OFI.
    Returns (clean_bars, model, X_test, y_test, y_pred).
    """
    print(f"Resampling data to {freq} intervals for predictive modeling...")

    # Resample and calculate OFI (Net Signed Volume)
    bars = df.resample(freq).agg({
        'price': 'last',
        'signed_volume': 'sum'  # This acts as our OFI proxy from tick data
    }).dropna()
    bars.rename(columns={'signed_volume': 'OFI'}, inplace=True)

    # Calculate the target variable (Log Return)
    bars['log_return'] = np.log(bars['price'] / bars['price'].shift(1))

    # Shift the target backwards to create a predictive relationship
    # (current OFI predicts NEXT period's return)
    bars['future_return'] = bars['log_return'].shift(-1)
    bars = bars.dropna()

    # Chronological split BEFORE any data-dependent preprocessing to avoid look-ahead
    split = int(len(bars) * (1 - test_size))
    train, test = bars.iloc[:split], bars.iloc[split:]

    # Outlier trimming with thresholds learned on the training set only
    q_low, q_high = train['OFI'].quantile([trim_quantile, 1 - trim_quantile])

    def trim(frame):
        return frame[(frame['OFI'] > q_low) & (frame['OFI'] < q_high)]

    train, test = trim(train), trim(test)
    clean_bars = pd.concat([train, test])

    # Scale target to Basis Points (bps) for readability
    X_train, y_train = train[['OFI']].to_numpy(), train['future_return'].to_numpy() * 10_000
    X_test, y_test = test[['OFI']].to_numpy(), test['future_return'].to_numpy() * 10_000

    # Ridge Regression (L2 Regularization)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    nonzero = y_test != 0
    hit_rate = np.mean(np.sign(y_pred[nonzero]) == np.sign(y_test[nonzero]))

    print("Model Training Complete.")
    print(f"Train / Test samples: {len(train)} / {len(test)}")
    print(f"R-Squared (Out of Sample): {r2:.4f}")
    print(f"Directional Hit Rate (Out of Sample, non-zero returns): {hit_rate:.2%}")

    return clean_bars, model, X_test, y_test, y_pred
