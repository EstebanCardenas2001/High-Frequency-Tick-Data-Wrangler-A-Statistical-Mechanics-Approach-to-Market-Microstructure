import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def predictive_ofi_modeling(df: pd.DataFrame, freq: str = '10s'):
    """
    Calculates Order Flow Imbalance (OFI) to predict future short-term returns.
    """
    print(f"Resampling data to {freq} intervals for predictive modeling...")
    
    # Resample and calculate OFI (Net Signed Volume)
    bars = df.resample(freq).agg({
        'price': 'last',
        'signed_volume': 'sum'  # This acts as our OFI proxy from tick data
    }).dropna()
    
    # Rename for clarity
    bars.rename(columns={'signed_volume': 'OFI'}, inplace=True)
    
    # Calculate the target variable (Log Return)
    bars['log_return'] = np.log(bars['price'] / bars['price'].shift(1))
    
    # Shift the target backwards to create a predictive relationship 
    # (We want current OFI to predict NEXT period's return)
    bars['future_return'] = bars['log_return'].shift(-1)
    bars = bars.dropna()
    
    # Outlier Removal (Trimming the top/bottom 1% for cleaner linear regression)
    q_low = bars['OFI'].quantile(0.01)
    q_high = bars['OFI'].quantile(0.99)
    clean_bars = bars[(bars['OFI'] > q_low) & (bars['OFI'] < q_high)]
    
    # 5. Machine Learning Setup
    X = clean_bars[['OFI']].values
    y = clean_bars['future_return'].values * 10000  # Scale to Basis Points (bps) for readability
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train Ridge Regression (L2 Regularization to prevent overfitting)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Training Complete.")
    print(f"R-Squared (Out of Sample): {r2:.4f}")
    
    return clean_bars, model, X_test, y_test, y_pred
