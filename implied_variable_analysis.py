import pickle
import numpy as np
import pandas as pd

# === Load saved OLS model ===
with open('return_multivar_ols_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(model.summary())

# === Hardcoded inputs (real space) ===
dp = 1    # dividend-price ratio
inf = 0.0   # expected annual inflation

# === Convert to log space (for model input) ===
dp_log = np.log(dp)
fwd_log_avg_inf = np.log(1 + inf)

# === Create schedule of assumed 10-year forward average returns (real space) ===
fwd_avg_returns = np.arange(0.00, 0.155, 0.01)
fwd_log_avg_trs = np.log1p(fwd_avg_returns)

# === Solve for required log dividend growth that makes model prediction match assumed returns ===
params = model.params

required_log_div_growth = (
    fwd_log_avg_trs 
    - params['const'] 
    - params['dp_log_lag1'] * dp_log 
    - params['fwd_log_avg_inf'] * fwd_log_avg_inf
) / params['fwd_log_real_div_growth']

# === Convert back to real-space dividend growth ===
required_div_growth_real = np.expm1(required_log_div_growth)

# === Build results table ===
result_df = pd.DataFrame({
    'Assumed Fwd Avg Real Return (%)': (fwd_avg_returns * 100).round(2),
    'Required Real Div Growth (%)': (required_div_growth_real * 100).round(2)
})

# === Display ===
print("Current Shareholder Yield: {:.2f}%".format(dp * 100))
print("Inflation Forecast: {:.2f}%".format(inf * 100))

print("\nRequired Dividend Growth Schedule:\n")
print(result_df.to_string(index=False))
