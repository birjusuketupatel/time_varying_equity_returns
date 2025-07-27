import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Shiller-Campbell style regression
# Independent Variable: lagged smoothed log of dividend yield
# Dependent Variable: average forward excess return
#   - nominal local excess return
#   - nominal USD-hedged excess return
#   - nominal USD-unhedged excess return
#   - real local excess return
#   - real USD-hedged excess return
#   - real USD-unhedged excess return

# === Define parameters ===
n = 10  # lookback for calculating D/P smoothing
m = 10  # lookahead for calculating average forward return
countries_to_include = []  

# === Load and prepare dataset ===
df = pd.read_csv("JSTdatasetR6.csv")
df = df.sort_values(['country', 'year'])

if len(countries_to_include) > 0:
    df = df[df['country'].isin(countries_to_include)]
    print(f"Restricting to countries: {countries_to_include}")
else:
    print("Including all countries")

# === Restrict to pre-buyback period ===
df = df[df['year'] < 1995]

# === Drop rows where smoothed log(D/P) cannot be computed ===
required_cols = ['eq_capgain', 'eq_dp', 'cpi']
df = df.dropna(subset=required_cols)

# === Build nominal price index ===
def build_index_with_gaps(data, return_col, index_col):
    data[index_col] = 1.0
    for country, group in data.groupby('country'):
        idx = group.index
        returns = group[return_col].fillna(0)
        cumulative = (1 + returns).cumprod()
        data.loc[idx, index_col] = cumulative
    return data

df = build_index_with_gaps(df, 'eq_capgain', 'equity_index_nom')

# === Compute nominal dividends ===
df['div_nom'] = df['equity_index_nom'] * df['eq_dp']

# === Compute real prices and real dividends ===
df['equity_index_real'] = 100 * df['equity_index_nom'] / df['cpi']
df['div_real'] = 100 * df['div_nom'] / df['cpi']

# === Compute n-year average real dividends (Shiller smoothing) ===
df = df.sort_values(['country', 'year'])
df['div_real_avg'] = df.groupby('country')['div_real'].transform(
    lambda x: x.rolling(window=n, min_periods=n).mean()
)

# === Compute lagged log(D/P) ===
df['dp_log'] = np.log(df['div_real_avg']) - np.log(df['equity_index_real'])
df['dp_log_lag1'] = df.groupby('country')['dp_log'].shift(1)

# === Compute USD FX return ===
df['xrusd'] = df.groupby('country')['xrusd'].ffill()
df['fx_return'] = df.groupby('country')['xrusd'].transform(lambda x: x.shift(1) / x)

# === Compute nominal local excess returns ===
df['eq_er_local_nom'] = df['eq_tr'] - df['bill_rate']

# === Compute nominal USD-hedged excess returns ===
df['eq_er_usd_hedged_nom'] = df['eq_er_local_nom'] * df['fx_return']

# === Compute nominal USD-unhedged excess return ===
us_rf = df[df['country'] == 'USA'][['year', 'bill_rate']].rename(columns={'bill_rate': 'us_bill_rate'})
df = df.merge(us_rf, on='year', how='left')

df['eq_er_usd_unhedged_nom'] = (1 + df['eq_tr']) * df['fx_return'] - (1 + df['us_bill_rate'])

# === Compute inflation rate ===
df['inflation'] = df.groupby('country')['cpi'].transform(lambda x: x / x.shift(1))

# === Compute real local excess returns ===
df['eq_er_local_real'] = df['eq_er_local_nom'] / df['inflation']

# === Compute real USD-hedged excess returns ===
us_inflation = df[df['country'] == 'USA'][['year', 'inflation']].rename(columns={'inflation': 'us_inflation'})
df = df.merge(us_inflation, on='year', how='left')

df['eq_er_usd_hedged_real'] = df['eq_er_usd_hedged_nom'] / df['us_inflation']

# === Compute real USD-unhedged excess return ===
df['eq_er_usd_unhedged_real'] = df['eq_er_usd_unhedged_nom'] / df['us_inflation']

def forward_avg_return(series, window):
    series = np.log1p(series)
    rev = series[::-1]
    return np.expm1(rev.rolling(window=window, min_periods=window).mean()[::-1])

# === Compute m-year forward average returns ===
return_cols = [
    'eq_er_local_nom',
    'eq_er_usd_hedged_nom',
    'eq_er_usd_unhedged_nom',
    'eq_er_local_real',
    'eq_er_usd_hedged_real',
    'eq_er_usd_unhedged_real',
]

for col in return_cols:
    new_col = f'{col}_fwd{m}'
    df[new_col] = df.groupby('country')[col].transform(lambda x: forward_avg_return(x, m))

# === Drop rows with missing lagged dp_log or forward returns ===
df_reg = df.dropna(subset=['dp_log_lag1'] + [f'{col}_fwd{m}' for col in return_cols])

# === Run regressions and plot ===
fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
axes = axes.flatten()

titles = {
    'eq_er_local_nom': 'Nominal Local Excess Return',
    'eq_er_usd_hedged_nom': 'Nominal USD-Hedged Excess Return',
    'eq_er_usd_unhedged_nom': 'Nominal USD-Unhedged Excess Return',
    'eq_er_local_real': 'Real Local Excess Return',
    'eq_er_usd_hedged_real': 'Real USD-Hedged Excess Return',
    'eq_er_usd_unhedged_real': 'Real USD-Unhedged Excess Return',
}

for i, col in enumerate(return_cols):
    fwd_col = f'{col}_fwd{m}'
    X = sm.add_constant(df_reg['dp_log_lag1'])
    y = df_reg[fwd_col]
    
    model = sm.OLS(y, X).fit()

    ax = axes[i]
    ax.scatter(df_reg['dp_log_lag1'], y, alpha=0.5, label='Data')
    ax.plot(df_reg['dp_log_lag1'], model.predict(X), color='red', label='OLS Fit')

    ax.set_title(f'{titles[col]}', fontsize=11)
    ax.set_xlabel('Lagged log(D/P)', labelpad=8)
    ax.set_ylabel(f'{m}-Year Avg Return', labelpad=8)
    ax.legend(fontsize=8)
    
    # === Print regression results to console ===
    print(f'\n=== OLS Regression: {titles[col]} ===')
    print(model.summary())

plt.show()
