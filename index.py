import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

selected_country = 'USA'
n = 10

# === Load and prepare dataset ===
df = pd.read_csv("JSTdatasetR6.csv")
df = df.sort_values(['country', 'year'])

required_cols = ['eq_tr', 'cpi']
df = df.dropna(subset=required_cols)

# === Compute annual inflation rate ===
df['inflation'] = df.groupby('country')['cpi'].transform(lambda x: (x / x.shift(1)) - 1)

# === Compute real equity total returns ===
df['eq_tr_real'] = ((1 + df['eq_tr']) / (1 + df['inflation'])) - 1

# === Build inflation-adjusted equity total return index ===
def build_index_with_gaps(data, return_col, index_col):
    data[index_col] = 1.0
    for country, group in data.groupby('country'):
        idx = group.index
        returns = group[return_col].fillna(0)
        cumulative = (1 + returns).cumprod()
        data.loc[idx, index_col] = cumulative
    return data

df = build_index_with_gaps(df, 'eq_tr_real', 'real_eq_tr_index')

# === Select country to analyze ===
df_country = df[df['country'] == selected_country].copy()

# === Plot real total return index over time ===
plt.figure(figsize=(10, 5))
plt.plot(df_country['year'], np.log(df_country['real_eq_tr_index']), marker='o')
plt.title(f'{selected_country} – Real Log Total Return Index Over Time')
plt.xlabel('Year')
plt.ylabel('Real Equity Total Return Index')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Summary statistics for each country ===
def geometric_mean(series):
    series = series.dropna()
    return (np.prod(1 + series))**(1 / len(series)) - 1 if len(series) > 0 else np.nan

summary_all = df.groupby('country')['eq_tr_real'].agg(
    GeometricMean=geometric_mean,
    Variance='var',
    Observations='count'
).reset_index()

summary_all['GeometricMean'] = summary_all['GeometricMean'].round(4)
summary_all['Variance'] = summary_all['Variance'].round(4)

print("\nSummary Statistics for Each Country:\n")
print(summary_all.sort_values(by='GeometricMean', ascending=False).to_string(index=False))

# === Overall summary statistics  ===
returns_all = df['eq_tr_real'].dropna()
geo_mean_all = geometric_mean(returns_all)
var_all = returns_all.var()
n_all = len(returns_all)

summary_overall = pd.DataFrame({
    'Metric': ['Geometric Mean', 'Variance', 'Observations'],
    'Value': [f"{geo_mean_all:.4f}", f"{var_all:.4f}", n_all]
})

print("\nOverall Summary Statistics:\n")
print(summary_overall.to_string(index=False))

# === Load and prepare dataset ===
df = pd.read_csv("JSTdatasetR6.csv")
df = df.sort_values(['country', 'year'])

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


# === Plot lagged log D/P ratio over time ===
df_country = df[df['country'] == selected_country].copy()
plt.figure(figsize=(10, 5))
plt.plot(df_country['year'], df_country['dp_log_lag1'], marker='o', color='darkorange')
plt.title(f'{selected_country} – Lagged Log(D/P) Ratio Over Time')
plt.xlabel('Year')
plt.ylabel('Lagged log(D/P)')
plt.grid(True)
plt.tight_layout()
plt.show()
