import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# === Define parameters ===
n = 10  # lookback for calculating D/P smoothing
m = 10  # lookahead for calculating average forward return
top_k = 3
benchmark_country = "USA"

# === Load and prepare dataset ===
df = pd.read_csv("JSTdatasetR6.csv")
df = df.sort_values(['country', 'year'])

# === Restrict to pre-buyback period ===
#df = df[df['year'] < 1995]

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
df['excess_return_local'] = df['eq_tr'] - df['bill_rate']

# === Compute nominal USD-hedged excess returns ===
df['excess_return_usd'] = df['excess_return_local'] * df['fx_return']

# Drop missing data
df = df.dropna(subset=['dp_log_lag1', 'excess_return_usd'])

# Build strategy dataframe
strategy_df = df[['year', 'country', 'dp_log_lag1', 'excess_return_usd']].copy()
strategy_df['rank'] = strategy_df.groupby('year')['dp_log_lag1'].rank(method='first', ascending=False)
strategy_df['selected'] = strategy_df['rank'] <= top_k

# Print country allocations each year
print("\n=== Annual Top-3 D/P Allocations ===")
allocations_by_year = (
    strategy_df[strategy_df['selected']]
    .groupby('year')['country']
    .apply(lambda x: ', '.join(x.sort_values()))
)
for year, countries in allocations_by_year.items():
    print(f"{year}: {countries}")

# Compute Top-k strategy returns
portfolio_returns = (
    strategy_df[strategy_df['selected']]
    .groupby('year')['excess_return_usd']
    .mean()
    .reset_index()
    .rename(columns={'excess_return_usd': 'strategy_return'})
)

# Benchmark returns (fixed country)
benchmark_returns = (
    df[df['country'] == benchmark_country]
    .groupby('year')['excess_return_usd']
    .mean()
    .reset_index()
    .rename(columns={'excess_return_usd': 'benchmark_return'})
)

# Merge and compute cumulative indices
returns = pd.merge(portfolio_returns, benchmark_returns, on='year', how='inner')
returns = returns.sort_values('year')
returns['strategy_index'] = (1 + returns['strategy_return']).cumprod()
returns['benchmark_index'] = (1 + returns['benchmark_return']).cumprod()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(returns['year'], np.log(returns['strategy_index']), label=f'Top-{top_k} D/P Strategy')
plt.plot(returns['year'], np.log(returns['benchmark_index']), label=f'{benchmark_country} Excess Return Strategy')
plt.title(f'USD Excess Return Strategy vs. {benchmark_country}')
plt.xlabel('Year')
plt.ylabel('Log Index Value (Start = $1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot only recent history
recent = returns[returns['year'] > 1990].copy()
recent['strategy_index'] /= recent['strategy_index'].iloc[0]
recent['benchmark_index'] /= recent['benchmark_index'].iloc[0]
plt.figure(figsize=(10, 6))
plt.plot(recent['year'], np.log(recent['strategy_index']), label=f'Top-{top_k} D/P Strategy')
plt.plot(recent['year'], np.log(recent['benchmark_index']), label=f'{benchmark_country} Excess Return Strategy')
plt.title(f'USD Excess Return Strategy vs. {benchmark_country}')
plt.xlabel('Year')
plt.ylabel('Log Index Value (Start = $1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute and print summary statistics
def summarize_returns(series, label):
    mean = series.mean()
    std = series.std()
    sharpe = mean / std if std != 0 else np.nan
    print(f"\n{label} Performance Summary:")
    print(f"  Mean Excess Return:   {mean:.4f}")
    print(f"  Std Dev (Volatility): {std:.4f}")
    print(f"  Sharpe Ratio:         {sharpe:.2f}")

summarize_returns(returns['strategy_return'], f"Top-{top_k} D/P Strategy")
summarize_returns(returns['benchmark_return'], f"{benchmark_country} Benchmark Strategy")
