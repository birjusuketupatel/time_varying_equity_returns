import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
selected_country = 'Portugal'
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
