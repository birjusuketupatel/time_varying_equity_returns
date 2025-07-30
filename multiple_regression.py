import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
import matplotlib.patches as mpatches
import pickle
import seaborn as sns
from scipy.stats import ttest_ind, levene

# === Define parameters ===
n = 10                      # lookback for calculating D/P smoothing
m = 10                      # lookahead for calculating average forward return
countries_to_include = []   # include only countries in this list, if empty include all
highlight_country = 'USA'   # highlight this country's data in the scatter plots

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

# === Compute annual inflation rate ===
df['inflation'] = df.groupby('country')['cpi'].transform(lambda x: (x / x.shift(1)) - 1)

# === Compute real equity total returns ===
df['eq_tr_real'] = ((1 + df['eq_tr']) / (1 + df['inflation'])) - 1

# === Compute m-year forward geometric average ===
def forward_avg_return(series, window):
    series = np.log1p(series)
    rev = series[::-1]
    return rev.rolling(window=window, min_periods=window).mean()[::-1]

df['fwd_log_avg_tr'] = df.groupby('country')['eq_tr_real'].transform(lambda x: forward_avg_return(x, m))

# === trims values beyond the a and (1 - a) percentile ===
def trim_outliers(series, a):
    lower = series.quantile(a)
    upper = series.quantile(1 - a)
    return series.where((series >= lower) & (series <= upper))


# === Compute forward real dividend growth over m years ===
df['fwd_log_real_div_growth'] = df.groupby('country')['div_real'].transform(
    lambda x: (np.log(x.shift(-m)) - np.log(x)) * (1 / m)
)

# === Compute forward average inflation ===
df['fwd_log_avg_inf'] = df.groupby('country')['inflation'].transform(lambda x: forward_avg_return(x, m))
df['fwd_log_avg_inf'] = trim_outliers(df['fwd_log_avg_inf'], 0.01)

# === Clean dataset for regression ===
df = df.replace([np.inf, -np.inf], np.nan)

# === Define dependent and independent variables ===
independent_vars = ['fwd_log_real_div_growth', 'dp_log_lag1', 'fwd_log_avg_inf']
dependent_var = 'fwd_log_avg_tr'

var_labels = {
    'fwd_log_real_div_growth': f'{m}-Year Fwd Log Real Dividend Growth',
    'dp_log_lag1': 'Smoothed and Lagged Log D/P',
    'fwd_log_avg_inf': f'{m}-Year Fwd Log Avg Inflation'
}

# === Select non-overlapping m-year intervals ===
df = df[df['year'] % m == 0]

# === Drop unecessary values ===
df_reg = df.dropna(subset=[dependent_var, 'country'] + independent_vars)
df_reg = df_reg[['country', 'year', dependent_var] + independent_vars]

# === Run one-variable regressions on forward returns individually ===
mask = df_reg['country'] == highlight_country

for ind_var in independent_vars:
    x = sm.add_constant(df_reg[ind_var])
    y = df_reg['fwd_log_avg_tr']
    
    model = sm.OLS(y, x).fit()
    print(model.summary())

    plt.figure()
    plt.scatter(df_reg.loc[~mask, ind_var], y[~mask], color='gray', alpha=0.4, label='Other Countries')
    plt.scatter(df_reg.loc[mask, ind_var], y[mask], color='green', alpha=0.8, label=highlight_country)
    plt.plot(df_reg[ind_var], model.predict(x), color='red', label='OLS Fit')

    plt.xlabel(var_labels[ind_var])
    plt.ylabel(f'{m}-Year Fwd Log Avg Real Equity Return')
    plt.title(f'{m}-Year Fwd Log Avg Real Equity Return vs. {var_labels[ind_var]}')

    plt.legend()
    plt.tight_layout()
    plt.show()

# === Run multi-variable regression ===
x = sm.add_constant(df_reg[independent_vars])
y = df_reg['fwd_log_avg_tr']
model = sm.OLS(y, x).fit()
print(model.summary())

# === Plot actual vs. predicted values ===
y_pred = model.predict(x)

plt.figure(figsize=(6, 6))
plt.scatter(y[~mask], y_pred[~mask], color='gray', alpha=0.4, label='Other Countries')
plt.scatter(y[mask], y_pred[mask], color='green', alpha=0.8, label=highlight_country)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Fit')

plt.xlabel(f'Actual {m}-Year Fwd Log Avg Return')
plt.ylabel(f'Predicted {m}-Year Fwd Log Avg Return')
plt.title(f'Actual vs. Predicted {m}-Year Fwd Avg Log Returns')

plt.legend()
plt.tight_layout()
plt.show()

# === Show that D/P ratio lacks predictive power on dividend growth ===
dp_median = df_reg['dp_log_lag1'].median()
high_dp = df_reg[df_reg['dp_log_lag1'] > dp_median]['fwd_log_real_div_growth'].dropna()
low_dp = df_reg[df_reg['dp_log_lag1'] <= dp_median]['fwd_log_real_div_growth'].dropna()

plt.figure(figsize=(10, 6))
sns.histplot(high_dp, color='blue', kde=True, stat='density', label='Above-Median D/P', alpha=0.6, bins=30)
sns.histplot(low_dp, color='orange', kde=True, stat='density', label='Below-Median D/P', alpha=0.6, bins=30)

plt.title(f'Distribution of Fwd {m}-Yr Log Real Dividend Growth\nHigh vs. Low Lagged Log D/P')
plt.xlabel(f'{m}-Year Forward Log Real Dividend Growth')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# === Statistical tests ===
t_stat, t_pval = ttest_ind(high_dp, low_dp, equal_var=False)
w_stat, w_pval = levene(high_dp, low_dp)

print("\n=== Statistical Tests: Dividend Growth (High vs. Low D/P) ===")
print(f"Levene’s Test for Equal Variance: W = {w_stat:.4f}, p = {w_pval:.4f}")
print(f"t-Test for Difference in Means:   t = {t_stat:.4f}, p = {t_pval:.4f}\n")

# === Regress D/P ratio on dividend growth ===
x = sm.add_constant(df_reg['dp_log_lag1'])
y = df_reg['fwd_log_real_div_growth']
model = sm.OLS(y, x).fit()
print(model.summary())

plt.figure()
plt.scatter(df_reg['dp_log_lag1'], y, color='gray', alpha=0.4, label='Data')
plt.plot(df_reg['dp_log_lag1'], model.predict(x), color='red', label='OLS Fit')

plt.xlabel('Smoothed and Lagged Log D/P')
plt.ylabel(f'{m}-Year Fwd Log Real Dividend Growth')
plt.title(f'{m}-Year Fwd Log Real Dividend Growth vs. Smoothed and Lagged Log D/P')

plt.legend()
plt.tight_layout()
plt.show()


# === Save data to csv ===
df_reg.to_csv('regression_raw_data.csv', index=False)

# === Save OLS model to be used by other scripts ===
with open('return_multivar_ols_model.pkl', 'wb') as f:
    pickle.dump(model, f)
