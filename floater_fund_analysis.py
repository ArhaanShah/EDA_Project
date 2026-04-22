import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.offsetbox import AnchoredText
import pmdarima as pm
import shap

warnings.simplefilter("ignore")

os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)

sns.set_theme(style="whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4), titlesize=18, titleweight='bold')
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)

file_name = 'floater_fund_data.csv'
df = pd.read_csv(file_name)

df.columns = df.columns.str.strip()

print(df.info())
print(df.head())

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Launch Date'] = pd.to_datetime(df['Launch Date'], format='%d-%m-%Y', errors='coerce')

all_cols = df.columns
cols_to_exclude = ['Date', 'Launch Date', 'Fund Name', 'Plan', 'AMC Name']
numeric_cols = [col for col in all_cols if col not in cols_to_exclude]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

data_static = df.sort_values(by='Date', ascending=False).drop_duplicates(subset='Fund Name', keep='first')
data_static.set_index('Fund Name', inplace=True)
print(data_static)

aum_by_amc = data_static.groupby('AMC Name')['AUM (Crore)'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
ax = aum_by_amc.plot(kind='bar', color='skyblue')
ax.set_title('Total Assets Under Management (AUM) by AMC')
ax.set_ylabel('AUM (in Crore)')
ax.set_xlabel('AMC Name')
ax.tick_params(axis='x', rotation=45)
ax.bar_label(ax.containers[0], fmt='%.0f Cr')
plt.show()

age_sorted = data_static.sort_values(by='Launch Date', ascending=True)
plt.figure(figsize=(9, 5))
ax = sns.barplot(data=age_sorted, x='AUM (Crore)', y=age_sorted.index, orient='h', palette='viridis')
ax.set_title('Fund Scale vs. Fund Age (Oldest to Newest)')
ax.set_xlabel('AUM (in Crore)')
ax.set_ylabel('Fund Name')
plt.show()

ter_sorted = data_static.sort_values('TER (%)', ascending=True)
plt.figure(figsize=(9, 5))
ax = sns.barplot(data=ter_sorted, x='TER (%)', y=ter_sorted.index, orient='h', palette='Reds_r')
ax.set_title('Fund Efficiency: Total Expense Ratio (TER %)')
ax.set_xlabel('TER (%)')
ax.set_ylabel('Fund Name')
ax.bar_label(ax.containers[0], fmt='%.2f%%')
plt.show()

plt.figure(figsize=(9, 5))
ax = sns.scatterplot(data=data_static, x='TER (%)', y='YTM (%)', hue=data_static.index, s=200, palette='tab10')
ax.set_title('Value for Money: Cost (TER) vs. Portfolio Yield (YTM)')
ax.set_xlabel('Total Expense Ratio (TER %)')
ax.set_ylabel('Yield to Maturity (YTM %)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

if pd.api.types.is_datetime64_any_dtype(data_static['Launch Date']) == False:
    data_static['Launch Date'] = pd.to_datetime(data_static['Launch Date'])

current_date = pd.Timestamp('now')
data_static['Fund_Age_Years'] = (current_date - data_static['Launch Date']).dt.days / 365.25
median_aum = data_static['AUM (Crore)'].median()
data_static['Risk_Category'] = np.where((data_static['AUM (Crore)'] < median_aum) & (data_static['Fund_Age_Years'] < 3), 'High Survival Risk', 'Stable')

return_cols = [col for col in data_static.columns if 'Return' in col]
target_metric = return_cols[0] if return_cols else 'YTM'

latest_date_indices = df.groupby('Fund Name')['Date'].idxmax()
df_snapshot = df.loc[latest_date_indices].reset_index(drop=True)

plt.figure(figsize=(9, 5))
funds = df_snapshot['Fund Name'].str.split(' - ').str[0]
plt.bar(funds, df_snapshot['% Debt Holding'], label='Debt Holding', color='#4c72b0')
plt.bar(funds, df_snapshot['% Cash Holding'], bottom=df_snapshot['% Debt Holding'], label='Cash Holding', color='#dd8452')
plt.xticks(rotation=45, ha='right')
plt.title('Portfolio Composition: Debt vs Cash Holdings', fontsize=16)
plt.ylabel('Percentage (%)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

returns_df = data_static[['1W (%)', '1M (%)', '3M (%)', '6M (%)', 'YTD (%)', '1Y (%)', '2Y (%)', '3Y (%)', '5Y (%)']]
returns_melted = returns_df.reset_index().melt('Fund Name', var_name='Horizon', value_name='Return (%)')
plt.figure(figsize=(9, 5))
ax = sns.barplot(data=returns_melted, x='Fund Name', y='Return (%)', hue='Horizon', palette='YlGnBu')
ax.set_title('Annualized Returns Across Time Horizons')
ax.set_ylabel('Annualized Return (%)')
ax.set_xlabel('Fund Name')
ax.tick_params(axis='x', rotation=90)
plt.legend(title='Horizon')
plt.show()

return_cols = ['1W (%)', '1M (%)', '3M (%)', '6M (%)', 'YTD (%)', '1Y (%)', '2Y (%)', '3Y (%)', '5Y (%)']
corr_matrix_returns = data_static[return_cols].corr()
plt.figure(figsize=(9, 5))
ax = sns.heatmap(corr_matrix_returns, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
ax.set_title('Correlation of Historical Returns')
plt.show()

pivot_nav = df.pivot(index='Date', columns='Fund Name', values='NAV')
pivot_nav = pivot_nav.ffill()
pivot_nav = pivot_nav.dropna(axis=1, thresh=len(pivot_nav)*0.5)

pivot_filled = pivot_nav.ffill().bfill()
nav_normalized = pivot_filled.apply(lambda x: (x / x.iloc[0]) * 100)
nav_normalized['Category Index'] = nav_normalized.mean(axis=1)

fund_cols = [c for c in nav_normalized.columns if c != 'Category Index']
num_funds = len(fund_cols)
color_map = matplotlib.colormaps['plasma'].resampled(num_funds)
colors = color_map(np.linspace(0, 1, num_funds))

plt.figure(figsize=(11, 6))
for i, col in enumerate(fund_cols):
    plt.plot(nav_normalized.index, nav_normalized[col], color=colors[i], linewidth=1.5, alpha=0.8, label=col)
plt.plot(nav_normalized.index, nav_normalized['Category Index'], color='black', linewidth=3, linestyle='--', label='Category Average', zorder=10)
plt.title('Wealth Index: Growth of 100 (All Funds vs. Average)', fontsize=16, fontweight='bold')
plt.ylabel('Value of Investment (Start = 100)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Funds")
plt.tight_layout()
plt.show()

fund_col = 'Fund Name'
pivot_nav = df.pivot_table(index='Date', columns=fund_col, values='NAV')
pivot_nav = pivot_nav.ffill().dropna()

window_days = 90
rolling_returns = pivot_nav.pct_change(window_days).apply(lambda x: (1 + x)**(365/window_days) - 1)

plt.figure(figsize=(24, 10))
sns.set_palette("bright")
for fund in rolling_returns.columns:
    plt.plot(rolling_returns.index, rolling_returns[fund], label=fund, linewidth=2, alpha=0.85)
plt.axhline(0, color='black', linewidth=2, linestyle='--')
plt.title(f'Consistency Check: {window_days}-Day Rolling Annualized Returns (All Funds)', fontsize=20, fontweight='bold')
plt.ylabel('Annualized Return', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7)
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
plt.minorticks_on()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4, fontsize=12)
plt.margins(x=0.01)
plt.tight_layout()
plt.show()

sip_cols = ['1Y', '2Y', '3Y', '5Y']
df_melted = df_snapshot.melt(id_vars=['Fund Name'], value_vars=sip_cols, var_name='Period', value_name='Return')
plt.figure(figsize=(12, 8))
sns.barplot(data=df_melted, x='Period', y='Return', hue='Fund Name', palette='viridis')
plt.title('Comparative SIP Returns across Time Horizons', fontsize=16)
plt.ylabel('Return (%)')
plt.xlabel('SIP Period')
plt.legend(title='Fund Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

volatility = df.groupby('Fund Name')['Change'].std().sort_values(ascending=False)
volatility_df = pd.DataFrame(volatility)
volatility_df.columns = ['Volatility']
print(volatility_df)

plt.figure(figsize=(10, 5))
ax = volatility_df.plot(kind='barh', color='tomato')
ax.set_title('Daily Volatility (Risk) by Fund')
ax.set_xlabel('Standard Deviation of Daily Change (%)')
ax.set_ylabel('Fund Name')
ax.invert_yaxis()
plt.show()

pivot_nav = df.pivot_table(index='Date', columns=fund_col, values='NAV')
pivot_nav = pivot_nav.ffill().dropna()
daily_returns_pct = pivot_nav.pct_change() * 100
volatility_series = daily_returns_pct.std()
most_volatile_fund = volatility_series.idxmax()
least_volatile_fund = volatility_series.idxmin()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(data=daily_returns_pct, x=most_volatile_fund, kde=True, bins=50, color='red', alpha=0.6, ax=axes[0])
axes[0].set_title(f'High Risk: {most_volatile_fund}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Daily Return (%)')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, linestyle='--', alpha=0.5)

sns.histplot(data=daily_returns_pct, x=least_volatile_fund, kde=True, bins=50, color='green', alpha=0.6, ax=axes[1])
axes[1].set_title(f'Low Risk: {least_volatile_fund}', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Daily Return (%)')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

risk_return_df = data_static.merge(volatility_df, left_index=True, right_index=True)
risk_return_df.columns = risk_return_df.columns.str.strip()
risk_return_df['Short Name'] = [' '.join(str(name).split()[:2]) for name in risk_return_df.index]

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(10, 13))
sns.barplot(x=risk_return_df['Short Name'], y=risk_return_df['1Y'], palette='viridis', ax=axes[0])
axes[0].set_title('1-Year Annualized Returns by Fund', fontsize=14)
axes[0].set_ylabel('1-Year Return (%)', fontsize=11)
axes[0].set_xlabel('')
axes[0].set_xticklabels([])

sns.barplot(x=risk_return_df['Short Name'], y=risk_return_df['Volatility'], palette='magma', ax=axes[1])
axes[1].set_title('Daily Volatility (Risk) by Fund', fontsize=14)
axes[1].set_ylabel('Volatility', fontsize=11)
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=45)
for label in axes[1].get_xticklabels():
    label.set_ha('right')

sns.scatterplot(data=risk_return_df, x='Volatility', y='1Y', hue='Short Name', s=150, palette='tab10', legend=False, ax=axes[2])
axes[2].set_title('Risk-Return Profile (1-Year)', fontsize=14)
axes[2].set_xlabel('Risk (Daily Volatility)', fontsize=11)
axes[2].set_ylabel('Reward (Return %)', fontsize=11)

y_range = risk_return_df['1Y'].max() - risk_return_df['1Y'].min()
y_offset = y_range * 0.02 if y_range > 0 else 0.001
for i, row in risk_return_df.iterrows():
    axes[2].text(row['Volatility'], row['1Y'] + y_offset, row['Short Name'], fontsize=9, ha='center')
axes[2].grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(h_pad=3.0)
plt.show()

def get_offset(col_name):
    y_range_val = risk_return_df[col_name].max() - risk_return_df[col_name].min()
    return y_range_val * 0.02 if y_range_val > 0 else 0.001

offset_1y = get_offset('1Y')
offset_3y = get_offset('3Y')
offset_5y = get_offset('5Y')

fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=False)
sns.scatterplot(data=risk_return_df, x='Volatility', y='1Y', hue='Short Name', s=200, palette='tab10', legend=False, ax=axes[0])
axes[0].set_title('Risk-Return Profile (1-Year)', fontsize=15)
axes[0].set_xlabel('Risk (Daily Volatility)', fontsize=12)
axes[0].set_ylabel('Reward (1-Year Annualized Return %)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.6)
for i, row in risk_return_df.iterrows():
    axes[0].text(row['Volatility'], row['1Y'] + offset_1y, row['Short Name'], fontsize=9, ha='center')

sns.scatterplot(data=risk_return_df, x='Volatility', y='3Y', hue='Short Name', s=200, palette='tab10', legend=False, ax=axes[1])
axes[1].set_title('Risk-Return Profile (3-Year)', fontsize=15)
axes[1].set_xlabel('Risk (Daily Volatility)', fontsize=12)
axes[1].set_ylabel('Reward (3-Year Annualized Return %)', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.6)
for i, row in risk_return_df.iterrows():
    axes[1].text(row['Volatility'], row['3Y'] + offset_3y, row['Short Name'], fontsize=9, ha='center')

sns.scatterplot(data=risk_return_df, x='Volatility', y='5Y', hue='Short Name', s=200, palette='tab10', legend=False, ax=axes[2])
axes[2].set_title('Risk-Return Profile (5-Year)', fontsize=15)
axes[2].set_xlabel('Risk (Daily Volatility)', fontsize=12)
axes[2].set_ylabel('Reward (5-Year Annualized Return %)', fontsize=12)
axes[2].grid(True, linestyle='--', alpha=0.6)
for i, row in risk_return_df.iterrows():
    axes[2].text(row['Volatility'], row['5Y'] + offset_5y, row['Short Name'], fontsize=9, ha='center')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.scatterplot(data=risk_return_df, x='Volatility', y='3Y', hue='Short Name', s=200, palette='tab10', legend=False)
ax.set_title('Risk-Return Profile (3-Year)', fontsize=15)
ax.set_xlabel('Risk (Daily Volatility)', fontsize=12)
ax.set_ylabel('Reward (3-Year Annualized Return %)', fontsize=12)
y_range = risk_return_df['3Y'].max() - risk_return_df['3Y'].min()
y_offset = y_range * 0.02 if y_range > 0 else 0.001
for i, row in risk_return_df.iterrows():
    ax.text(row['Volatility'], row['3Y'] + y_offset, row['Short Name'], fontsize=9, ha='center')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.scatterplot(data=risk_return_df, x='Volatility', y='5Y', hue=risk_return_df.index, s=200, palette='tab10', legend=False)
ax.set_title('Risk-Return Profile (5-Year)')
ax.set_xlabel('Risk (Daily Volatility)')
ax.set_ylabel('Reward (5-Year Annualized Return %)')
for i, row in risk_return_df.iterrows():
    ax.text(row['Volatility'] + 0.001, row['5Y'], i, fontsize=9)
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.regplot(data=risk_return_df, x='TER (%)', y='Volatility', scatter_kws={'s': 100}, line_kws={'color': 'green'})
ax.set_title('Do Higher Fees Buy Safety? (Cost vs. Risk)', fontsize=14)
ax.set_xlabel('Cost (TER %)', fontsize=12)
ax.set_ylabel('Risk (Daily Volatility)', fontsize=12)
for i, row in risk_return_df.iterrows():
    ax.text(row['TER (%)'], row['Volatility'], i.split(' - ')[0], fontsize=9)
plt.show()

fund_col = 'Fund Name'
pivot_nav = df.pivot_table(index='Date', columns=fund_col, values='NAV')
pivot_nav = pivot_nav.ffill().dropna()
pivot_change = pivot_nav.pct_change().dropna()
skew = pivot_change.skew()
kurt = pivot_change.kurt()
dist_stats = pd.DataFrame({'Skewness': skew, 'Kurtosis': kurt})
print(dist_stats.sort_values('Kurtosis', ascending=False).head(5))

volatility_rank = pivot_change.std().sort_values().index
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Change', y='Fund Name', order=volatility_rank, palette='coolwarm', showfliers=True)
plt.title('Daily Return Distribution & Outliers (Fat Tails Check)', fontsize=14)
plt.xlabel('Daily Change (%)')
plt.ylabel('Fund Name')
plt.grid(True, axis='x', alpha=0.3)
plt.show()

fund_col = 'Fund Name' if 'Fund Name' in df.columns else 'Fund_Name'
df_clean = df.drop_duplicates(subset=['Date', fund_col])
pivot_nav = df_clean.pivot(index='Date', columns=fund_col, values='NAV')
pivot_nav = pivot_nav.ffill().dropna()
rolling_max = pivot_nav.cummax()
drawdown_series = (pivot_nav - rolling_max) / rolling_max
max_dd = drawdown_series.min()

pivot_change = pivot_nav.pct_change()
var_95 = pivot_change.quantile(0.05)
cvar_95 = pivot_change[pivot_change <= var_95].mean()
risk_metrics = pd.DataFrame({'Max Drawdown': max_dd, 'VaR (95%)': var_95, 'CVaR (Avg Worst Case)': cvar_95})
risk_metrics = risk_metrics.sort_values('Max Drawdown', ascending=False)
print(risk_metrics)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_snapshot, x='Standard Deviation', y='3Y', size='AUM (Crore)', sizes=(100, 1000), hue='Fund Name', alpha=0.7)
for i in range(df_snapshot.shape[0]):
    plt.text(df_snapshot['Standard Deviation'].iloc[i]+0.001, df_snapshot['3Y'].iloc[i], df_snapshot['Fund Name'].iloc[i].split()[0], fontsize=9)
plt.title('Risk vs Return Analysis (Bubble Size = AUM)', fontsize=16)
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return (3Y SIP %)')
plt.grid(True, linestyle='--')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

df_sharpe_sorted = df_snapshot.sort_values('Sharpe Ratio', ascending=False)
df_alpha_sorted = df_snapshot.sort_values("Jension's Alpha", ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(ax=axes[0], data=df_sharpe_sorted, y='Fund Name', x='Sharpe Ratio', palette='magma', errorbar=None)
axes[0].set_title('Sharpe Ratio (Higher is Better)', fontsize=14)
axes[0].set_ylabel('')
sns.barplot(ax=axes[1], data=df_alpha_sorted, y='Fund Name', x="Jension's Alpha", palette='coolwarm', errorbar=None)
axes[1].set_title("Jensen's Alpha (Outperformance vs Benchmark)", fontsize=14)
axes[1].set_ylabel('')
axes[1].set_yticks([])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(ax=axes[0], data=df_snapshot.sort_values('Beta', ascending=True), y='Fund Name', x='Beta', palette='Blues_d')
axes[0].set_title('Beta (Lower is Less Volatile vs Market)', fontsize=14)
axes[0].set_ylabel('')
axes[0].axvline(1, color='red', linestyle='--', alpha=0.5, label='Market Benchmark (1.0)')
axes[0].legend()
sns.barplot(ax=axes[1], data=df_snapshot.sort_values("Treynor's Ratio", ascending=False), y='Fund Name', x="Treynor's Ratio", palette='viridis')
axes[1].set_title("Treynor's Ratio (Higher is Better)", fontsize=14)
axes[1].set_ylabel('')
axes[1].set_yticks([])
plt.tight_layout()
plt.show()

risk_free_rate = 0.05
df_sharpe = df_snapshot.copy()
df_sharpe['Annual_Return'] = df_sharpe['1Y'] / 100.0
df_sharpe['Annual_Vol'] = df_sharpe['Standard Deviation'] * np.sqrt(252) / 100.0
df_sharpe['Sharpe'] = (df_sharpe['Annual_Return'] - risk_free_rate) / df_sharpe['Annual_Vol']
fund_col = 'Fund_Name' if 'Fund_Name' in df.columns else 'Fund Name'
pivot_nav = df.pivot_table(index='Date', columns=fund_col, values='NAV').sort_index().ffill()
daily_ret = pivot_nav.pct_change()
rolling_window = 90
rolling_rf = (1 + risk_free_rate) ** (1/252) - 1
rolling_sharpe = ((daily_ret.rolling(rolling_window).mean() - rolling_rf) / daily_ret.rolling(rolling_window).std()) * np.sqrt(252)
err_bars = rolling_sharpe.std().reindex(df_sharpe['Fund Name']).values
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df_sharpe.sort_values('Sharpe', ascending=False), y='Fund Name', x='Sharpe', palette='magma')
ax.errorbar(df_sharpe.sort_values('Sharpe', ascending=False)['Sharpe'], range(len(df_sharpe)), xerr=err_bars[df_sharpe.sort_values('Sharpe', ascending=False).index], fmt='none', ecolor='black', capsize=3)
ax.set_title('Annualized Sharpe Ratio (Error Bars from 90-day Rolling)', fontsize=14)
ax.set_xlabel('Sharpe')
ax.set_ylabel('Fund')
plt.tight_layout()
plt.show()

size_metric = 'AUM (Crore)' if 'AUM (Crore)' in df_snapshot.columns else ('Sharpe Ratio' if 'Sharpe Ratio' in df_snapshot.columns else None)
sizes = None
if size_metric:
    sizes = (100 + 900 * (df_snapshot[size_metric] - df_snapshot[size_metric].min()) / (df_snapshot[size_metric].max() - df_snapshot[size_metric].min() + 1e-9))
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_snapshot, x='Beta', y='Treynor\'s Ratio', size=sizes if sizes is not None else None, sizes=(100, 1000), hue='Fund Name', alpha=0.7)
for i in range(df_snapshot.shape[0]):
    plt.text(df_snapshot['Beta'].iloc[i]+0.01, df_snapshot['Treynor\'s Ratio'].iloc[i], df_snapshot['Fund Name'].iloc[i].split()[0], fontsize=9)
plt.title('Treynor Ratio vs Beta (Bubble Size = AUM or Sharpe)', fontsize=16)
plt.xlabel('Beta (Systematic Risk)')
plt.ylabel('Treynor Ratio (Reward per Unit of Beta)')
plt.grid(True, linestyle='--')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

mean_window = 30
vol_window = 90
for fund in daily_ret.columns:
    series = daily_ret[fund].dropna()
    rolling_mean = series.rolling(mean_window).mean() * 252
    rolling_vol = series.rolling(vol_window).std() * np.sqrt(252)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(rolling_mean.index, rolling_mean, color='tab:blue', label='Rolling Mean (30d, annualized)')
    ax2.plot(rolling_vol.index, rolling_vol, color='tab:red', label='Rolling Vol (90d, annualized)')
    ax1.set_title(f'{fund}: Rolling Mean & Volatility')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Annualized Return', color='tab:blue')
    ax2.set_ylabel('Annualized Volatility', color='tab:red')
    ax1.grid(True, alpha=0.3)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.tight_layout()
    plt.show()

window = 90
rf_daily = (1 + 0.05) ** (1/252) - 1
roll_mean = daily_ret.rolling(window).mean()
roll_std = daily_ret.rolling(window).std()
roll_sharpe = ((roll_mean - rf_daily) / roll_std) * np.sqrt(252)
roll_sharpe_weekly = roll_sharpe.resample('W').mean().dropna(how='all')
roll_sharpe_weekly.index = roll_sharpe_weekly.index.strftime('%Y-%m-%d')
plt.figure(figsize=(12, 6))
sns.heatmap(roll_sharpe_weekly.T, cmap='YlGnBu', center=0, xticklabels=13, cbar_kws={'label': 'Rolling Sharpe (90d, annualized)'})
plt.title('Rolling Sharpe Heatmap (Weekly Resampled)', fontsize=14)
plt.xlabel('Date (Week Ending)', fontsize=12)
plt.ylabel('Fund', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

rolling_max = pivot_nav.cummax()
drawdown = (pivot_nav - rolling_max) / rolling_max
max_dd = drawdown.min() * 100.0
plt.figure(figsize=(10, 6))
sns.barplot(x=max_dd.values, y=max_dd.index, palette='Reds')
plt.title('Max Drawdown (%) per Fund')
plt.xlabel('Max Drawdown (%)')
plt.ylabel('Fund')
plt.tight_layout()
plt.show()

for fund in drawdown.columns:
    series = drawdown[fund].dropna()
    worst = series.min()
    worst_date = series.idxmin()
    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series * 100.0, color='tab:red')
    plt.axvline(worst_date, color='black', linestyle='--', alpha=0.7)
    plt.title(f'{fund}: Drawdown Time-Series (Worst at {worst_date.date()} = {worst*100:.2f}%)')
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

pivot_nav.index = pd.to_datetime(pivot_nav.index)
recovery_days = {}
for fund in pivot_nav.columns:
    s = pivot_nav[fund].dropna()
    cummax = s.cummax()
    max_recovery = 0
    last_peak_date = s.index[0]
    for dt, val in s.items():
        if val >= cummax.loc[dt]:
            days_to_recover = (dt - last_peak_date).days
            max_recovery = max(max_recovery, days_to_recover)
            last_peak_date = dt
    ongoing_drawdown = (s.index[-1] - last_peak_date).days
    max_recovery = max(max_recovery, ongoing_drawdown)
    recovery_days[fund] = max_recovery
rec_df = pd.Series(recovery_days).sort_values(ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(rec_df.index, rec_df.values, color='steelblue')
plt.title('Time to Recovery (Max Days Underwater)', fontsize=14)
plt.xlabel('Days', fontsize=12)
plt.ylabel('')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

fund_col = 'Fund_Name' if 'Fund_Name' in df.columns else 'Fund Name'
fund_sel = df[fund_col].unique()[0]
ts_nav = df[df[fund_col] == fund_sel].set_index('Date')['NAV'].sort_index().asfreq('D').ffill()
stl = STL(ts_nav, period=30)
res = stl.fit()
fig = res.plot()
fig.suptitle(f'STL Decomposition (NAV): {fund_sel}')
plt.tight_layout()
plt.show()

monthly_ret = ts_nav.pct_change().resample('ME').sum().dropna()
stl_m = STL(monthly_ret, period=12)
res_m = stl_m.fit()
fig2 = res_m.plot()
fig2.suptitle(f'STL Decomposition (Monthly Returns): {fund_sel}')
plt.tight_layout()
plt.show()

def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    if y is None:
        y = x.shift(lag)
    if standardize:
        x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(x, y, **kwargs)
    ax.set_ylabel(f"Lag {lag}")
    ax.set_xlabel(x.name)
    corr = y.corr(x)
    at = AnchoredText(f"{corr:.2f}", loc="upper left", frameon=False)
    ax.add_artist(at)
    return ax

def plot_lags(x, y=None, lags=6, lagplot_kwargs={}, **kwargs):
    kwargs.setdefault('nrows', int(np.ceil(lags / 2)))
    kwargs.setdefault('ncols', 2)
    kwargs.setdefault('figsize', (kwargs['ncols'] * 6, kwargs['nrows'] * 2.5 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig

def plot_periodogram(ts, detrend='linear', ax=None):
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    frequencies, spectrum = periodogram(ts, fs=fs, detrend=detrend, window="boxcar", scaling='spectrum')
    if ax is None:
        _, ax = plt.subplots()
    ax.step(frequencies, spectrum, color="purple")
    ax.set_xscale("log")
    periods = [1, 2, 7, 14, 30, 90, 180, 365]
    xticks_freq = [1/p for p in periods]
    xticklabels_period = ['Daily', 'Bi-daily', 'Weekly', 'Bi-weekly', 'Monthly', 'Quarterly', 'Semi-annual', 'Annual']
    sorted_pairs = sorted(zip(xticks_freq, xticklabels_period), key=lambda x: x[0])
    xticks_freq_sorted, xticklabels_period_sorted = zip(*sorted_pairs)
    ax.set_xticks(list(xticks_freq_sorted))
    ax.set_xticklabels(list(xticklabels_period_sorted), rotation=30)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax

def make_lags(ts, lags, lead_time=1):
    return pd.concat({f'y_lag_{i}': ts.shift(i) for i in range(lead_time, lags + lead_time)}, axis=1)

fund_name = 'ICICI Prudential Floating Interest Fund Direct - Growth'
ts_df = df[df['Fund Name'] == fund_name].copy()
ts_df = ts_df.set_index('Date').sort_index()
ts_nav = ts_df['NAV'].to_period('D')

moving_average_365 = ts_nav.rolling(window=365, center=True, min_periods=183).mean()
ax = ts_nav.plot(style='.', color='0.5', markersize=1, title=f'{fund_name} - 365-Day Moving Average')
moving_average_365.plot(ax=ax, linewidth=3, label='365-Day Moving Average')
ax.legend()
plt.show()

y = ts_nav.copy()
y.index = y.index.to_timestamp()
df_trend = pd.DataFrame()
df_trend['time'] = np.arange(len(y.index))
model_lin = LinearRegression()
model_lin.fit(df_trend[['time']], y)
y_pred_lin = pd.Series(model_lin.predict(df_trend[['time']]), index=y.index)
ax = y.plot(style='.', color='0.5', markersize=1, title='NAV vs. Linear Trend Model')
y_pred_lin.plot(ax=ax, linewidth=3, label='Linear Trend')
ax.legend()
plt.show()

df_trend['time_sq'] = df_trend['time']**2
model_poly = LinearRegression()
model_poly.fit(df_trend, y)
y_pred_poly = pd.Series(model_poly.predict(df_trend), index=y.index)
ax = y.plot(style='.', color='0.5', markersize=1, title='NAV vs. Quadratic Trend Model')
y_pred_poly.plot(ax=ax, linewidth=3, label='Quadratic Trend')
ax.legend()
plt.show()

y = ts_nav.copy()
y.index = y.index.to_timestamp()
df_trend = pd.DataFrame(index=y.index)
df_trend['time'] = np.arange(len(y.index))
df_trend['time_sq'] = df_trend['time']**2
y_lags = make_lags(y, lags=7).fillna(0.0)
X = df_trend.join(y_lags)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model_final = LinearRegression()
model_final.fit(X_train, y_train)
y_pred_final = pd.Series(model_final.predict(X_test), index=y_test.index)
plt.figure(figsize=(12, 7))
ax = y_train.plot(label='Train', title=f'NAV Forecast for {fund_name}')
y_test.plot(ax=ax, label='Test (Actual)', style='.', markersize=4)
y_pred_final.plot(ax=ax, label='Forecast (Predicted)', style='--')
ax.set_ylabel('NAV')
ax.legend()
plt.show()

if '1Y' in data_static.columns:
    target_fund_name = data_static.sort_values('1Y', ascending=False).index[0]
else:
    target_fund_name = data_static.index[0]

ts_data = pivot_nav[target_fund_name].asfreq('D').ffill()
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
ts_data = ts_data.bfill()
freqs, power = periodogram(ts_data.pct_change().dropna(), fs=365)
axes[0].semilogy(freqs, power, color='purple')
axes[0].set_title('Spectral Analysis: Periodogram (Hidden Cycles)', fontsize=14)
axes[0].set_xlabel('Frequency (Cycles per Year)')
axes[0].set_ylabel('Power Density')
axes[0].set_xlim(0, 52)
decomp = seasonal_decompose(ts_data, model='multiplicative', period=7)
decomp.trend.plot(ax=axes[1], title='Trend Component (Direction)', color='blue')
axes[1].set_ylabel('Trend NAV')
plot_acf(ts_data.pct_change().dropna(), lags=40, ax=axes[2], color='green')
axes[2].set_title('Autocorrelation (Memory of the Time Series)')
plt.tight_layout()
plt.show()

latest_aum = df.sort_values('Date').groupby('Fund Name')['AUM (Crore)'].last()
target_fund = latest_aum.idxmax()
ts_data = df[df['Fund Name'] == target_fund].set_index('Date')['NAV'].sort_index()
ml_df = pd.DataFrame({'Target': ts_data.pct_change()})
lags = [1, 5, 21]
for lag in lags:
    ml_df[f'Lag_{lag}'] = ml_df['Target'].shift(lag)
ml_df.dropna(inplace=True)
column_mapping = {'Lag_1': 'Yesterday\'s Return (1-Day)', 'Lag_5': 'Weekly Trend (5-Day)', 'Lag_21': 'Monthly Trend (21-Day)'}
ml_df.rename(columns=column_mapping, inplace=True)
X = ml_df.drop('Target', axis=1)
y = ml_df['Target']
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title(f'What drives returns for {target_fund}?', fontsize=16)
plt.xlabel('Average Impact on Model Prediction (Magnitude)', fontsize=12)
plt.show()

daily_returns = pivot_nav.pct_change()
mean_returns = daily_returns.mean() * 252
cov_matrix = daily_returns.cov() * 252
num_portfolios = 5000
results = np.zeros((3, num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(len(mean_returns))
    weights /= np.sum(weights)
    port_return = np.sum(mean_returns * weights)
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    results[0, i] = port_std
    results[1, i] = port_return
    results[2, i] = port_return / port_std
plt.figure(figsize=(10, 6))
plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.7)
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier: Risk vs. Reward Trade-offs', fontsize=16)
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Annualized Return (Reward)')
plt.grid(True, alpha=0.3)
plt.show()

fund_name = df['Fund Name'].unique()[0]
data = df[df['Fund Name'] == fund_name].copy()
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data.set_index('Date', inplace=True)
ts = data['NAV']
ts = ts.asfreq('D').ffill()
plt.figure(figsize=(10, 6))
plt.plot(ts)
plt.title(f'NAV History: {fund_name}')
plt.xlabel('Date')
plt.ylabel('NAV')
plt.show()

def check_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

check_stationarity(ts)

train_size = int(len(ts) * 0.8)
train = ts.iloc[:train_size]
test = ts.iloc[train_size:]

model_auto = pm.auto_arima(train, start_p=0, start_q=0, max_p=5, max_q=5, d=1, seasonal=False, trend='c', trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

test_days = 60
train = ts.iloc[:-test_days]
test = ts.iloc[-test_days:]
plt.figure(figsize=(10, 6))
plt.plot(train, label='Training Data')
plt.plot(test, label='Test Data', color='orange')
plt.legend()
plt.show()

forced_order = (0, 1, 1)
model_trend = ARIMA(train, order=forced_order, trend='t')
model_fit_trend = model_trend.fit()
forecast_obj = model_fit_trend.get_forecast(steps=len(test))
predictions = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int(alpha=0.05)
predictions.index = test.index
conf_int.index = test.index
plt.figure(figsize=(12, 6))
plt.plot(train.index[-300:], train.tail(300), label='Historical Data', color='black', alpha=0.6)
plt.plot(test.index, test, label='Actual NAV', color='green', linewidth=2)
plt.plot(predictions.index, predictions, label='Trend Forecast', color='red', linestyle='--', linewidth=2)
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.1, label='95% Confidence Interval')
plt.title('Corrected Forecast: ARIMA with Trend (Drift)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('NAV')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

rmse = np.sqrt(mean_squared_error(test, predictions))
mae = mean_absolute_error(test, predictions)

order = (1, 1, 0)
final_model = ARIMA(ts, order=order, trend='t')
final_model_fit = final_model.fit()
future_days = 30
future_forecast = final_model_fit.forecast(steps=future_days)
last_date = ts.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
future_forecast.index = future_dates
plt.figure(figsize=(12, 6))
plt.plot(ts.index[-100:], ts.tail(100), label='Historical Data')
plt.plot(future_forecast.index, future_forecast, label='Future Forecast (30 Days)', color='purple', linewidth=2)
plt.title('Future NAV Prediction (with Trend)', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

nav_pivot = df.pivot(index='Date', columns='Fund Name', values='NAV')
largest_fund = df_snapshot.sort_values('AUM (Crore)', ascending=False).iloc[0]['Fund Name']
fund_nav = nav_pivot[largest_fund].dropna()
ma_50 = fund_nav.rolling(window=50).mean()
plt.figure(figsize=(14, 6))
plt.plot(fund_nav.index, fund_nav, label='Actual NAV', alpha=0.5)
plt.plot(fund_nav.index, ma_50, label='50-Day Moving Average (Trend)', color='red', linestyle='--')
plt.title(f'Trend Analysis: {largest_fund}', fontsize=16)
plt.legend()
plt.show()