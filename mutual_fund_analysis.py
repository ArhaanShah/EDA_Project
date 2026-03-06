# -*- coding: utf-8 -*-
"""
Time Series Analysis & EDA - Mutual Fund Data
Comprehensive analysis of mutual fund performance, market share, and statistical trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import simplefilter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.signal import periodogram

# ============================================================================
# SECTION 1: SETUP & DATA PREPARATION
# ============================================================================

# Configure environment
simplefilter("ignore")
sns.set_theme(style="whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4), titlesize=18, titleweight='bold')
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)

print("=" * 70)
print("TIME SERIES ANALYSIS & EDA - MUTUAL FUND DATA")
print("=" * 70)

# Load dataset
print("\n[1] Loading dataset...")
try:
    df = pd.read_csv('mutual_fund_data.csv')  # Update filename as needed
except FileNotFoundError:
    print("ERROR: Could not find 'mutual_fund_data.csv'")
    print("Please ensure the CSV file is in the project directory.")
    exit(1)

# Clean column names (remove whitespace)
df.columns = df.columns.str.strip()

# Display basic info
print(f"   ✓ Loaded {len(df)} records, {len(df.columns)} columns")
print(f"   ✓ Date range: {df['Date'].min()} to {df['Date'].max()}")

# Data type conversion
print("\n[2] Converting data types...")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
if 'Launch_Date' in df.columns:
    df['Launch_Date'] = pd.to_datetime(df['Launch_Date'], format='%d-%m-%Y', errors='coerce')

# Convert numeric columns
numeric_cols = [col for col in df.columns if col not in ['Date', 'Launch_Date', 'Fund_Name', 'Plan', 'AMC_Name']]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"   ✓ Numeric conversion complete ({len(numeric_cols)} columns)")

# Create snapshot (latest data for each fund)
print("\n[3] Creating snapshot (latest data per fund)...")
data_static = df.sort_values(by='Date', ascending=False).drop_duplicates(subset=['Fund_Name'], keep='first').copy()
data_static.set_index('Fund_Name', inplace=True)
print(f"   ✓ Snapshot contains {len(data_static)} funds")

# ============================================================================
# SECTION 2: FUND SELECTION & VALUE METRICS
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 2: FUND SELECTION & VALUE METRICS")
print("=" * 70)

# Feature 1: Market Share by AMC
print("\n[Feature 1] Market Share by Asset Manager...")
if 'AMC_Name' in data_static.columns and 'AUM (Crore)' in data_static.columns:
    aum_by_amc = data_static.groupby('AMC_Name')['AUM (Crore)'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = aum_by_amc.plot(kind='bar', color='skyblue')
    ax.set_title('Total AUM by Asset Management Company')
    ax.set_ylabel('AUM (Crore)')
    ax.set_xlabel('AMC Name')
    plt.xticks(rotation=45, ha='right')
    ax.bar_label(ax.containers[0], fmt='%.0f Cr')
    plt.tight_layout()
    plt.show()
    print("   ✓ Plot generated")
else:
    print("   ⚠ Required columns not found (AMC_Name, AUM (Crore))")

# Feature 2: Fund Age vs Scale
print("\n[Feature 2] Fund Age vs Fund Scale...")
if 'Launch_Date' in data_static.columns and 'AUM (Crore)' in data_static.columns:
    age_sorted = data_static.sort_values(by='Launch_Date', ascending=True)
    
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=age_sorted, x='AUM (Crore)', y=age_sorted.index, orient='h', palette='viridis')
    ax.set_title('Fund Scale vs Fund Age (Oldest to Newest)')
    ax.set_xlabel('AUM (Crore)')
    ax.set_ylabel('Fund Name')
    plt.tight_layout()
    plt.show()
    print("   ✓ Plot generated")
else:
    print("   ⚠ Required columns not found")

# Feature 3: Fund Efficiency (TER)
print("\n[Feature 3] Fund Efficiency - Total Expense Ratio...")
if 'TER (%)' in data_static.columns:
    ter_sorted = data_static.sort_values('TER (%)', ascending=True)
    
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=ter_sorted, x='TER (%)', y=ter_sorted.index, orient='h', palette='Reds_r')
    ax.set_title('Fund Efficiency: Total Expense Ratio')
    ax.set_xlabel('TER (%)')
    ax.set_ylabel('Fund Name')
    ax.bar_label(ax.containers[0], fmt='%.2f%%')
    plt.tight_layout()
    plt.show()
    print("   ✓ Plot generated")
else:
    print("   ⚠ TER (%) column not found")

# Feature 4: Value for Money (TER vs Return)
print("\n[Feature 4] Value for Money Analysis...")
if 'TER (%)' in data_static.columns and '1Y' in data_static.columns:
    plt.figure(figsize=(9, 5))
    ax = sns.scatterplot(data=data_static, x='TER (%)', y='1Y', hue=data_static.index, s=200, palette='tab10')
    ax.set_title('Value for Money: Cost (TER) vs Return (1Y)')
    ax.set_xlabel('Total Expense Ratio (%)')
    ax.set_ylabel('1-Year Return (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()
    print("   ✓ Plot generated")
else:
    print("   ⚠ Required columns not found")

# ============================================================================
# SECTION 3: RISK & VOLATILITY ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 3: RISK & VOLATILITY ANALYSIS")
print("=" * 70)

# Prepare time series data
print("\n[Preparing] Time Series Data...")
if 'Fund_Name' in df.columns and 'Date' in df.columns and 'NAV' in df.columns:
    df_ts = df.sort_values('Date').copy()
    pivot_nav = df_ts.pivot_table(index='Date', columns='Fund_Name', values='NAV')
    pivot_nav = pivot_nav.ffill().dropna(how='all')
    print(f"   ✓ Time series prepared for {len(pivot_nav.columns)} funds")
    
    # Feature 5: Daily Volatility
    print("\n[Feature 5] Daily Volatility Analysis...")
    daily_returns = pivot_nav.pct_change() * 100
    volatility = daily_returns.std().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 5))
    volatility.plot(kind='barh', color='tomato')
    plt.title('Daily Volatility (Risk) by Fund')
    plt.xlabel('Standard Deviation of Daily Returns (%)')
    plt.tight_layout()
    plt.show()
    print("   ✓ Plot generated")
    print("\n   Top 3 Riskiest Funds:")
    for i, (fund, vol) in enumerate(volatility.head(3).items(), 1):
        print(f"      {i}. {fund}: {vol:.2f}%")
    
    # Feature 6: Risk-Return Profile (1Y)
    print("\n[Feature 6] Risk-Return Profile Analysis...")
    if '1Y' in data_static.columns:
        risk_return_df = data_static[['1Y']].copy()
        risk_return_df['Volatility'] = volatility
        risk_return_df = risk_return_df.dropna()
        
        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(data=risk_return_df, x='Volatility', y='1Y', s=200, palette='viridis')
        ax.set_title('Risk-Return Profile (1-Year)')
        ax.set_xlabel('Risk (Daily Volatility)')
        ax.set_ylabel('Return (1-Year %)')
        
        # Add fund name labels
        for idx, row in risk_return_df.iterrows():
            ax.text(row['Volatility'] + 0.05, row['1Y'], idx.split(' - ')[0], fontsize=9)
        
        plt.tight_layout()
        plt.show()
        print("   ✓ Plot generated")
    else:
        print("   ⚠ 1Y column not found")
else:
    print("   ⚠ Required columns for time series not found")

# ============================================================================
# SECTION 4: COMPARATIVE PERFORMANCE ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 4: COMPARATIVE PERFORMANCE ANALYSIS")
print("=" * 70)

# Feature 7: Multi-Horizon Returns
print("\n[Feature 7] Multi-Horizon Historical Returns...")
return_cols = [col for col in data_static.columns if any(x in col for x in ['1W', '1M', '3M', '6M', 'YTD', '1Y', '2Y', '3Y', '5Y'])]
return_cols = [col for col in return_cols if col in data_static.columns]

if return_cols:
    returns_df = data_static[return_cols]
    returns_melted = returns_df.reset_index().melt('Fund_Name', var_name='Horizon', value_name='Return (%)')
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=returns_melted, x='Fund_Name', y='Return (%)', hue='Horizon', palette='YlGnBu')
    ax.set_title('Annualized Returns Across Time Horizons')
    ax.set_ylabel('Return (%)')
    ax.set_xlabel('Fund Name')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Horizon', bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.show()
    print("   ✓ Plot generated")
else:
    print("   ⚠ No return columns found in expected format")

# Feature 8: Normalized NAV Growth
print("\n[Feature 8] Normalized NAV Growth (Indexed to 100)...")
if 'Fund_Name' in df.columns and 'NAV' in df.columns:
    nav_normalized = pivot_nav.fillna(method='ffill').fillna(method='bfill')
    nav_normalized = nav_normalized.apply(lambda x: (x / x.iloc[0]) * 100 if x.iloc[0] > 0 else x)
    
    plt.figure(figsize=(12, 6))
    for col in nav_normalized.columns:
        plt.plot(nav_normalized.index, nav_normalized[col], label=col.split(' - ')[0], linewidth=1.5, alpha=0.8)
    
    plt.title('Wealth Index: Growth of 100 (All Funds)', fontsize=14)
    plt.ylabel('Value of Investment (Start = 100)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=9)
    plt.tight_layout()
    plt.show()
    print("   ✓ Plot generated")
else:
    print("   ⚠ Required columns not found")

# Feature 9: Historical Return Correlation
print("\n[Feature 9] Return Correlation Heatmap...")
if return_cols and len(return_cols) >= 2:
    corr_matrix = data_static[return_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Correlation of Historical Returns')
    plt.tight_layout()
    plt.show()
    print("   ✓ Plot generated")
else:
    print("   ⚠ Not enough return columns for correlation analysis")

# ============================================================================
# SECTION 5: SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 5: SUMMARY STATISTICS")
print("=" * 70)

print("\n[Fund Overview]")
print(f"Total Funds Analyzed: {len(data_static)}")
print(f"\nTop Performers (1-Year Return):")
if '1Y' in data_static.columns:
    top_performers = data_static.sort_values('1Y', ascending=False).head(3)
    for i, (idx, row) in enumerate(top_performers.iterrows(), 1):
        print(f"   {i}. {idx}: {row['1Y']:.2f}%")
else:
    print("   ⚠ 1Y column not found")

print("\nDataset Structure Summary:")
print(f"   - Total Records: {len(df)}")
print(f"   - Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"   - Columns: {len(df.columns)}")
print(f"   - Missing Values: {df.isnull().sum().sum()}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
