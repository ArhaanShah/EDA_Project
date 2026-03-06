# Time Series Analysis & EDA - Mutual Fund Data

A comprehensive exploratory data analysis and time series analysis project for analyzing mutual fund performance, market share, and statistical trends.

## Project Overview

This project performs extensive data analysis on mutual fund datasets, including:
- **Asset Manager Market Share Analysis**: Visualization of AUM distribution across Asset Management Companies
- **Fund Performance Metrics**: Statistical analysis of fund returns and key performance indicators
- **Time Series Analysis**: Decomposition, trend analysis, and forecasting of fund performance over time
- **Statistical Tests & Modeling**: Correlation analysis, regression models, and predictive analytics

## Key Features

- **Data Loading & Preprocessing**: Automated data cleaning, type conversion, and validation
- **Fund Selection & Metrics**: Market share visualization and AUM analysis by asset manager
- **Performance Analysis**: Comprehensive metrics including returns, growth rates, and comparative analysis
- **Time Series Decomposition**: Trend, seasonality, and residual analysis
- **Statistical Modeling**: Correlation analysis, regression models, and trend identification
- **Visualization**: High-quality plots using Matplotlib and Seaborn for insights

## Requirements

### Python Libraries
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `seaborn` - Statistical data visualization
- `statsmodels` - Time series and statistical modeling
- `scipy` - Scientific computing (signal processing)
- `scikit-learn` - Machine learning models

### Python Version
- Python 3.7 or higher

## Installation

1. Clone or download the project
2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy scikit-learn
```

**Recommended**: Use `time_series_analysis_clean.py` for a working, concise analysis.

## Usage

### Interactive Notebook

Open and run the Jupyter notebook for an interactive analysis experience:
```bash
jupyter notebook time_series_analysis.ipynb
```

### Standalone Script (Recommended)

For a clean, sequential analysis, run the Python script:
```bash
python time_series_analysis_clean.py
```

**Note:** Ensure `mutual_fund_data.csv` is in the project directory before running.

The script will:
1. Load and validate the dataset
2. Perform data type conversions
3. Generate 9 comprehensive visualizations
4. Display summary statistics
5. Handle missing data automatically (with warnings)

## Dataset Requirements

### File Format
- **Filename**: `mutual_fund_data.csv` (place in project root directory)
- **Format**: CSV (comma-separated values)
- **Encoding**: UTF-8

### Required Columns

**Core Columns:**
- `Date` - Record date (format: **DD-MM-YYYY**)
- `Fund_Name` - Name of the mutual fund (string)
- `NAV` - Net Asset Value per unit (numeric)
- `Launch_Date` - Fund launch date (format: **DD-MM-YYYY**)
- `Plan` - Investment plan type (e.g., "Direct", "Regular")
- `AMC_Name` - Asset Management Company name (string)

**Financial Metrics (AUM & Costs):**
- `AUM (Crore)` - Assets Under Management in Indian Crores (numeric)
- `TER (%)` - Total Expense Ratio in percent (numeric)

**Return Metrics (at least one required):**
- `1W` - 1-week return (%)
- `1M` - 1-month return (%)
- `3M` - 3-month return (%)
- `6M` - 6-month return (%)
- `YTD` - Year-to-date return (%)
- `1Y` - 1-year annualized return (%)
- `2Y` - 2-year annualized return (%)
- `3Y` - 3-year annualized return (%)
- `5Y` - 5-year annualized return (%)

**Optional Columns (for enhanced analysis):**
- `YTM (%)` - Yield to Maturity
- `Beta` - Beta coefficient (market volatility measure)
- `Sharpe_Ratio` - Risk-adjusted return metric
- `Debt_Holding(%)` - Debt holdings percentage
- `Cash_Holding(%)` - Cash holdings percentage
- `Volatility` or `Standard_Deviation` - Daily volatility

### Data Notes
- All numeric columns should use `.` (dot) as decimal separator
- Date format must be strictly DD-MM-YYYY (e.g., 15-03-2020)
- Missing values will be automatically handled during processing
- Records should be sorted chronologically (one record per fund per date)

## Analysis Features

The script performs the following analyses:

### Section 2: Fund Selection & Value Metrics
- **Feature 1** - Market Share by AMC (Asset Manager concentration)
- **Feature 2** - Fund Age vs Scale (first-mover advantage analysis)
- **Feature 3** - Fund Efficiency (Total Expense Ratio ranking)
- **Feature 4** - Value for Money (Cost vs Return scatter plot)

### Section 3: Risk & Volatility Analysis
- **Feature 5** - Daily Volatility Analysis (Standard deviation of daily returns)
- **Feature 6** - Risk-Return Profile (1-year risk-adjusted performance)

### Section 4: Comparative Performance Analysis
- **Feature 7** - Multi-Horizon Returns (1W to 5Y comparison)
- **Feature 8** - Normalized NAV Growth (indexed portfolio comparison)
- **Feature 9** - Return Correlation Heatmap (diversification analysis)

### Section 5: Summary Statistics
- Top performers by 1-year return
- Dataset overview and data quality metrics

## Notes

- **Data Format**: All dates must be in DD-MM-YYYY format; numeric columns use dot (.) as decimal separator
- **Auto-handling**: Missing values and data type mismatches are automatically corrected during processing
- **Column Names**: Column names are automatically cleaned (whitespace trimmed)
- **Plotting**: All visualizations use consistent styling with matplotlib and seaborn
- **Error Reporting**: The script provides clear feedback when required columns are missing, allowing for targeted data fixes
- **Performance**: Analysis completes in seconds even for large datasets (1000+ funds)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: mutual_fund_data.csv` | Ensure CSV file is in the project root directory |
| Missing columns warning | Check [Dataset Requirements](#dataset-requirements) for required column names |
| Date conversion errors | Verify date format is exactly DD-MM-YYYY (e.g., 15-03-2020) |
| Plots not displaying | Ensure matplotlib backend is properly configured |
| Memory issues with large datasets | The script uses efficient pandas operations; reduce dataset size if still problematic |

## License

This project is for educational and analytical purposes.
