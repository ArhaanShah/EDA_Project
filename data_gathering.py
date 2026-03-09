"""
EDA Project - Data Gathering and Processing Script

This script processes mutual fund NAV history and summary data:
1. Merges multiple NAV history CSV files
2. Loads and cleans Excel files with returns data
3. Combines all data into unified datasets
4. Generates visualizations and saves processed data
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")


def load_nav_history_files(pattern="Dhan - *.csv"):
    """
    Load and merge NAV history files.
    
    Args:
        pattern: Glob pattern for NAV history files (default: "Dhan - *.csv")
    
    Returns:
        DataFrame with merged NAV data or None if no files found
    """
    nav_history_files = glob.glob(pattern)
    all_nav_data = []

    print(f"Found {len(nav_history_files)} NAV history files to merge.")

    for filename in nav_history_files:
        try:
            # Extract the fund name from the filename
            # Format: "Dhan - [Fund Name] NAV History - 5 Years.csv"
            fund_name = os.path.basename(filename).split(' NAV History')[0].split(' - ', 1)[1]

            # Read the individual CSV
            df = pd.read_csv(filename)

            # Add the new 'Fund Name' column
            df['Fund Name'] = fund_name

            # Add this dataframe to our list
            all_nav_data.append(df)
            print(f"Processed: {fund_name}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Concatenate all dataframes in the list into one single dataframe
    if all_nav_data:
        merged_nav_df = pd.concat(all_nav_data, ignore_index=True)
        print("\nSuccessfully merged all NAV history files.")
        return merged_nav_df
    else:
        print("\nNo NAV files were processed. Please check file names.")
        return None


def clean_nav_data(merged_nav_df):
    """
    Clean and process NAV data.
    
    Args:
        merged_nav_df: DataFrame with merged NAV history
    
    Returns:
        Cleaned DataFrame
    """
    if merged_nav_df is None:
        return None

    # Date Index Logic
    if 'Date' in merged_nav_df.columns:
        try:
            merged_nav_df['Date'] = pd.to_datetime(merged_nav_df['Date'])
        except Exception as e:
            print(f"Date parse error: {e}. Trying dayfirst=True...")
            try:
                merged_nav_df['Date'] = pd.to_datetime(merged_nav_df['Date'], dayfirst=True)
            except Exception as e2:
                print(f"Date conversion failed. Error: {e2}")

        merged_nav_df = merged_nav_df.set_index('Date')
    else:
        print("'Date' column not found, assuming it's already the index.")
        if not pd.api.types.is_datetime64_any_dtype(merged_nav_df.index):
            merged_nav_df.index = pd.to_datetime(merged_nav_df.index)

    # NAV Cleaning
    if 'NAV' in merged_nav_df.columns:
        merged_nav_df['NAV'] = merged_nav_df['NAV'].astype(str).str.replace(',', '', regex=False)
        merged_nav_df['NAV'] = pd.to_numeric(merged_nav_df['NAV'], errors='coerce')

    # Clean 'Change' column
    if 'Change' in merged_nav_df.columns:
        merged_nav_df['Change'] = pd.to_numeric(merged_nav_df['Change'], errors='coerce')

    # Clean 'Change(%)' column
    if 'Change(%)' in merged_nav_df.columns:
        merged_nav_df['Change(%)'] = merged_nav_df['Change(%)'].astype(str).str.replace('%', '', regex=False)
        merged_nav_df['Change(%)'] = pd.to_numeric(merged_nav_df['Change(%)'], errors='coerce')

    print("--- Cleaned NAV Data Info ---")
    print(merged_nav_df.info())

    print("\n--- Data Head After Cleaning ---")
    print(merged_nav_df.head())

    return merged_nav_df


def plot_nav_performance(merged_nav_df, output_file=None):
    """
    Plot NAV performance comparison for all funds.
    
    Args:
        merged_nav_df: Cleaned NAV DataFrame
        output_file: Optional file path to save the plot
    """
    if merged_nav_df is None:
        return

    print("Pivoting data for plotting...")

    # Pivot the data to have Dates as rows and Fund Names as columns
    nav_pivot = merged_nav_df.pivot_table(
        index=merged_nav_df.index,
        columns='Fund Name',
        values='NAV'
    )

    print("Generating NAV performance plot...")

    # Plot the pivoted data
    plt.figure(figsize=(16, 8))
    nav_pivot.plot(ax=plt.gca())

    plt.title('5-Year NAV Performance Comparison (All Funds)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('NAV (Net Asset Value)', fontsize=12)

    # Move the legend outside the plot
    plt.legend(title='Fund Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")

    plt.show()

    return nav_pivot


def plot_normalized_growth(nav_pivot, output_file=None):
    """
    Plot normalized growth starting from $100 for each fund.
    
    Args:
        nav_pivot: Pivoted NAV DataFrame
        output_file: Optional file path to save the plot
    """
    if nav_pivot is None:
        return

    # Normalize each column by its own first valid value
    normalized_nav = nav_pivot.apply(lambda x: (x / x.dropna().iloc[0]) * 100)

    print("Generating normalized growth plot...")

    # Plot the normalized data
    plt.figure(figsize=(16, 8))
    normalized_nav.plot(ax=plt.gca())

    plt.title('Normalized 5-Year Growth (All Funds)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Growth (Started at $100)', fontsize=12)

    # Move the legend outside the plot
    plt.legend(title='Fund Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")

    plt.show()

    return normalized_nav


def load_summary_files():
    """
    Load Excel files with fund summary data.
    
    Returns:
        Tuple of (annual_returns_df, trailing_returns_df, sip_performance_df)
    """
    annual_ret_df = None
    trailing_ret_df = None
    sip_ret_df = None

    # Load Annual Returns
    try:
        annual_ret_df = pd.read_excel(
            "Mutual-Fund-Annual-Returns.xlsx",
            skiprows=2,
            header=0
        )
        print("--- Loaded Mutual Fund Annual Returns ---")
        print(f"Columns: {list(annual_ret_df.columns)}")
        print(annual_ret_df.head())
    except FileNotFoundError:
        print("Could not find 'Mutual-Fund-Annual-Returns.xlsx'")

    # Load Trailing Returns
    try:
        trailing_ret_df = pd.read_excel(
            "Top-Performing-Mutual-Funds-Trailing-returns.xlsx",
            skiprows=3,
            header=[0, 1]  # Read the first two rows as a MultiIndex
        )
        print("\n--- Loaded Top Performing Trailing Returns ---")
        print(f"Columns: {list(trailing_ret_df.columns)}")
        print(trailing_ret_df.head())
    except FileNotFoundError:
        print("Could not find 'Top-Performing-Mutual-Funds-Trailing-returns.xlsx'")

    # Load SIP Performance
    try:
        sip_ret_df = pd.read_excel(
            "Top-Performing-Systematic-Investment-Plan.xlsx",
            skiprows=2,
            header=0
        )
        print("\n--- Loaded Top Performing SIP ---")
        print(f"Columns: {list(sip_ret_df.columns)}")
        print(sip_ret_df.head())
    except FileNotFoundError:
        print("Could not find 'Top-Performing-Systematic-Investment-Plan.xlsx'")

    return annual_ret_df, trailing_ret_df, sip_ret_df


def clean_summary_df(df, name_col, cols_to_clean, drop_other_cols=True):
    """
    Helper function to clean summary dataframes.
    
    Args:
        df: DataFrame to clean
        name_col: Column name containing fund names
        cols_to_clean: List of columns to clean
        drop_other_cols: Whether to drop non-essential columns
    
    Returns:
        Cleaned DataFrame or None if error occurs
    """
    if df is None:
        return None

    if name_col not in df.columns:
        print(f"Warning: Name column '{name_col}' not found in DataFrame. Skipping cleaning.")
        print(f"Available columns are: {list(df.columns)}")
        return None

    df_clean = df.copy()

    # Rename the name column to a standard 'Fund Name'
    df_clean = df_clean.rename(columns={name_col: 'Fund Name'})

    # If dropping other columns is requested
    if drop_other_cols:
        cols_to_keep = ['Fund Name'] + [col for col in cols_to_clean if col in df_clean.columns]
        df_clean = df_clean[cols_to_keep]

    # Set 'Fund Name' as the index
    df_clean = df_clean.set_index('Fund Name')

    # Clean specified columns
    for col in cols_to_clean:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].str.replace('%', '', regex=False)
            df_clean[col] = df_clean[col].str.replace('--', 'NaN', regex=False)
            df_clean[col] = df_clean[col].str.replace('NA', 'NaN', regex=False)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Drop rows where the index (Fund Name) is NaN or empty
    df_clean = df_clean.loc[df_clean.index.dropna()]
    df_clean = df_clean.loc[df_clean.index != '']

    # Handle duplicate index values
    if not df_clean.index.is_unique:
        print(f"Warning: Duplicate index values found. Dropping duplicates.")
        df_clean = df_clean.loc[~df_clean.index.duplicated(keep='first')]

    print(f"Successfully cleaned DataFrame based on '{name_col}'.")
    return df_clean


def clean_all_summary_files(annual_ret_df, trailing_ret_df, sip_ret_df):
    """
    Clean all summary dataframes.
    
    Returns:
        Tuple of cleaned dataframes
    """
    annual_ret_clean = None
    trailing_ret_clean = None
    sip_ret_clean = None

    # Clean Annual Returns
    if annual_ret_df is not None:
        annual_ret_df_renamed = annual_ret_df.rename(
            columns={'Unnamed: 2': 'AUM (Cr)', 'Unnamed: 3': 'Expense Ratio (%)'}
        )
        annual_cols = ['AUM (Cr)', 'Expense Ratio (%)', '2025', '2024', '2023', '2022', '2021']
        annual_ret_clean = clean_summary_df(annual_ret_df_renamed, 'Unnamed: 0', annual_cols)

    # Clean Trailing Returns
    if trailing_ret_df is not None:
        # Flatten MultiIndex columns
        new_cols = []
        for col in trailing_ret_df.columns:
            if 'Unnamed' in str(col[0]):
                new_cols.append(col[1])
            else:
                new_cols.append(f"{col[0]} {col[1]}")
        trailing_ret_df.columns = new_cols

        trailing_cols = [
            'AUM (Crore)',
            'Expense Ratio (%)',
            '1 Week Returns (%)',
            '1 Week Rank',
            '1 Month Returns (%)',
            '1 Month Rank',
            '3 Months Returns (%)',
            '3 Months Rank',
            '6 Months Returns (%)',
            '6 Months Rank',
            '1 Year Returns (%)',
            '1 Year Rank',
            '3 Years Returns (%)',
            '3 Years Rank',
            '5 Years Returns (%)',
            '5 Years Rank',
            '10 Years Returns (%)',
            '10 Years Rank',
            '10 Years YTD Ret (%)',
            '10 Years Since Launch Ret (%)'
        ]
        trailing_ret_clean = clean_summary_df(trailing_ret_df, 'Scheme Name', trailing_cols)

    # Clean SIP Returns
    if sip_ret_df is not None:
        sip_cols = ['AUM (Crore)', 'Expense Ratio (%)', 'Invested Amount', 'Current Value', 'Return (%)']
        sip_ret_clean = clean_summary_df(sip_ret_df, 'Scheme Name', sip_cols)

    return annual_ret_clean, trailing_ret_clean, sip_ret_clean


def apply_name_mapping(annual_ret_clean, trailing_ret_clean, sip_ret_clean):
    """
    Apply standardized fund name mapping across all dataframes.
    
    Args:
        annual_ret_clean: Cleaned annual returns DataFrame
        trailing_ret_clean: Cleaned trailing returns DataFrame
        sip_ret_clean: Cleaned SIP returns DataFrame
    
    Returns:
        Tuple of mapped dataframes
    """
    # Define fund name mapping for consistency
    name_mapping = {
        'HDFC Floating Rate Debt Fund Gr': 'HDFC Floating Rate Debt Fund - Growth',
        'ICICI Pru Floating Interest Fund Gr': 'ICICI Prudential Floating Interest Fund - Growth',
        'Kotak Floating Rate Reg Gr': 'Kotak Floating Rate Fund - Growth',
        'ABSL Floating Rate Reg Gr': 'Aditya Birla Sun Life Floating Rate Fund - Growth',
        'ABSL Floating Rate Retail Gr': 'Aditya Birla Sun Life Floating Rate Fund - Retail Growth',
    }

    try:
        if annual_ret_clean is not None:
            annual_ret_clean = annual_ret_clean.rename(index=name_mapping)
            print("Applied mapping to annual returns")

        if trailing_ret_clean is not None:
            trailing_ret_clean = trailing_ret_clean.rename(index=name_mapping)
            print("Applied mapping to trailing returns")

        if sip_ret_clean is not None:
            sip_ret_clean = sip_ret_clean.rename(index=name_mapping)
            print("Applied mapping to SIP returns")

        print("Name mapping complete.")

    except Exception as e:
        print(f"An error occurred during mapping: {e}")

    return annual_ret_clean, trailing_ret_clean, sip_ret_clean


def merge_summary_datasets(annual_ret_clean, trailing_ret_clean, sip_ret_clean):
    """
    Merge all cleaned summary datasets.
    
    Returns:
        Merged DataFrame or None if no data to merge
    """
    print("\nMerging cleaned summary files...")

    summary_dfs_to_merge = []

    if annual_ret_clean is not None:
        summary_dfs_to_merge.append(annual_ret_clean)
    if trailing_ret_clean is not None:
        summary_dfs_to_merge.append(trailing_ret_clean)
    if sip_ret_clean is not None:
        summary_dfs_to_merge.append(sip_ret_clean)

    if summary_dfs_to_merge:
        merged_summary_dataset = pd.concat(summary_dfs_to_merge, axis=1, join='outer')
        print("Successfully merged summary files.")
        print("\n--- Merged Summary Dataset Info ---")
        print(merged_summary_dataset.info())
        return merged_summary_dataset
    else:
        print("Error: No cleaned summary dataframes were found to merge.")
        return None


def rearrange_summary_columns(merged_summary_dataset):
    """
    Rearrange columns in merged summary dataset for better readability.
    
    Args:
        merged_summary_dataset: Merged DataFrame
    
    Returns:
        Rearranged DataFrame
    """
    if merged_summary_dataset is None:
        return None

    all_columns = list(merged_summary_dataset.columns)
    new_order = ['Fund Name']

    # 1. Add Fund Info (AUM, Expense Ratio)
    info_cols = ['AUM (Crore)', 'Expense Ratio (%)']
    for col in info_cols:
        if col in all_columns:
            new_order.append(col)

    # 2. Add Trailing Returns in order
    trailing_cols = [
        '1 Week Returns (%)', '1 Month Returns (%)', '3 Months Returns (%)',
        '6 Months Returns (%)', '1 Year Returns (%)', '3 Years Returns (%)',
        '5 Years Returns (%)', '10 Years Returns (%)', '10 Years YTD Ret (%)',
        '10 Years Since Launch Ret (%)'
    ]
    for col in trailing_cols:
        if col in all_columns:
            new_order.append(col)

    # 3. Add Annual Returns
    annual_cols = ['2025', '2024', '2023', '2022', '2021']
    for col in annual_cols:
        if col in all_columns:
            new_order.append(col)

    # 4. Add SIP Return
    if 'Return (%)' in all_columns:
        new_order.append('Return (%)')

    # 5. Add all remaining columns
    for col in all_columns:
        if col not in new_order:
            new_order.append(col)

    # Rearrange the DataFrame
    rearranged_df = merged_summary_dataset[new_order].copy()

    # Set Fund Name as index
    if 'Fund Name' in rearranged_df.columns:
        rearranged_df = rearranged_df.set_index('Fund Name')

    print("--- Successfully Rearranged Dataset ---")
    print(rearranged_df.info())

    print("\n--- Rearranged Dataset Head ---")
    pd.set_option('display.max_columns', None)
    print(rearranged_df.head())

    return rearranged_df


def save_datasets(merged_nav_df, nav_pivot, merged_summary_dataset, rearranged_df):
    """
    Save all processed datasets to CSV files.
    
    Args:
        merged_nav_df: Merged NAV data
        nav_pivot: Pivoted NAV data
        merged_summary_dataset: Merged summary dataset
        rearranged_df: Rearranged summary dataset
    """
    try:
        if merged_nav_df is not None:
            merged_nav_df.to_csv('Group#_Data_Merged_NAV_History.csv')
            print("Saved: Group#_Data_Merged_NAV_History.csv")

        if nav_pivot is not None:
            nav_pivot.to_csv('Group#_Data_NAV_Pivot.csv')
            print("Saved: Group#_Data_NAV_Pivot.csv")

        if merged_summary_dataset is not None:
            merged_summary_dataset.to_csv('Group#_Data_Merged_Summaries.csv')
            print("Saved: Group#_Data_Merged_Summaries.csv")

        if rearranged_df is not None:
            rearranged_df.to_csv('Group#_Data_Merged_Summaries_Rearranged.csv')
            print("Saved: Group#_Data_Merged_Summaries_Rearranged.csv")

    except Exception as e:
        print(f"An error occurred while saving: {e}")


def main():
    """
    Main execution function that orchestrates the entire data gathering pipeline.
    """
    print("=" * 80)
    print("EDA Project - Data Gathering and Processing")
    print("=" * 80)

    # Step 1: Load and process NAV history
    print("\n--- STEP 1: Processing NAV History ---")
    merged_nav_df = load_nav_history_files()

    if merged_nav_df is not None:
        merged_nav_df = clean_nav_data(merged_nav_df)

        # Generate visualizations
        print("\n--- STEP 2: Generating NAV Visualizations ---")
        nav_pivot = plot_nav_performance(merged_nav_df, 'nav_performance.png')
        plot_normalized_growth(nav_pivot, 'normalized_growth.png')
    else:
        nav_pivot = None

    # Step 3: Load summary files
    print("\n--- STEP 3: Loading Summary Files ---")
    annual_ret_df, trailing_ret_df, sip_ret_df = load_summary_files()

    # Step 4: Clean summary files
    print("\n--- STEP 4: Cleaning Summary Files ---")
    annual_ret_clean, trailing_ret_clean, sip_ret_clean = clean_all_summary_files(
        annual_ret_df, trailing_ret_df, sip_ret_df
    )

    # Step 5: Apply name mapping
    print("\n--- STEP 5: Applying Name Mapping ---")
    annual_ret_clean, trailing_ret_clean, sip_ret_clean = apply_name_mapping(
        annual_ret_clean, trailing_ret_clean, sip_ret_clean
    )

    # Step 6: Merge summary datasets
    print("\n--- STEP 6: Merging Summary Datasets ---")
    merged_summary_dataset = merge_summary_datasets(
        annual_ret_clean, trailing_ret_clean, sip_ret_clean
    )

    # Step 7: Rearrange columns
    print("\n--- STEP 7: Rearranging Columns ---")
    rearranged_df = rearrange_summary_columns(merged_summary_dataset)

    # Step 8: Save all datasets
    print("\n--- STEP 8: Saving Datasets ---")
    save_datasets(merged_nav_df, nav_pivot, merged_summary_dataset, rearranged_df)

    print("\n" + "=" * 80)
    print("Data gathering and processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
