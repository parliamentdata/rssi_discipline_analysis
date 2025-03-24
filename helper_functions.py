import numpy as np
import yaml
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

def load_config():
    with open("config.yml", 'r') as file:
        return yaml.safe_load(file)

def create_db_engine(db_config):
    db_url = (
        f"postgresql+psycopg2://"
        f"{db_config['uid']}:{db_config['pwd']}@"
        f"{db_config['server']}:{db_config['port']}/"
        f"{db_config['database']}"
    )
    return create_engine(db_url)

def load_data(engine, query):
    with engine.connect() as connection:
        return pd.read_sql(query, connection)

def aggregate_columns(df, aggregation_map):
    for new_col, cols_to_sum in aggregation_map.items():
        df[cols_to_sum] = df[cols_to_sum].apply(pd.to_numeric, errors='coerce').fillna(0)
        df[new_col] = df[cols_to_sum].sum(axis=1)
    return df

def filter_rssi_schools(df, engine):
    rssi_cohorts_query = "SELECT rcdts FROM rssi_cohorts"
    rssi_cohorts_df = load_data(engine, rssi_cohorts_query)

    return rssi_cohorts_df.merge(df, on='rcdts', how='left')

def calculate_exclusion_ratio(
        df,
        suspension_col,
        enrollment_col,
        new_col_name,
        missing_data_value=None
    ):
    """
    Calculates the ratio of disciplinary actions to total enrollment for given columns.

    Parameters:
    - df: DataFrame to operate on.
    - suspension_cols: List of columns representing counts of disciplinary actions.
    - enrollment_col: Column name for enrollment data.
    - new_col_name: Name of the new column for the calculated ratio.
    - missing_data_value: Value to assign if calculation cannot be performed.

    Returns:
    - Modified DataFrame with the new ratio column.
    """

    df = df.copy()

    df.loc[:, suspension_col] = df[suspension_col].apply(pd.to_numeric, errors='coerce')
    df.loc[:, enrollment_col] = pd.to_numeric(df[enrollment_col], errors='coerce')

    df[new_col_name] = missing_data_value

    for idx, row in df.iterrows():
        if pd.isnull(row[enrollment_col]) and pd.isnull(row[suspension_col]):
            df.loc[idx, new_col_name] = np.nan
        elif pd.isnull(row[enrollment_col]) or row[enrollment_col] <= 0:
            df.loc[idx, new_col_name] = missing_data_value
        else:
            disciplinary_action_count = row[suspension_col]
            df.loc[idx, new_col_name] = disciplinary_action_count / row[enrollment_col]

    return df

def calculate_violent_inc_ratio(
        df,
        numerator_cols,
        enrollment_col,
        new_col_name,
        missing_data_value=None
    ):
    """
    Calculates the ratio of violent incidents to total enrollment for given columns.
    
    Parameters:
    - df: DataFrame to operate on.
    - numerator_cols: List of columns representing counts of violent incidents.
    - enrollment_col: Column name for enrollment data.
    - new_col_name: Name of the new column for the calculated ratio.
    - missing_data_value: Value to assign if calculation cannot be performed.

    Returns:
    - Modified DataFrame with the new ratio column.
    """

    df = df.copy()

    # Apply transformations
    df.loc[:, numerator_cols] = df[numerator_cols].apply(pd.to_numeric, errors='coerce')
    df.loc[:, enrollment_col] = pd.to_numeric(df[enrollment_col], errors='coerce')

    df[new_col_name] = missing_data_value

    for idx, row in df.iterrows():
        if pd.isnull(row[enrollment_col]) or row[enrollment_col] <= 0:
            df.loc[idx, new_col_name] = missing_data_value
        else:
            violence_incidents_count = row[numerator_cols].sum(skipna=True)
            df.loc[idx, new_col_name] = violence_incidents_count / row[enrollment_col]
    
    return df

def calculate_disciplinary_to_violence_ratio(
    df,
    disciplinary_cols,
    violence_cols,
    new_col_name,
    missing_data_value=None
):
    """
    Calculates the ratio of disciplinary actions to violent incidents for given columns.

    Parameters:
    - df: DataFrame to operate on.
    - disciplinary_cols: List of columns representing counts of disciplinary actions.
    - violence_cols: List of columns representing counts of violent incidents.
    - new_col_name: Name of the new column for the calculated ratio.
    - missing_data_value: Value to assign if calculation cannot be performed.

    Returns:
    - Modified DataFrame with the new ratio column.
    """
    df = df.copy()

    df.loc[:, disciplinary_cols] = df[disciplinary_cols].apply(pd.to_numeric, errors='coerce')
    df.loc[:, violence_cols] = df[violence_cols].apply(pd.to_numeric, errors='coerce')

    df[new_col_name] = missing_data_value

    for idx, row in df.iterrows():
        disciplinary_action_count = row[disciplinary_cols].sum(skipna=True)
        violence_incidents_count = row[violence_cols].sum(skipna=True)

        if pd.isnull(violence_incidents_count) or violence_incidents_count <= 0:
            df.loc[idx, new_col_name] = missing_data_value
        else:
            df.loc[idx, new_col_name] = disciplinary_action_count / violence_incidents_count

    return df

def generate_histograms(report_card_df):
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle(
        'Histograms for Disciplinary Action and Violent Incidents',
        fontsize=16
    )

    bin_sizes = [5, 10, 20, 100]  # Different bin sizes for comparison

    for i, bins in enumerate(bin_sizes):
        # Disciplinary Action Ratio
        axes[0, i].hist(
            report_card_df['disciplinary_action_ratio'].dropna(), bins=bins,
            color='blue', edgecolor='black', alpha=0.7
        )
        axes[0, i].set_title(f"Disciplinary Action Ratio (Bins = {bins})")
        axes[0, i].set_xlabel("Ratio")
        axes[0, i].set_ylabel("Frequency")
        axes[0, i].grid(axis='y', linestyle='--', alpha=0.7)

        # Violent Incidents Ratio
        axes[1, i].hist(
            report_card_df['violent_incidents_ratio'].dropna(), bins=bins,
            color='red', edgecolor='black', alpha=0.7
        )
        axes[1, i].set_title(f"Violent Incidents Ratio (Bins = {bins})")
        axes[1, i].set_xlabel("Ratio")
        axes[1, i].set_ylabel("Frequency")
        axes[1, i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    plt.show()


def normalize_scores(df, column, bin_size):
    """
    Normalize a given column to a 100-point scale and categorize into bins.

    Args:
        df (pd.DataFrame): The input dataframe.
        column (str): The column to normalize and categorize.
        bin_size (int): The bin size (5 or 10).

    Returns:
        pd.DataFrame: The dataframe with an additional column for normalized scores.
    """
    new_df = df.copy()
    max_ratio = df[column].max()
    min_ratio = df[column].min()

    new_df[column + f'_score_normal_{bin_size}'] = (
        100 * ((new_df[column] - min_ratio) / (max_ratio - min_ratio))
    )

    new_df[column + f'_score_ln_normal_{bin_size}'] = np.log1p(new_df[column])

    max_ln = new_df[column + f'_score_ln_normal_{bin_size}'].max()
    min_ln = new_df[column + f'_score_ln_normal_{bin_size}'].min()
    new_df[column + f'_score_ln_normal_{bin_size}'] = (
        100 * ((new_df[column + f'_score_ln_normal_{bin_size}'] - min_ln) / (max_ln - min_ln))
    )

    if bin_size == 5:
        labels = list(range(0, 101, 20))
    elif bin_size == 10:
        labels = list(range(0, 101, 10))
    elif bin_size == 20:
        labels = list(range(0, 101, 5))
    elif bin_size == 100:
        labels = list(range(0, 101, 1))
    else:
        raise ValueError("Unsupported bin size. Use 5, 10, 20, 100.")

    bins = labels + [float('inf')]

    new_df[column + f'_score_normal_{bin_size}'] = pd.cut(
        new_df[column + f'_score_normal_{bin_size}'],
        bins=bins, labels=labels[::-1], right=False, include_lowest=True
    )

    new_df[column + f'_score_ln_normal_{bin_size}'] = pd.cut(
        new_df[column + f'_score_ln_normal_{bin_size}'],
        bins=bins, labels=labels[::-1], right=False, include_lowest=True
    )

    return new_df

def custom_score(df, column, custom_bins_scores, bin_size):
    """
    Calculate custom scores based on bins and scores for a specific column.

    Args:
        df (pd.DataFrame): The input dataframe.
        column (str): The column to calculate scores for.
        custom_bins_scores (dict): A dictionary with 'bins' and 'scores'.

    Returns:
        pd.DataFrame: The dataframe with an additional column for custom scores.
    """
    new_df = df.copy()
    bins = custom_bins_scores[column]['bins']
    scores = custom_bins_scores[column]['scores']

    new_df[column + f'_score_custom_{bin_size}'] = pd.cut(
        new_df[column], bins=bins, labels=scores, right=True
    )
    new_df = new_df.dropna(subset=[column + f'_score_custom_{bin_size}'])
    new_df.loc[:, column + f'_score_custom_{bin_size}'] = (
        new_df.loc[:, column + f'_score_custom_{bin_size}'].astype(float)
    )

    return new_df

def quantile_scores(df, column, bin_size):
    """
    Determine the quantile scores for a given column.

    Args:
        df (pd.DataFrame): The input dataframe.
        column (str): The column name for which to calculate quantiles.
        bin_size (int): The bin size to use (5 or 10).

    Returns:
        pd.DataFrame: The dataframe with an additional column for quantile scores.
    """
    new_df = df.copy()
    if bin_size == 5:
        half_bin = (20 / 100) / 2
    else:
        half_bin = (bin_size/ 100) / 2

    quantile_steps = np.linspace(half_bin, 1 - half_bin, bin_size)
    quantiles = df[column].quantile([0] + quantile_steps.tolist() + [1])

    def assign_quantile(x):
        for i, q in enumerate(quantile_steps):
            if x <= quantiles[q]:
                if bin_size == 5:
                    return 100 - (i * 20)
                return 100 - (i * bin_size)
        return 0

    new_df[column + f'_score_quantile_{bin_size}'] = new_df[column].apply(assign_quantile)

    return new_df


def score_distribution_table(df, column, bin_size):
    """
    Creates a table displaying the count and ratio of values for each exact score 
    in different scoring methods.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        column (str): The column name to generate distribution tables for.
    
    Returns:
        pd.DataFrame: A formatted table with multi-index headers for method names 
        and subheaders for count and ratio.
    """
    if bin_size == 5:
        score_values = list(range(0, 101, 20))  # 0, 20, 40, 60, 80, 100
    elif bin_size == 10:
        score_values = list(range(0, 101, 10))  # 0, 10, 20, ..., 100
    else:
        raise ValueError("Unsupported bin size. Use 5 or 10.")

    scoring_methods = {
        "Quantile": f"{column}_score_quantile_{bin_size}",
        "Normal": f"{column}_score_normal_{bin_size}",
        "Log Normal": f"{column}_score_ln_normal_{bin_size}",
        "Custom": f"{column}_score_custom_{bin_size}"
    }

    results = []

    for score in score_values:
        row = {("Score", ""): score}

        for method, method_column in scoring_methods.items():
            filtered_df = df[df[method_column] == score]
            count = len(filtered_df)

            min_ratio = filtered_df[column].min()
            max_ratio = filtered_df[column].max()
            median_ratio = filtered_df[column].median()

            row[(method, "Count")] = count
            row[(method, "Min Ratio")] = (
                f"{min_ratio * 100:.1f}%" if not pd.isna(min_ratio) else "N/A"
            )
            row[(method, "Max Ratio")] = (
                f"{max_ratio * 100:.1f}%" if not pd.isna(max_ratio) else "N/A"
            )
            row[(method, "Median Ratio")] = (
                f"{median_ratio * 100:.1f}%" if not pd.isna(median_ratio) else "N/A"
            )

        results.append(row)

    df_result = pd.DataFrame(results)
    df_result.columns = pd.MultiIndex.from_tuples(df_result.columns)
    return df_result

def score_large_distribution_table(df, column, bin_size):
    """
    Creates a table displaying the count and ratio of values for each exact score 
    in different scoring methods.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        column (str): The column name to generate distribution tables for.
    
    Returns:
        pd.DataFrame: A formatted table with multi-index headers for method names 
        and subheaders for count and ratio.
    """
    if bin_size == 20:
        score_values = list(range(0, 101, 5))  # 0, 5, 10, ..., 100
    elif bin_size == 100:
        score_values = list(range(0, 101, 1))  # 0, 1, 2, ..., 100
    else:
        raise ValueError("Unsupported bin size. Use 20 or 100.")

    scoring_methods = {
        "Normal": f"{column}_score_normal_{bin_size}",
        "Log Normal": f"{column}_score_ln_normal_{bin_size}",
    }

    results = []

    for score in score_values:
        row = {("Score", ""): score}

        for method, method_column in scoring_methods.items():
            filtered_df = df[df[method_column] == score]
            count = len(filtered_df)

            min_ratio = filtered_df[column].min()
            max_ratio = filtered_df[column].max()
            median_ratio = filtered_df[column].median()

            row[(method, "Count")] = count
            row[(method, "Min Ratio")] = (
                f"{min_ratio * 100:.1f}%" if not pd.isna(min_ratio) else "N/A"
            )
            row[(method, "Max Ratio")] = (
                f"{max_ratio * 100:.1f}%" if not pd.isna(max_ratio) else "N/A"
            )
            row[(method, "Median Ratio")] = (
                f"{median_ratio * 100:.1f}%" if not pd.isna(median_ratio) else "N/A"
            )

        results.append(row)

    df_result = pd.DataFrame(results)
    df_result.columns = pd.MultiIndex.from_tuples(df_result.columns)
    return df_result

def plot_score_distribution(df, column, bin_size, methods=None):
    """
    Creates dynamic plots for the score distribution of each specified scoring method.
    Plots include count distribution per score, and min, max, and median ratios.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        column (str): The column name to generate distribution plots for.
        bin_size (int): The number of bins for the distribution.
        methods (list, optional): List of scoring methods to include.
        Defaults to ["Quantile", "Normal", "Custom"].
    """
    if methods is None:
        methods = ["Quantile", "Normal", "Custom"]
    if bin_size in (5, 10):
        df_result = score_distribution_table(df, column, bin_size).replace("N/A", np.nan)
    elif bin_size in (20, 100):
        df_result = score_large_distribution_table(df, column, bin_size).replace("N/A", np.nan)
    else:
        raise ValueError("Unsupported bin size. Use 5, 10, 20 or 100.")

    ratio_types = ["Count", "Min Ratio", "Max Ratio", "Median Ratio"]

    fig, axes = plt.subplots(1, len(ratio_types), figsize=(5 * len(ratio_types), 5), sharey=True)
    fig.suptitle(
        f"Score Distribution for {column.replace('_', ' ').title()} (Bins={bin_size})",
        fontsize=16
    )

    bar_width = 0.25
    index = np.arange(len(df_result[("Score", "")]))

    for ax, ratio_type in zip(axes, ratio_types):
        for i, method in enumerate(methods):
            if (method, ratio_type) in df_result:
                values = df_result[(method, ratio_type)].apply(
                    lambda x: float(x.strip('%')) / 100 if isinstance(x, str) else x
                )
                ax.barh(index + i * bar_width, values, height=bar_width, label=method)

        ax.set_yticks(index + bar_width)
        ax.set_yticklabels(df_result[("Score", "")])
        ax.set_title(f"{ratio_type} Distribution")
        ax.set_xlabel(ratio_type)
        ax.set_ylabel("Score")
        ax.legend(title="Method")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main(arg):
    CONFIG = load_config()
    DB_CONFIG = CONFIG['rssi-data-warehouse']['dataconnection']
    ENGINE = create_db_engine(DB_CONFIG)

    REPORT_CARD_QUERY = """
    SELECT rcdts, num_stdnt_enrlmnt
    FROM report_card WHERE school_year = '2023-24'
    """

    REPORT_CARD_DISC_QUERY = """
    SELECT rcdts, isbe_type, school_name, district, 
        num_stdnts_with_discipline_incidents,
        num_disc_incdts_viol_harm, num_disc_incdts_viol_noharm, 
        num_disc_incdts_firearm, num_disc_incdts_oth_wpn
    FROM report_card_disc WHERE school_year = '2023-24'
    """

    REPORT_CARD_DF = load_data(ENGINE, REPORT_CARD_QUERY)
    REPORT_CARD_DISC_DF = load_data(ENGINE, REPORT_CARD_DISC_QUERY)

    report_card_df = pd.merge(REPORT_CARD_DF, REPORT_CARD_DISC_DF, on='rcdts', how='inner')

    aggregation_map_disc = {
        'num_disc_incdts_violent': [
            'num_disc_incdts_viol_harm', 'num_disc_incdts_viol_noharm'
        ],
        'num_disc_incdts_other': [
            'num_disc_incdts_firearm', 'num_disc_incdts_oth_wpn'
        ]
    }
    report_card_df = aggregate_columns(report_card_df, aggregation_map_disc)

    report_card_df = calculate_exclusion_ratio(
        report_card_df, 'num_stdnts_with_discipline_incidents',
        'num_stdnt_enrlmnt', 'disciplinary_action_ratio', np.nan
    )

    report_card_df = calculate_violent_inc_ratio(
        report_card_df, ['num_disc_incdts_violent', 'num_disc_incdts_other'],
        'num_stdnts_with_discipline_incidents', 'violent_incidents_ratio', np.nan
    )

    report_card_df['disciplinary_action_ratio'] = pd.to_numeric(
        report_card_df['disciplinary_action_ratio'], errors='coerce'
    )
    report_card_df['violent_incidents_ratio'] = pd.to_numeric(
        report_card_df['violent_incidents_ratio'], errors='coerce'
    )

    if arg == 'rssi':
        report_card_df = filter_rssi_schools(report_card_df, ENGINE)

    summary_table = (
        report_card_df[['disciplinary_action_ratio', 'violent_incidents_ratio']]
        .describe(include='all')
        .round(3)
    )

    summary_table.loc['size'] = len(report_card_df)
    summary_table.loc['null count'] = report_card_df.isnull().sum()

    columns_to_score = [
        'disciplinary_action_ratio',
        'violent_incidents_ratio'
    ]

    custom_bins_scores_5 = {
        'disciplinary_action_ratio': {
            'bins': [0, 0.005, 0.02, 0.1, 0.3, 0.5, 1],
            'scores': [100, 80, 60, 40, 20, 0],
        },
        'violent_incidents_ratio': {
            'bins': [0, 0.25, 0.5, 1.0, 1.5, 3.0, float('inf')],
            'scores': [100, 80, 60, 40, 20, 0],
        }
    }

    custom_bins_scores_10 = {
        'disciplinary_action_ratio': {
            'bins': [0, 0.0025, 0.005, 0.01, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 1],
            'scores': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0],
        },
        'violent_incidents_ratio': {
            'bins': [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 3.0, float('inf')],
            'scores': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0],
        }
    }

    return summary_table, report_card_df, columns_to_score, custom_bins_scores_5, custom_bins_scores_10

# EXTRA:

    # # All Schools: Disciplinary-to-Violence Ratio
    # axes[2, 0].hist(
    #     report_card_df['disciplinary_to_violence_ratio'].dropna(), bins=10,
    #     color='orange', edgecolor='black', alpha=0.7
    # )
    # axes[2, 0].set_title("All Schools - Disciplinary-to-Violence Ratio")
    # axes[2, 0].set_xlabel("Ratio")
    # axes[2, 0].set_ylabel("Frequency")
    # axes[2, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # # RSSI Schools: Disciplinary-to-Violence Ratio
    # axes[2, 1].hist(
    #     rssi_schools_df['disciplinary_to_violence_ratio'].dropna(), bins=10,
    #     color='cyan', edgecolor='black', alpha=0.7
    # )
    # axes[2, 1].set_title("RSSI Schools - Disciplinary-to-Violence Ratio")
    # axes[2, 1].set_xlabel("Ratio")
    # axes[2, 1].set_ylabel("Frequency")
    # axes[2, 1].grid(axis='y', linestyle='--', alpha=0.7)

# report_card_df = calculate_disciplinary_to_violence_ratio(
#     report_card_df, ['num_stdnts_with_discipline_incidents'],
#     ['num_disc_incdts_violent', 'num_disc_incdts_other'],
#     'disciplinary_to_violence_ratio'
# )

# report_card_df['disciplinary_to_violence_ratio'] = pd.to_numeric(
#     report_card_df['disciplinary_to_violence_ratio'], errors='coerce'
# )

# rssi_schools_df = calculate_disciplinary_to_violence_ratio(
#     rssi_schools_df, ['num_stdnts_with_discipline_incidents'],
#     ['num_disc_incdts_violent', 'num_disc_incdts_other'],
#     'disciplinary_to_violence_ratio'
# )

# rssi_schools_df['disciplinary_to_violence_ratio'] = pd.to_numeric(
#     rssi_schools_df['disciplinary_to_violence_ratio'], errors='coerce'
# )

# all_schools_disciplinary_to_violence_quantiles = (
#     report_card_df['disciplinary_to_violence_ratio'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
#                                                     .round(3)
# )

# rssi_disciplinary_to_violence_quantiles = (
#     rssi_schools_df['disciplinary_to_violence_ratio'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
#                                                      .round(3)
# )

# def granular_rescale(df, column):
#     low_threshold = df[column].quantile(0.25)
#     high_threshold = df[column].quantile(0.75)

#     def rescale_low(x):
#         return 100 * (1 - (x - low_threshold) / (high_threshold - low_threshold))

#     def rescale_mid(x):
#         return 100 * (1 - (x - low_threshold) / (high_threshold - low_threshold))

#     def rescale_high(x):
#         return 100 * (1 - (x - high_threshold) / (df[column].max() - high_threshold))

#     df[column + '_score_granular'] = df[column].apply(
#         lambda x: rescale_low(x) if x <= low_threshold
#         else rescale_mid(x) if x <= high_threshold
#         else rescale_high(x))

#     bins = [float('-inf'), 0, 20, 40, 60, 80, 100]
#     labels = [0, 20, 40, 60, 80, 100]
#     df[column + '_score_granular'] = pd.cut(
#       df[column + '_score_granular'], bins=bins, labels=labels, right=True, include_lowest=True
#     )

#     return df

    # 'disciplinary_to_violence_ratio': {
    #     'bins': [0.2, 0.6, 1.0, 3.0, 5.0, 7.0, float('inf')],
    #     'scores': [100, 80, 60, 40, 20, 0],
    # },
