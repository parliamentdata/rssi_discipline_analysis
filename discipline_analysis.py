#!/usr/bin/env python
# coding: utf-8

# #### Import Libraries

# In[283]:


import pandas as pd
import yaml
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np


# #### Helper Functions

# In[284]:


def load_config():
    with open("config.yml", 'r') as file:
        return yaml.safe_load(file)


# In[285]:


def create_db_engine(db_config):
    db_url = f"postgresql+psycopg2://{db_config['uid']}:{db_config['pwd']}@{db_config['server']}:{db_config['port']}/{db_config['database']}"
    return create_engine(db_url)


# In[286]:


def load_data(engine, query):
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


# In[287]:


def aggregate_columns(df, aggregation_map):
    for new_col, cols_to_sum in aggregation_map.items():
        df[cols_to_sum] = df[cols_to_sum].apply(pd.to_numeric, errors='coerce').fillna(0)
        df[new_col] = df[cols_to_sum].sum(axis=1)
    return df


# In[288]:


def calculate_exclusion_ratio(
        df,
        suspension_cols,
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

    df.loc[:, suspension_cols] = df[suspension_cols].apply(pd.to_numeric, errors='coerce')
    df.loc[:, enrollment_col] = pd.to_numeric(df[enrollment_col], errors='coerce')

    df[new_col_name] = missing_data_value

    for idx, row in df.iterrows():
        if pd.isnull(row[enrollment_col]) or row[enrollment_col] <= 0:
            df.loc[idx, new_col_name] = missing_data_value
        else:
            disciplinary_action_count = row[suspension_cols].sum(skipna=True)
            df.loc[idx, new_col_name] = disciplinary_action_count / row[enrollment_col]

    return df


# In[289]:


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


# In[ ]:


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


# In[ ]:


def count_schools_by_quantile(df, column, quantiles, column_label="quantile"):
    """
    Calculates the number of schools in each quantile for a given column using
    precomputed quantiles.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        column (str): The column for which quantile counts are calculated.
        quantiles (Series): Precomputed quantile values.
        column_label (str): The name of the new column to store quantile labels.

    Returns:
        DataFrame: A DataFrame with quantile ranges and counts.
    """

    quantile_labels = [f"Q{i+1}" for i in range(len(quantiles) - 1)]

    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column])

    df[column_label] = pd.cut(
        df[column], bins=quantiles, labels=quantile_labels, include_lowest=True
    )

    quantile_counts = (
        df.groupby(column_label, observed=False)
          .size()
          .reset_index(name="school_count")
    )

    quantile_ranges = [
        f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]" for i in range(len(quantiles))
    ]

    quantile_summary = pd.DataFrame({
        "quantile": quantile_labels,
        "quantile_range": quantile_ranges,
        "school_count": quantile_counts["school_count"].values
    })

    return quantile_summary


# In[ ]:


def generate_histograms(report_card_df, rssi_schools_df):
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle(
        'Histograms for Disciplinary Action, Violent Incidents, and Disciplinary-to-Violence Ratios',
        fontsize=16
    )

    # All Schools: Disciplinary Action Ratio
    axes[0, 0].hist(
        report_card_df['disciplinary_action_ratio'].dropna(), bins=10,
        color='blue', edgecolor='black', alpha=0.7
    )
    axes[0, 0].set_title("All Schools - Disciplinary Action Ratio")
    axes[0, 0].set_xlabel("Ratio")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # All Schools: Violent Incidents Ratio
    axes[0, 1].hist(
        report_card_df['violent_incidents_ratio'].dropna(), bins=10, color='red',
        edgecolor='black', alpha=0.7
    )
    axes[0, 1].set_title("All Schools - Violent Incidents Ratio")
    axes[0, 1].set_xlabel("Ratio")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # RSSI Schools: Disciplinary Action Ratio
    axes[1, 0].hist(
        rssi_schools_df['disciplinary_action_ratio'].dropna(), bins=10,
        color='green', edgecolor='black', alpha=0.7
    )
    axes[1, 0].set_title("RSSI Schools - Disciplinary Action Ratio")
    axes[1, 0].set_xlabel("Ratio")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # RSSI Schools: Violent Incidents Ratio
    axes[1, 1].hist(
        rssi_schools_df['violent_incidents_ratio'].dropna(), bins=10, color='purple',
        edgecolor='black', alpha=0.7
    )
    axes[1, 1].set_title("RSSI Schools - Violent Incidents Ratio")
    axes[1, 1].set_xlabel("Ratio")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # All Schools: Disciplinary-to-Violence Ratio
    axes[2, 0].hist(
        report_card_df['disciplinary_to_violence_ratio'].dropna(), bins=10,
        color='orange', edgecolor='black', alpha=0.7
    )
    axes[2, 0].set_title("All Schools - Disciplinary-to-Violence Ratio")
    axes[2, 0].set_xlabel("Ratio")
    axes[2, 0].set_ylabel("Frequency")
    axes[2, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # RSSI Schools: Disciplinary-to-Violence Ratio
    axes[2, 1].hist(
        rssi_schools_df['disciplinary_to_violence_ratio'].dropna(), bins=10,
        color='cyan', edgecolor='black', alpha=0.7
    )
    axes[2, 1].set_title("RSSI Schools - Disciplinary-to-Violence Ratio")
    axes[2, 1].set_xlabel("Ratio")
    axes[2, 1].set_ylabel("Frequency")
    axes[2, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# #### Load the configuration and establish the database connection

# In[293]:


config = load_config()
db_config = config['rssi-data-warehouse']['dataconnection']
engine = create_db_engine(db_config)


# #### Define the queries and load the data

# In[ ]:


REPORT_CARD_QUERY = """
SELECT rcdts, num_stdnt_enrlmnt
FROM report_card
"""

REPORT_CARD_DISC_QUERY = """
SELECT rcdts, isbe_type, school_name, district, 
    num_stdnts_with_discipline_incidents,
    num_disc_incdts_viol_harm, num_disc_incdts_viol_noharm, 
    num_disc_incdts_firearm, num_disc_incdts_oth_wpn
FROM report_card_disc
"""

RSSI_COHORTS_QUERY = """
SELECT rcdts
FROM rssi_cohorts
"""

report_card_df = load_data(engine, REPORT_CARD_QUERY)
report_card_disc_df = load_data(engine, REPORT_CARD_DISC_QUERY)
rssi_cohorts_df = load_data(engine, RSSI_COHORTS_QUERY)


# #### Merge dataframes & aggregate columns

# In[295]:


report_card_df = pd.merge(report_card_df, report_card_disc_df, on='rcdts', how='inner')

aggregation_map_disc = {
    'num_disc_incdts_violent': [
        'num_disc_incdts_viol_harm', 'num_disc_incdts_viol_noharm'
    ],
    'num_disc_incdts_other': [
        'num_disc_incdts_firearm', 'num_disc_incdts_oth_wpn'
    ]
}
report_card_df = aggregate_columns(report_card_df, aggregation_map_disc)


# #### Calculate and describe ratios

# In[ ]:


report_card_df = calculate_exclusion_ratio(
    report_card_df, ['num_stdnts_with_discipline_incidents'],
    'num_stdnt_enrlmnt', 'disciplinary_action_ratio'
)

report_card_df = calculate_violent_inc_ratio(
    report_card_df, ['num_disc_incdts_violent', 'num_disc_incdts_other'],
    'num_stdnts_with_discipline_incidents', 'violent_incidents_ratio'
)

report_card_df = calculate_disciplinary_to_violence_ratio(
    report_card_df, ['num_stdnts_with_discipline_incidents'],
    ['num_disc_incdts_violent', 'num_disc_incdts_other'],
    'disciplinary_to_violence_ratio'
)

report_card_df['disciplinary_action_ratio'] = pd.to_numeric(
    report_card_df['disciplinary_action_ratio'], errors='coerce'
)
report_card_df['violent_incidents_ratio'] = pd.to_numeric(
    report_card_df['violent_incidents_ratio'], errors='coerce'
)
report_card_df['disciplinary_to_violence_ratio'] = pd.to_numeric(
    report_card_df['disciplinary_to_violence_ratio'], errors='coerce'
)
report_card_df[
    ['disciplinary_action_ratio', 'violent_incidents_ratio', 'disciplinary_to_violence_ratio']
].describe().round(2)


# The data reveals that, on average, the **disciplinary action ratio** across the 18,781 schools is relatively low, with a mean of 0.05 and a standard deviation of 0.07. This suggests that most schools have a small proportion of students involved in disciplinary actions. The distribution of this ratio is skewed, with the 25th percentile at 0.01 and the 75th percentile at 0.07, indicating that the majority of schools report minimal disciplinary incidents. However, the maximum value of 0.77 suggests that there are a small number of schools experiencing significantly higher disciplinary actions, creating a considerable spread in the data.
# 
# In contrast, the **violent incidents ratio** has a much higher mean of 0.59 and a similar standard deviation of 0.58, indicating greater variability in the frequency of violent incidents. The data is more spread out, with the 25th percentile at 0.12 and the 75th percentile at 0.93, implying that many schools report a relatively small number of violent incidents, but a significant proportion of schools experience much higher rates. The maximum value of 6.00, a considerable outlier, suggests that there are a few schools with exceptionally high rates of violent incidents. This highlights that while violent incidents may be rare in many schools, a small number of schools have disproportionately high occurrences.

# Filter RSSI schools

# In[ ]:


rssi_rcdts = rssi_cohorts_df['rcdts']
rssi_schools_df = report_card_df[report_card_df['rcdts'].isin(rssi_rcdts)]

rssi_schools_df = calculate_exclusion_ratio(
    rssi_schools_df, ['num_stdnts_with_discipline_incidents'],
    'num_stdnt_enrlmnt', 'disciplinary_action_ratio'
)

rssi_schools_df = calculate_violent_inc_ratio(
    rssi_schools_df, ['num_disc_incdts_violent', 'num_disc_incdts_other'],
    'num_stdnts_with_discipline_incidents', 'violent_incidents_ratio'
)

rssi_schools_df = calculate_disciplinary_to_violence_ratio(
    rssi_schools_df, ['num_stdnts_with_discipline_incidents'],
    ['num_disc_incdts_violent', 'num_disc_incdts_other'],
    'disciplinary_to_violence_ratio'
)

rssi_schools_df['disciplinary_action_ratio'] = pd.to_numeric(
    rssi_schools_df['disciplinary_action_ratio'], errors='coerce'
)
rssi_schools_df['violent_incidents_ratio'] = pd.to_numeric(
    rssi_schools_df['violent_incidents_ratio'], errors='coerce'
)
rssi_schools_df['disciplinary_to_violence_ratio'] = pd.to_numeric(
    rssi_schools_df['disciplinary_to_violence_ratio'], errors='coerce'
)
rssi_schools_df[
    ['disciplinary_action_ratio', 'violent_incidents_ratio', 'disciplinary_to_violence_ratio']
].describe().round(2)


# When analyzing the **RSSI schools** subset, we observe some similarities and differences compared to the overall school population. The **disciplinary action ratio** for RSSI schools has a mean of 0.06 and a standard deviation of 0.07, which is slightly higher than the mean for all schools (0.05). The spread of this ratio in RSSI schools is comparable to all schools, with the 25th percentile at 0.01 and the 75th percentile at 0.08, showing that most RSSI schools also report low disciplinary incidents. However, the maximum value for RSSI schools is 0.64, which is lower than the maximum of 0.77 observed for all schools, indicating fewer extreme cases in this subset.
# 
# For **violent incidents**, the mean for RSSI schools is 0.55, slightly lower than the 0.59 observed for all schools, with a standard deviation of 0.56. The spread of this ratio in RSSI schools is also noticeable: the 25th percentile is at 0.08, and the 75th percentile is at 0.86, which indicates a higher concentration of schools with violent incident ratios closer to the higher end of the distribution, compared to all schools. The maximum value of 4.65, while lower than the all-schools maximum of 6.00, still represents a significant outlier, showing that while most RSSI schools report low rates of violent incidents, a subset has notably higher rates. The distribution of violent incidents in RSSI schools appears slightly more skewed than that for all schools.

# #### Quantile calculations

# In[ ]:


all_schools_disciplinary_quantiles = (
    report_card_df['disciplinary_action_ratio'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
                                               .round(2)
)
all_schools_violent_quantiles = (
    report_card_df['violent_incidents_ratio'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
                                             .round(2)
)
all_schools_disciplinary_to_violence_quantiles = (
    report_card_df['disciplinary_to_violence_ratio'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
                                                    .round(2)
)

rssi_disciplinary_quantiles = (
    rssi_schools_df['disciplinary_action_ratio'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
                                                .round(2)
)
rssi_violent_quantiles = (
    rssi_schools_df['violent_incidents_ratio'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
                                              .round(2)
)
rssi_disciplinary_to_violence_quantiles = (
    rssi_schools_df['disciplinary_to_violence_ratio'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
                                                     .round(2)
)


# #### Create quantiles table

# In[ ]:


columns = pd.MultiIndex.from_product(
    [[
        'Disciplinary Action Ratio', 'Violent Incidents Ratio',
        'Discplinary-to-Violence Ratio'
    ],
    ['All Schools', 'RSSI Schools'],],
    names=['Metric', 'School Type']
)

quantiles_table = pd.DataFrame(
    data=np.array([
        all_schools_disciplinary_quantiles.values,
        rssi_disciplinary_quantiles.values,
        all_schools_violent_quantiles.values,
        rssi_violent_quantiles.values,
        all_schools_disciplinary_to_violence_quantiles.values,
        rssi_disciplinary_to_violence_quantiles.values

    ]).T,
    columns=columns,
    index=all_schools_disciplinary_quantiles.index
)

quantiles_table.index.name = 'Quantile'

display(quantiles_table)


# For the **disciplinary action ratio**, both groups show low median values (0.018 for all schools and 0.025 for RSSI schools), indicating that the majority of schools have relatively low disciplinary actions. However, the 80th and 100th percentiles for all schools are slightly higher than those for RSSI schools, suggesting that a subset of all schools exhibits more extreme ratios of disciplinary actions compared to RSSI schools.
# 
# In terms of the **violent incidents ratio**, the data indicates greater variability. The median ratio for all schools (0.333) is higher than for RSSI schools (0.286), showing that violent incidents are somewhat more common in all schools. However, the maximum ratio for all schools (6.0) exceeds that of RSSI schools (4.65), suggesting that extreme cases of violence are more frequent in the broader school population. Both distributions are skewed, as evidenced by higher values in the upper quantiles, but RSSI schools exhibit less extreme outliers. These findings may indicate differing behavioral trends or reporting practices between the two groups.

# #### Create histograms

# In[300]:


generate_histograms(report_card_df, rssi_schools_df)


# In[ ]:


def normalize_scores(df, column):
    max_ratio = df[column].max()
    min_ratio = df[column].min()

    df[column + '_score_normal'] = 100 * (1 - ((df[column] - min_ratio) / (max_ratio - min_ratio)))

    bins = [float('-inf'), 0, 20, 40, 60, 80, float('inf')]
    labels = [0, 20, 40, 60, 80, 100]
    df[column + '_score_normal'] = pd.cut(
        df[column + '_score_normal'], bins=bins, labels=labels, right=True, include_lowest=True
    )

    return df

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

def custom_score(df, column, custom_bins_scores):
    """
    Calculate custom scores based on bins and scores for a specific column.

    Args:
        df (pd.DataFrame): The input dataframe.
        column (str): The column to calculate scores for.
        custom_bins_scores (dict): A dictionary with 'bins' and 'scores'.

    Returns:
        pd.DataFrame: The dataframe with an additional column for custom scores.
    """
    bins = custom_bins_scores[column]['bins']
    scores = custom_bins_scores[column]['scores']

    df[column + '_score_custom'] = pd.cut(df[column], bins=bins, labels=scores, right=True)
    df = df.dropna(subset=[column + '_score_custom'])
    df[column + '_score_custom'] = df[column + '_score_custom'].astype(float)

    return df

def quantile_scores(df, column):
    """
    Determine the quantile scores for a given column.

    Args:
        df (pd.DataFrame): The input dataframe.
        column (str): The column name for which to calculate quantiles.

    Returns:
        pd.DataFrame: The dataframe with an additional column for quantile scores.
    """
    quantiles = df[column].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])

    def assign_quantile(x):
        if x <= quantiles[0.0]:
            return 100
        if x <= quantiles[0.2]:
            return 80
        if x <= quantiles[0.4]:
            return 60
        if x <= quantiles[0.6]:
            return 40
        if x <= quantiles[0.8]:
            return 20
        if x <= quantiles[1.0]:
            return 0

    df[column + '_score_quantile'] = df[column].apply(assign_quantile)

    return df

columns_to_score = [
    'disciplinary_action_ratio',
    'violent_incidents_ratio',
    'disciplinary_to_violence_ratio'
]

custom_bins_scores = {
    'disciplinary_action_ratio': {
        'bins': [0, 0.005, 0.02, 0.1, 0.3, 0.5, 1],
        'scores': [100, 80, 60, 40, 20, 0],
    },
    'violent_incidents_ratio': {
        'bins': [0, 0.25, 0.5, 1.0, 1.5, 2.0, float('inf')],
        'scores': [100, 80, 60, 40, 20, 0],
    },
    'disciplinary_to_violence_ratio': {
        'bins': [0.2, 0.6, 1.0, 3.0, 5.0, 7.0, float('inf')],
        'scores': [100, 80, 60, 40, 20, 0],
    },
}

for column in columns_to_score:
    report_card_df = normalize_scores(report_card_df, column)
    report_card_df = custom_score(report_card_df, column, custom_bins_scores)
    # report_card_df = granular_rescale(report_card_df, column)
    report_card_df = quantile_scores(report_card_df, column)


# In[ ]:


def score_distribution_table(df, column):
    """
    Creates a table displaying the count and ratio of values for each exact score 
    (0, 20, 40, 60, 80, 100) in different scoring methods.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        column (str): The column name to generate distribution tables for.
    
    Returns:
        pd.DataFrame: A formatted table with multi-index headers for method names 
        and subheaders for count and ratio.
    """

    score_values = [0, 20, 40, 60, 80, 100]

    scoring_methods = {
        "Quantile": f"{column}_score_quantile",
        "Normal": f"{column}_score_normal",
        "Custom": f"{column}_score_custom" #,
        # "Granular": f"{column}_score_granular"
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
            row[(method, "Min Ratio")] = f"{min_ratio * 100:.1f}%" if not pd.isna(min_ratio) else "N/A"
            row[(method, "Max Ratio")] = f"{max_ratio * 100:.1f}%" if not pd.isna(max_ratio) else "N/A"
            row[(method, "Median Ratio")] = f"{median_ratio * 100:.1f}%" if not pd.isna(median_ratio) else "N/A"

        results.append(row)

    df_result = pd.DataFrame(results)
    df_result.columns = pd.MultiIndex.from_tuples(df_result.columns)
    return df_result

columns_to_print = [
    'disciplinary_action_ratio',
    'violent_incidents_ratio' #,
    # 'disciplinary_to_violence_ratio'
]

# Generate tables for each column
for column in columns_to_print:
    print(f"\nDistribution Table for {column.replace('_', ' ').title()}:")
    display(score_distribution_table(report_card_df, column))


# ### Granular Rescaling for Skewed Distribution of Ratios
# 
# The goal of this approach is to fine-tune the scoring of two heavily skewed ratio metrics (`disciplinary_action_ratio` and `violent_incidents_ratio`) to better represent the distribution of these variables. The ratios are often characterized by a heavy concentration of values in the lower ranges, which makes traditional linear scoring inadequate in capturing subtle differences in the lower and higher ends of the distribution. 
# 
# The updated approach utilizes several techniques to improve the granularity and fairness of the scores:
# 
# 1. **Log Transformation (`log1p`)**:
#    - The `log1p` function is applied to both ratios to compress the upper tail and make the distribution more symmetric. This helps reduce the skewness, especially for outliers in the higher range.
# 
# 2. **Custom Scoring**:
#    - For comparison, a custom binning approach divides the data into predefined ranges (e.g., `0.01`, `0.05`, etc.) and assigns scores based on these ranges. This method provides a categorical distinction between different levels of the ratio.
#    
# 3. **Granular Rescaling**:
#    - **Low range (below 25th percentile)**: A steep linear scaling is used to better differentiate between values in this region, which typically contain more observations.
#    - **Mid range (between 25th and 75th percentiles)**: A linear rescaling approach is employed here to provide a moderate adjustment to scores.
#    - **High range (above 75th percentile)**: A more gradual scaling (based on a logarithmic adjustment) is used to prevent the scores from expanding too rapidly, as outliers in this region are less common.
# 
# ### Current Limitations:
# While this approach improves the representation of values across different segments of the distribution, there are still a few limitations:
# - **Outlier Handling**: Even though the logarithmic and granular transformations help, extreme outliers may still affect the scoring, especially in the higher ranges. The scaling doesn't fully normalize these extreme values.
# - **Interpretability**: For users unfamiliar with the transformation or the custom scoring bins, it might be difficult to interpret what the raw scores mean in the context of the original data.
# - **Dependence on Quantiles**: The rescaling depends on the quantiles (25th and 75th percentiles), which might not always be the most effective thresholds, especially if the data distribution is highly uneven or has unusual peaks.
# - **Performance**: For very large datasets, applying these transformations across many columns and ratios might require optimization to ensure quick computations. 

# The scoring results for both `disciplinary_action_ratio` and `violent_incidents_ratio` demonstrate the differences between the two scoring approaches: **standard scoring** and **granular rescaling**.
# 
# #### **Disciplinary Action Ratio Scores**:
# - **Standard Scoring**: The mean score is 89.53, indicating that most schools fall on the higher end of the scale. The distribution is fairly concentrated around the upper range, with the 25th percentile at 85.73 and the 75th percentile at 97.29. The relatively low standard deviation (11.19) suggests that the scores are fairly consistent across the dataset.
# - **Granular Rescaling**: The granular rescaling method, however, yields a mean of 79.14 and a higher standard deviation of 30.33. This approach creates more variation in the scores, with the 25th percentile at 60.72 and the 75th percentile reaching 100. This indicates that the granular scaling accounts for more variability, especially by increasing the scores of higher values more significantly.
# 
# #### **Violent Incidents Ratio Scores**:
# - **Standard Scoring**: The violent incidents ratio shows a slightly lower mean of 79.03, with a similar distribution to the disciplinary action ratio. The standard deviation (16.54) is larger than the disciplinary action ratio’s, reflecting a greater spread of values. The 25th percentile is 66.12, and the 75th percentile is 93.95, suggesting a moderate concentration around the upper-middle range.
# - **Granular Rescaling**: The granular score for violent incidents ratio shows a mean of 74.86 and a larger standard deviation of 35.83. Like the disciplinary action ratio, this approach introduces more spread, with a 25th percentile of 46.87 and a 75th percentile reaching 100. This demonstrates how the granular rescaling method adjusts for a wider range of values, particularly for the lower and mid-range values.
# 
