import os
import pandas as pd
import yaml
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# Load Database Configuration
def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        return yaml.safe_load(file)

# Create Database Connection
def create_db_engine(db_config):
    db_url = f"postgresql+psycopg2://{db_config['uid']}:{db_config['pwd']}@{db_config['server']}:{db_config['port']}/{db_config['database']}"
    return create_engine(db_url)

# Execute SQL Query and Load DataFrame
def load_data(engine, query):
    with engine.connect() as connection:
        return pd.read_sql(query, connection)

# Aggregate Columns in DataFrame
def aggregate_columns(df, aggregation_map):
    for new_col, cols_to_sum in aggregation_map.items():
        df[cols_to_sum] = df[cols_to_sum].apply(pd.to_numeric, errors='coerce').fillna(0)
        df[new_col] = df[cols_to_sum].sum(axis=1)
    return df

# Calculate Exclusion Discipline Action Ratio
def calculate_exclusion_ratio(df, suspension_cols, enrollment_col, new_col_name, missing_data_value=None):
    # Convert columns to numeric, handle missing values
    df[suspension_cols] = df[suspension_cols].apply(pd.to_numeric, errors='coerce')
    df[enrollment_col] = pd.to_numeric(df[enrollment_col], errors='coerce')
    
    # Initialize the new column with missing data value
    df[new_col_name] = missing_data_value
    
    # Apply the logic to calculate exclusion ratio
    for idx, row in df.iterrows():
        if pd.isnull(row[enrollment_col]) or row[enrollment_col] < 0:
            df.at[idx, new_col_name] = missing_data_value
        else:
            # If suspension columns are all null, use the missing data value
            if row[suspension_cols].isnull().all():
                df.at[idx, new_col_name] = missing_data_value
            else:
                # Sum suspension values and calculate the ratio
                disciplinary_action_count = row[suspension_cols].sum()
                df.at[idx, new_col_name] = disciplinary_action_count / row[enrollment_col] 

    return df


# Calculate Violent Incident Ratio
def calculate_violent_inc_ratio(df, numerator_cols, enrollment_col, new_col_name, missing_data_value=None):
    # Convert columns to numeric, handle missing values
    df[numerator_cols] = df[numerator_cols].apply(pd.to_numeric, errors='coerce')
    df[enrollment_col] = pd.to_numeric(df[enrollment_col], errors='coerce')
    
    # Initialize the new column with missing data value
    df[new_col_name] = missing_data_value
    
    # Apply the logic to calculate violent incident ratio
    for idx, row in df.iterrows():
        if pd.isnull(row[enrollment_col]) or row[enrollment_col] < 0:
            df.at[idx, new_col_name] = missing_data_value
        else:
            # If violent incident columns are all null, use the missing data value
            if row[numerator_cols].isnull().all():
                df.at[idx, new_col_name] = missing_data_value
            else:
                # Sum violent incident values and calculate the ratio
                violence_incidents_count = row[numerator_cols].sum()
                df.at[idx, new_col_name] = violence_incidents_count / row[enrollment_col]

    return df

def count_schools_by_quantile(df, column, quantiles, column_label="quantile"):
    """
    Calculates the number of schools in each quantile for a given column using precomputed quantiles.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        column (str): The column for which quantile counts are calculated.
        quantiles (Series): Precomputed quantile values.
        column_label (str): The name of the new column to store quantile labels.

    Returns:
        DataFrame: A DataFrame with quantile ranges and counts.
    """
    # Define quantile labels
    quantile_labels = [f"Q{i+1}" for i in range(len(quantiles) - 1)]

    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column])
    
    # Categorize the data based on the quantile bins
    df[column_label] = pd.cut(df[column], bins=quantiles, labels=quantile_labels, include_lowest=True)
    
    # Calculate counts
    quantile_counts = df.groupby(column_label, observed=False).size().reset_index(name="school_count")
    
    # Create a summary DataFrame with quantile ranges and counts
    quantile_summary = pd.DataFrame({
        "quantile": quantile_labels,
        "quantile_range": [f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]" for i in range(len(quantiles) - 1)],
        "school_count": quantile_counts["school_count"]
    })
    
    return quantile_summary

# Generate Histograms
def generate_histograms(report_card_df, report_card_disc_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Histograms for Disciplinary Action and Violent Incidents Ratios', fontsize=16)

    # Old Disciplinary Action Ratio
    axes[0, 0].hist(report_card_df['disciplinary_action_ratio'].dropna(), bins=10, color='blue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title("Old Disciplinary Action Ratio")
    axes[0, 0].set_xlabel("Ratio")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # New Disciplinary Action Ratio
    axes[0, 1].hist(report_card_disc_df['disciplinary_action_ratio'].dropna(), bins=10, color='green', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title("New Disciplinary Action Ratio")
    axes[0, 1].set_xlabel("Ratio")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # Old Violent Incidents Ratio
    axes[1, 0].hist(report_card_df['violent_incidents_ratio'].dropna(), bins=10, color='red', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title("Old Violent Incidents Ratio")
    axes[1, 0].set_xlabel("Ratio")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # New Violent Incidents Ratio
    axes[1, 1].hist(report_card_disc_df['violent_incidents_ratio'].dropna(), bins=10, color='purple', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title("New Violent Incidents Ratio")
    axes[1, 1].set_xlabel("Ratio")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Main Script
def main():
    try:
        # Load Configurations
        config_path = "../utility/config.yml"
        config = load_config(config_path)
        db_config = config['rssi-data-warehouse']['dataconnection']

        # Create Database Engine
        engine = create_db_engine(db_config)

        # SQL Queries
        report_card_query = """
        SELECT rcdts, isbe_type, school_name, district, num_crdc_total_enrlmnt_2018, 
               num_crdc_in_school_suspensions_2018, num_crdc_out_of_school_suspensions_2018, 
               num_crdc_expulsions_2018, num_crdc_school_related_arrests_2018, 
               num_crdc_referral_to_law_enforcement_2018, num_crdc_incidents_of_violence_2018
        FROM report_card
        """
        report_card_disc_query = """
        SELECT rcdts, isbe_type, school_name, district, num_stdnts_with_discipline_incidents,
               "num_disc_incdtsViol_hame", num_disc_incdtsviol_noharm, 
               num_disc_incdtsfirearm, num_disc_incdtsoth_wpn
        FROM report_card_disc
        """

        # Load Data
        report_card_df = load_data(engine, report_card_query)
        report_card_disc_df = load_data(engine, report_card_disc_query)

        # Aggregation Map
        aggregation_map_disc = {
            'num_disc_incdts_violent': ['num_disc_incdtsViol_hame', 'num_disc_incdtsviol_noharm'],
            'num_disc_incdts_weapon': ['num_disc_incdtsfirearm', 'num_disc_incdtsoth_wpn']
        }

        # Aggregate Columns
        report_card_disc_df = aggregate_columns(report_card_disc_df, aggregation_map_disc)

        # Load Enrollment Data from Excel
        input_dir = os.path.join("data", "input")
        excel_path = os.path.join(input_dir, "report_card.xlsx")
        enrollment_data = pd.read_excel(excel_path, sheet_name="General", usecols=["RCDTS", "# Student Enrollment"])

        # Map Enrollment Data
        report_card_disc_df = report_card_disc_df.merge(
            enrollment_data.rename(columns={"RCDTS": "rcdts", "# Student Enrollment": "student_enrollment"}),
            on="rcdts",
            how="left"
        )

        # Calculate Exclusion Ratios
        missing_tally_element_data_value = np.nan
        report_card_df = calculate_exclusion_ratio(
            report_card_df, 
            ['num_crdc_in_school_suspensions_2018', 'num_crdc_out_of_school_suspensions_2018', 'num_crdc_expulsions_2018'], 
            'num_crdc_total_enrlmnt_2018', 
            "disciplinary_action_ratio",
            missing_data_value=missing_tally_element_data_value
        )

        report_card_disc_df = calculate_exclusion_ratio(
            report_card_disc_df, 
            ['num_stdnts_with_discipline_incidents'], 
            'student_enrollment', 
            "disciplinary_action_ratio",
            missing_data_value=missing_tally_element_data_value
        )

        # Calculate Violent Incident Ratios
        report_card_df = calculate_violent_inc_ratio(
            report_card_df, 
            ['num_crdc_school_related_arrests_2018', 'num_crdc_referral_to_law_enforcement_2018', 'num_crdc_incidents_of_violence_2018'], 
            'num_crdc_total_enrlmnt_2018', 
            'violent_incidents_ratio',
            missing_data_value=missing_tally_element_data_value
        )

        report_card_disc_df = calculate_violent_inc_ratio(
            report_card_disc_df, 
            ['num_disc_incdts_violent', 'num_disc_incdts_weapon'], 
            'student_enrollment', 
            'violent_incidents_ratio',
            missing_data_value=missing_tally_element_data_value
        )

        # Calculate Quantiles for disciplinary_action_ratio and violent_incidents_ratio
        disciplinary_quantiles = report_card_df['disciplinary_action_ratio'].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        violent_quantiles = report_card_df['violent_incidents_ratio'].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Optionally, display for the report_card_disc_df if you need:
        disciplinary_quantiles_disc = report_card_disc_df['disciplinary_action_ratio'].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        violent_quantiles_disc = report_card_disc_df['violent_incidents_ratio'].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        print("\nRunning `.quantile` on NEW and OLD data...")

        try:

            # Quantile Counts for Disciplinary Action Ratio
            disciplinary_action_quantiles_old = count_schools_by_quantile(
                report_card_df, "disciplinary_action_ratio", disciplinary_quantiles
            )

            disciplinary_action_quantiles_new = count_schools_by_quantile(
                report_card_disc_df, "disciplinary_action_ratio", disciplinary_quantiles_disc
            )

            # Quantile Counts for Violent Incidents Ratio
            violent_incidents_quantiles_old = count_schools_by_quantile(
                report_card_df, "violent_incidents_ratio", violent_quantiles
            )
            violent_incidents_quantiles_new = count_schools_by_quantile(
                report_card_disc_df, "violent_incidents_ratio", violent_quantiles_disc
            )

            # Print Results
            print("Disciplinary Action Ratio (Old):")
            print(disciplinary_action_quantiles_old)

            print("\nDisciplinary Action Ratio (New):")
            print(disciplinary_action_quantiles_new)

            print("\nViolent Incidents Ratio (Old):")
            print(violent_incidents_quantiles_old)

            print("\nViolent Incidents Ratio (New):")
            print(violent_incidents_quantiles_new)

        except Exception as e:
            print(f"An error occurred: {e}")

        generate_histograms(report_card_df, report_card_disc_df)
            
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()