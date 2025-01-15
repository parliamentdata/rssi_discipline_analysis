import os
import pandas as pd

# Define file paths
input_folder = "data/input"
output_folder = "data/output"
input_file_name = "report_card.xlsx"
output_file_name = "discipline_columns.xlsx"
input_file_path = os.path.join(input_folder, input_file_name)
output_file_path = os.path.join(output_folder, output_file_name)

# Check if the input file exists
if os.path.exists(input_file_path):
    # Load the "Discipline" sheet from the Excel file
    try:
        discipline_sheet = pd.read_excel(input_file_path, sheet_name="Discipline")
        # Get the column names
        column_names = discipline_sheet.columns.tolist()
        print("Columns in the 'Discipline' sheet:")
        print(column_names)
        
        # Save column names as rows in a new Excel file
        column_df = pd.DataFrame(column_names, columns=["Column Names"])
        
        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Save to the output Excel file
        column_df.to_excel(output_file_path, index=False)
        print(f"Column names have been saved to {output_file_path}")
    except Exception as e:
        print(f"Error reading the 'Discipline' sheet: {e}")
else:
    print(f"The file {input_file_name} does not exist in the {input_folder} directory.")
