## RSSI Discipline Analysis

This project focuses on analyzing disciplinary actions and incidents within the RSSI dataset. The `discipline_analysis.ipynb` notebook provides a comprehensive overview of the current analysis, including data cleaning, processing, and insights drawn from the data.

### Getting Started

Follow these steps to set up your environment and start analyzing the data:

### 1. **Clone the Repository:**

Begin by cloning the repository to your local machine:
```bash
git clone https://github.com/parliamentdata/rssi_discipline_analysis.git
```

### 2. **Set Up the Virtual Environment:**

Activate the virtual environment to ensure you have a clean, isolated environment for this project:
```bash
# Activate the virtual environment
# For macOS/Linux
source discipline_venv/bin/activate

# For Windows
.\discipline_venv\Scripts\activate
```

### 3. **Install Dependencies:**

Install the necessary Python libraries by using the `requirements.txt` file included in the repository. This will install all the required packages for running the notebook:
```bash
pip install -r requirements.txt
```

### 4. **Install the Jupyter Kernel:**

To use the virtual environment in Jupyter, you'll need to install the kernel for it. Run the following command:
```bash
python -m ipykernel install --user --name=discipline_venv --display-name "Python (discipline_venv)"
```

This will allow you to select the virtual environment kernel from within Jupyter.

### 5. **Configure the PostgreSQL Database Connection:**

- **Move the Postgres Configuration File:** Locate and rename the database configuration file to `config.yml` if necessary.
- **Move the Configuration File to the Project Root:** Place the `config.yml` file in the root directory of your project.
- **Check Database Credentials:** Ensure the `config.yml` file has the correct credentials for the RSSI data warehouse. If needed, find the credentials in Parliament's OnePassword vault.

### 6. **Open the Jupyter Notebook:**

Launch Jupyter Notebook and open the `discipline_analysis.ipynb` file:
```bash
code discipline_analysis.ipynb
# OR
jupyter notebook discipline_analysis.ipynb
```

### 7. **Run the Code:**

Execute the code cells sequentially to follow the analysis and generate the insights. Each section of the notebook provides explanations and comments about the steps being taken.

### Additional Notes:

- **Updating the Environment:** If you need to install new packages in the virtual environment, activate the `discipline_venv` and use `pip install <package-name>`.
  
- **Deactivating the Virtual Environment:** When you are done, deactivate the virtual environment by running:
  ```bash
  deactivate
  ```