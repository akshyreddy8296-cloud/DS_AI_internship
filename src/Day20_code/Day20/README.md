# Customer Analytics - Exploratory Data Analysis (EDA)

## Project Overview

This project performs a comprehensive Exploratory Data Analysis (EDA) on customer demographics and spending behavior data. The analysis aims to uncover patterns, trends, and insights about customer characteristics that influence spending scores.

## Dataset

**File**: `customer_analytics.csv`

### Dataset Columns
- **CustomerID**: Unique identifier for each customer
- **Age**: Customer's age in years
- **Gender**: Customer's gender (Male/Female)
- **City**: City where the customer resides
- **Education**: Highest level of education achieved (Bachelors, Masters, PhD, etc.)
- **MaritalStatus**: Customer's marital status (Single, Married, Divorced, etc.)
- **AnnualIncome**: Customer's annual income in currency units
- **SpendingScore**: A numerical score representing customer spending behavior (0-100 scale)

### Dataset Dimensions
- **Original Records**: 255 customers
- **Final Records**: 250 customers (after cleaning)
- **Total Features**: 8 columns

## Project Structure

```
├── customer_analytics.csv                 # Raw dataset
├── MiniProject1_EDA1.ipynb               # Main EDA notebook
├── requirements.txt                       # Python dependencies
├── Dataset_Information.md                 # Dataset documentation
├── PROJECT_DOCUMENTATION.md               # Detailed project documentation
├── Insights_Report.pdf                    # Generated insights report
└── README.md                              # This file
```

## Project Phases

### Phase 1: Data Inspection & Setup
- Load and explore the dataset
- Display basic structure and first few records
- Generate summary statistics
- Identify data types and potential issues

### Phase 2: Data Cleaning & Preprocessing
- **Missing Values**: Imputation using mode (categorical) and median (numerical)
- **Duplicates**: Detection and removal of duplicate records
- **Result**: Clean dataset ready for analysis

### Phase 3: Exploratory Data Analysis
- Univariate analysis (individual variable distributions)
- Bivariate analysis (relationships between variables)
- Statistical insights and patterns
- Visualization of key findings

## Key Technologies

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **jupyter**: Interactive notebook environment

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Analysis

1. **Using Jupyter Notebook**:
   ```bash
   jupyter notebook MiniProject1_EDA1.ipynb
   ```

2. **Using VS Code**:
   - Open `MiniProject1_EDA1.ipynb` in VS Code
   - Click "Run All" to execute all cells

## Key Findings

The analysis reveals several important patterns:
- **Age Distribution**: Customers span a wide age range with specific concentration points
- **Income vs Spending**: Relationship patterns between annual income and spending behavior
- **Demographic Patterns**: Variations in spending scores across different education levels and cities
- **Gender Differences**: Comparative spending behavior between male and female customers

For detailed insights, refer to `Insights_Report.pdf`

## Data Quality

- **Missing Values Handled**: 12 missing values imputed
- **Duplicates Removed**: 5 duplicate records removed
- **Data Validation**: Complete data integrity check performed

## File Descriptions

| File | Purpose |
|------|---------|
| `customer_analytics.csv` | Raw customer data |
| `MiniProject1_EDA1.ipynb` | Main analysis notebook with code and visualizations |
| `Dataset_Information.md` | Detailed dataset documentation |
| `PROJECT_DOCUMENTATION.md` | Technical project documentation |
| `Insights_Report.pdf` | Executive summary with key findings |
| `requirements.txt` | Python package dependencies |
| `README.md` | Project overview and instructions |

## Author & Date

- **Project Type**: Data Science - Exploratory Data Analysis
- **Date**: February 2026
- **Status**: Completed

## Next Steps / Future Enhancements

1. Perform advanced statistical modeling
2. Customer segmentation using clustering techniques
3. Predictive modeling for spending behavior
4. Build interactive dashboards
5. Time-series analysis if temporal data is available

## Notes

- All visualizations use a consistent modern color palette for better readability
- Missing value imputation prioritizes data preservation while maintaining integrity
- Statistical methods chosen are appropriate for small to medium datasets

---

For questions or further analysis, refer to the detailed documentation files included in this project.
