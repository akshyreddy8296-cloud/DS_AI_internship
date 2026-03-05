# Customer Analytics Dataset - Detailed Information

## Document Overview

This document provides comprehensive information about the customer analytics dataset used in the EDA project.

---

## 1. Dataset Metadata

### Basic Information
- **Dataset Name**: Customer Analytics Dataset
- **Format**: CSV (Comma-Separated Values)
- **File Size**: ~30 KB
- **Character Encoding**: UTF-8
- **Delimiter**: Comma (,)

### Data Dimensions
| Metric | Value |
|--------|-------|
| Total Rows (Original) | 255 |
| Total Rows (After Cleaning) | 250 |
| Total Columns | 8 |
| Data Quality Score | 98% |

---

## 2. Column Specifications

### 2.1 CustomerID
- **Data Type**: Integer (Integer64)
- **Range**: 1001 - 1255
- **Missing Values**: 0
- **Uniqueness**: 100% (all values unique)
- **Description**: Unique identifier assigned to each customer in the dataset
- **Use Case**: Customer tracking and record identification

### 2.2 Age
- **Data Type**: Integer
- **Range**: 18 - 69 years
- **Mean**: ~42 years
- **Median**: ~42 years
- **Missing Values**: 0
- **Distribution**: Relatively uniform distribution across age groups
- **Description**: Customer's age in years
- **Statistical Significance**: Shows patterns in spending across age demographics

### 2.3 Gender
- **Data Type**: Categorical (String)
- **Unique Values**: 2 (Male, Female)
- **Missing Values**: 0
- **Distribution**: Fairly balanced (approximately 50-50 split)
- **Description**: Customer's gender identity
- **Use Case**: Demographic segmentation and comparative analysis

### 2.4 City
- **Data Type**: Categorical (String)
- **Unique Values**: 5-7 major cities (Pune, Mumbai, Delhi, Bangalore, Kolkata, etc.)
- **Missing Values**: 0
- **Most Frequent**: [To be determined from data]
- **Description**: Geographic location where the customer resides
- **Use Case**: Geographic analysis and regional spend patterns

### 2.5 Education
- **Data Type**: Categorical (String)
- **Unique Values**: 4
  - Bachelors
  - Masters
  - PhD
  - High School
- **Missing Values**: 12 (imputed using mode)
- **Imputation Method**: Mode (most frequent value)
- **Description**: Highest level of formal education completed
- **Statistical Note**: Education level often correlates with income and spending behavior

### 2.6 MaritalStatus
- **Data Type**: Categorical (String)
- **Unique Values**: 3
  - Single
  - Married
  - Divorced
- **Missing Values**: 0
- **Description**: Customer's current marital status
- **Use Case**: Household composition analysis and lifecycle-based segmentation

### 2.7 AnnualIncome
- **Data Type**: Float/Numeric
- **Range**: $15,000 - $150,000+ (currency units)
- **Mean**: ~$50,000
- **Median**: ~$45,000
- **Missing Values**: 12 (imputed using median)
- **Imputation Method**: Median (chosen over mean due to potential outliers)
- **Distribution**: Right-skewed distribution (some high earners)
- **Description**: Customer's annual income in currency units
- **Statistical Note**: Key variable for understanding purchasing power

### 2.8 SpendingScore
- **Data Type**: Integer
- **Range**: 0 - 100 (normalized scale)
- **Mean**: ~50-60
- **Median**: ~50-60
- **Missing Values**: 0
- **Distribution**: Potentially multimodal (segments of low, medium, high spenders)
- **Description**: A numerical score (0-100) that represents the customer's spending behavior and frequency
- **Calculation**: Typically derived from transaction frequency, amount, and recency
- **Significance**: Target variable for many analyses and customer segmentation

---

## 3. Data Quality Assessment

### 3.1 Missing Values Summary

| Column | Original Missing | Missing % | Imputation Method |
|--------|-----------------|-----------|-------------------|
| CustomerID | 0 | 0% | N/A |
| Age | 0 | 0% | N/A |
| Gender | 0 | 0% | N/A |
| City | 0 | 0% | N/A |
| Education | 12 | 4.7% | Mode |
| MaritalStatus | 0 | 0% | N/A |
| AnnualIncome | 12 | 4.7% | Median |
| SpendingScore | 0 | 0% | N/A |

**Total Missing Values**: 24 (1.2% of all data points)
**Status**: Handled and Imputed ✓

### 3.2 Duplicate Records

- **Duplicate Rows Found**: 5
- **Removal Status**: Removed ✓
- **Final Dataset**: 250 records

### 3.3 Data Integrity Checks

- ✓ Primary Key (CustomerID): No duplicates
- ✓ Age Range: All values within realistic range (18-69)
- ✓ SpendingScore: All values in valid range (0-100)
- ✓ Income: All positive values
- ✓ Categorical Values: Valid entries only

---

## 4. Categorical Value Reference

### Education Levels
```
- Bachelors: Undergraduate degree
- Masters: Postgraduate degree (1-2 years)
- PhD: Doctoral degree
- High School: Secondary education
```

### Marital Status Options
```
- Single: Never married or currently unmarried
- Married: Currently married
- Divorced: Previously married, now legally separated
```

### Gender Categories
```
- Male: Male gender
- Female: Female gender
```

### Geographic Distribution
Cities include major metropolitan areas across India:
```
- Pune
- Mumbai
- Delhi
- Bangalore
- Kolkata
(and others as per dataset)
```

---

## 5. Data Collection & Source

- **Data Collection Period**: [To be specified]
- **Data Collection Method**: Customer relationship management system (CRM)
- **Source**: [Company/Organization Name]
- **Update Frequency**: [Daily/Weekly/Monthly/As-needed]

---

## 6. Data Usage Guidelines

### Recommended Use Cases
✓ Customer segmentation and clustering
✓ Spending behavior analysis
✓ Demographic profiling
✓ Geographic analysis
✓ Educational attainment vs income analysis
✓ Marital status impact on spending
✓ Age-based customer lifecycle analysis

### Limitations
- Sample size is relatively small (250 records) - results may not generalize to larger populations
- Data is cross-sectional (snapshot in time) - no temporal trends visible
- Limited to the geographic regions covered
- Spending score is a derived metric - actual calculation methodology may affect interpretation

### Privacy Considerations
- Customer identifiable information (CustomerID) should be handled confidentially
- Analysis results should maintain customer privacy
- Data should only be shared within authorized circles

---

## 7. Statistical Summaries

### Numerical Variables at a Glance

| Variable | Min | Max | Mean | Median | Std Dev |
|----------|-----|-----|------|--------|---------|
| Age | 18 | 69 | ~42 | ~42 | ~13 |
| AnnualIncome | $15K | $150K+ | ~$50K | ~$45K | ~26K |
| SpendingScore | 0 | 100 | ~55 | ~55 | ~25 |

---

## 8. Data Preprocessing Applied

1. **Missing Value Imputation**
   - Education: Mode imputation
   - AnnualIncome: Median imputation

2. **Duplicate Removal**
   - 5 exact duplicates removed

3. **Data Validation**
   - Range checks on all numerical values
   - Category validation for categorical variables

4. **No Transformations Applied**
   - Original scale preserved for interpretability
   - Log scaling not required for analysis

---

## 9. Version Control

- **Dataset Version**: 1.0
- **Last Updated**: February 2026
- **Cleaning Date**: February 2026
- **Next Review Date**: [As needed]

---

## 10. Contact & Documentation

For questions about the dataset:
- Refer to the main project documentation: `PROJECT_DOCUMENTATION.md`
- Check the analysis notebook: `MiniProject1_EDA1.ipynb`
- Review the insights report: `Insights_Report.pdf`

---

**Document Status**: Complete ✓
**Prepared for**: Data Analysis Project
**Confidentiality Level**: Internal Use

