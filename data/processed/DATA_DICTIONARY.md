# Bank Churners Data Dictionary

| Column Name | Data Type | Feature Group | Modeling | Nulls | Has 'Unknown' | Description |
|---|---|---|:---:|:---:|:---:|---|
| `CLIENTNUM` | int64 | Identifier | No | 0 | No | Unique customer account identifier |
| `Attrition_Flag` | object | Target | No | 0 | No | Target variable: Existing Customer (0) or Attrited Customer (1) |
| `Customer_Age` | int64 | Demographics | Yes | 0 | No | Customer age in years |
| `Gender` | object | Demographics | Yes | 0 | No | Customer gender (M=Male, F=Female) |
| `Dependent_count` | int64 | Demographics | Yes | 0 | No | Number of financial dependents |
| `Education_Level` | object | Demographics | Yes | 0 | Yes | Educational qualification of the account holder |
| `Marital_Status` | object | Demographics | Yes | 0 | Yes | Marital status (Married, Single, Divorced, Unknown) |
| `Income_Category` | object | Demographics | Yes | 0 | Yes | Annual income bracket of the account holder |
| `Card_Category` | object | Product / Relationship | Yes | 0 | No | Type of credit card held (Blue, Silver, Gold, Platinum) |
| `Months_on_book` | int64 | Product / Relationship | Yes | 0 | No | Tenure of customer relationship with the bank in months |
| `Total_Relationship_Count` | int64 | Product / Relationship | Yes | 0 | No | Total number of banking products held by the customer |
| `Months_Inactive_12_mon` | int64 | Engagement | Yes | 0 | No | Number of inactive months in the last 12 months |
| `Contacts_Count_12_mon` | int64 | Engagement | Yes | 0 | No | Number of contacts between customer and bank in last 12 months |
| `Credit_Limit` | float64 | Financial / Credit | Yes | 0 | No | Credit limit on the credit card account |
| `Total_Revolving_Bal` | int64 | Financial / Credit | Yes | 0 | No | Total revolving balance on the credit card |
| `Avg_Open_To_Buy` | float64 | Financial / Credit | Yes | 0 | No | Average open to buy credit line (Credit_Limit - Total_Revolving_Bal) |
| `Avg_Utilization_Ratio` | float64 | Financial / Credit | Yes | 0 | No | Average credit card utilization ratio |
| `Total_Amt_Chng_Q4_Q1` | float64 | Transaction Behavior | Yes | 0 | No | Ratio of total transaction amount in Q4 compared to Q1 |
| `Total_Trans_Amt` | int64 | Transaction Behavior | Yes | 0 | No | Total transaction amount in the last 12 months |
| `Total_Trans_Ct` | int64 | Transaction Behavior | Yes | 0 | No | Total transaction count in the last 12 months |
| `Total_Ct_Chng_Q4_Q1` | float64 | Transaction Behavior | Yes | 0 | No | Ratio of total transaction count in Q4 compared to Q1 |
