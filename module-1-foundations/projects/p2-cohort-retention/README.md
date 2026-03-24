# P2: Cohort Retention Analysis

## Objective
Build a cohort retention table and heatmap that shows how customer engagement changes over time. This is a core product analytics technique used by growth teams.

## Dataset
- **Option A:** UCI "Online Retail" dataset — https://archive.ics.uci.edu/dataset/352/online+retail
- **Option B:** Use the synthetic generator below

## What is Cohort Analysis?
A **cohort** is a group of users who share a common characteristic — usually the month they first purchased. Retention analysis tracks what % of each cohort continues to purchase in subsequent months.

## Skills Practiced
- Pandas: groupby, pivot_table, merge, datetime operations
- Seaborn heatmap visualization
- Business metric calculation (retention rate)
- Translating data into stakeholder insights

## Deliverables
1. `notebook.ipynb` — Full cohort analysis
2. A retention heatmap saved to `outputs/`
3. 3-5 bullet point summary of findings

## Suggested Steps

### Step 1: Load & Clean Data
- Load transactions dataset
- Remove returns (negative quantities) and missing CustomerIDs
- Ensure dates are parsed correctly

### Step 2: Assign Cohorts
- For each customer, find their **first purchase month** — this is their cohort
- Add a `cohort_month` column to the transactions

### Step 3: Calculate Cohort Index
- For each transaction, compute how many months after the cohort month it occurred
- `cohort_index = (transaction_month - cohort_month)` in months

### Step 4: Build Retention Table
- Pivot: rows = cohort_month, columns = cohort_index, values = count of unique customers
- Divide each row by its first column (month 0) to get retention %

### Step 5: Visualize
- Create a heatmap with `sns.heatmap(retention_table, annot=True, fmt='.0%')`
- Add a line chart showing average retention by cohort_index

### Step 6: Insights
- Which cohorts retained best/worst?
- At what month do most customers drop off?
- Any seasonal patterns?

## How to Run
```bash
jupyter lab notebook.ipynb
```
