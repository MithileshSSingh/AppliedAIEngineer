# P1: Weekly Sales & Margin Pipeline

## Objective
Build an automated data pipeline that ingests raw sales data, cleans it, computes key business metrics (revenue, margin, growth), and produces a formatted report with visualizations.

## Dataset
- **Source:** Kaggle "Superstore Sales" dataset
- **Download:** https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
- **Alternative:** Use the synthetic data generator in `src/generate_data.py`
- Place the CSV in the `data/` folder

## Skills Practiced
- Python OOP (pipeline as a class)
- Pandas (cleaning, groupby, merge, pivot)
- Matplotlib/Seaborn (visualizations)
- Method chaining
- Code organization (src/ modules)

## Deliverables
1. `notebook.ipynb` — Main analysis notebook
2. `src/pipeline.py` — `SalesPipeline` class encapsulating the full ETL
3. `src/generate_data.py` — Synthetic data generator (if not using Kaggle)
4. `outputs/` — Generated charts and summary report

## Suggested Approach

### Week 1 (Days 4-5): Data loading & cleaning
- Load the dataset, explore with `.info()`, `.describe()`, `.isnull().sum()`
- Clean: handle missing values, fix dtypes (dates, numerics), remove duplicates
- Create a `SalesPipeline` class with `load()`, `clean()` methods

### Week 2 (Days 4-5): Analysis & reporting
- Add `transform()`: compute revenue, margin, month column
- Add `analyze()`: groupby summaries (by product, region, month)
- Add `report()`: generate charts + print key metrics
- Create 4-5 visualizations:
  - Monthly revenue trend (line chart)
  - Revenue by category (bar chart)
  - Margin by region (grouped bar)
  - Top 10 products by revenue (horizontal bar)
  - Revenue heatmap (month × category)

## How to Run
```bash
cd module-1-foundations/projects/p1-sales-pipeline
pip install -r ../../requirements.txt
jupyter lab notebook.ipynb
```
