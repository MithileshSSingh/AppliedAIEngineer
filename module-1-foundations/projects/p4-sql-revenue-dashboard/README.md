# P4: SQL Revenue Dashboard

## Objective
Build a Python + SQL analytics dashboard that computes key revenue metrics from a relational database and produces interactive visualizations. This is your capstone project for Module 1.

## Architecture
```
SQLite Database (data/store.db)
    ├── customers
    ├── products
    ├── orders
    └── order_items
         │
    SQL Queries (src/queries.py)
         │
    Analytics Engine (src/dashboard.py)
         │
    Output: HTML Report / Streamlit App
```

## Key Metrics to Compute
1. **MRR (Monthly Recurring Revenue)** — total revenue per month
2. **Revenue Growth** — month-over-month % change
3. **Customer LTV** — lifetime value per customer
4. **Churn Rate** — % of customers who didn't purchase in the last 3 months
5. **Revenue by Segment** — by region, category, tier
6. **Top Products** — by revenue and by units sold
7. **Basket Analysis** — average order value, items per order

## Skills Practiced
- SQL: complex queries with CTEs and window functions
- Python + SQL integration (sqlite3 + pandas)
- Data visualization (Plotly for interactivity, or Matplotlib/Seaborn)
- Code organization: separating queries, logic, and presentation
- Optional: Streamlit for a web-based dashboard

## Deliverables
1. `notebook.ipynb` — Main analysis with all metrics and charts
2. `src/queries.py` — All SQL queries as named functions
3. `src/dashboard.py` — Dashboard generation logic
4. `src/setup_db.py` — Script to create and populate the database
5. `outputs/dashboard.html` — Exported report (or Streamlit app)

## Suggested Approach

### Day 1: Database Setup
- Create the SQLite database with 4 tables
- Populate with realistic data (use the generator)
- Write and test individual SQL queries for each metric

### Day 2: Query Library
- Create `src/queries.py` with functions that return DataFrames
- Each function takes a `connection` and returns a `pd.DataFrame`
- Test all queries in the notebook

### Day 3: Visualizations & Dashboard
- Create 6-8 charts covering all key metrics
- Build an HTML report or Streamlit app
- Add executive summary section

## How to Run
```bash
# Set up the database
python src/setup_db.py

# Run the notebook
jupyter lab notebook.ipynb

# Or run the Streamlit dashboard (optional)
# streamlit run src/dashboard.py
```
