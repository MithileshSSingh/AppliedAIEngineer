# P5: ML ROI Dashboard & Presentation

## Objective
Calculate the business ROI of the churn prediction model and build an executive-facing Streamlit dashboard for monitoring and stakeholder communication.

## Architecture
```
Production Metrics → ROI Analysis → Streamlit Dashboard → Executive Summary
```

## Key Skills
- ML ROI calculation (cost-benefit, NPV, sensitivity analysis)
- Production monitoring (system, model, data, business metrics)
- Streamlit dashboard development
- Stakeholder communication and executive summaries

## Deliverables
1. `notebook.ipynb` — ROI analysis and monitoring visualizations
2. `src/dashboard.py` — Streamlit monitoring and ROI dashboard

## Suggested Approach
**Week 14, Thursday-Friday:**
1. Generate 90 days of simulated production metrics
2. Build monitoring visualizations
3. Calculate churn model ROI with cost-benefit analysis
4. Run sensitivity analysis on key parameters
5. Build the Streamlit dashboard
6. Write executive summary

## How to Run
```bash
cd module-3-mlops-production/projects/p5-roi-presentation
jupyter lab notebook.ipynb
streamlit run src/dashboard.py
```
