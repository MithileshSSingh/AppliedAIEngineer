# P3: Feature Engineering Competition

## Objective
Compete on a Kaggle "Getting Started" competition with a focus on **feature engineering** — not model complexity. The goal is to maximize your leaderboard score using creative features with a simple model.

## Recommended Competitions (Pick One)
1. **House Prices** — https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
   - Regression task (predict sale price)
   - Rich feature set: 79 variables (numeric + categorical)
   - Great for practicing encoding, interactions, domain features

2. **Spaceship Titanic** — https://www.kaggle.com/competitions/spaceship-titanic
   - Classification task (transported or not)
   - Mixed features with missing data
   - Good for practicing imputation and creative encoding

## Rules for This Project
1. **Model constraint:** Use only `LinearRegression`/`LogisticRegression` or `RandomForestRegressor`/`RandomForestClassifier` with default hyperparameters
2. **Focus area:** All improvement must come from features, not model tuning
3. **Track experiments:** Log every feature set and its CV score
4. **Document insights:** Write up which features helped and why

## Skills Practiced
- All feature engineering techniques from the concept notebook
- Working with real messy data (missing values, mixed types)
- Kaggle submission workflow
- Experiment tracking and iteration

## Deliverables
1. `notebook.ipynb` — Full feature engineering pipeline
2. `experiments.md` — Log of all experiments with scores
3. At least one Kaggle submission
4. Write-up: top 5 features that improved your score

## Suggested Workflow

### Phase 1: Baseline (1-2 hours)
- Download data, do quick EDA
- Train a model with minimal preprocessing
- Submit to Kaggle → record baseline score

### Phase 2: Feature Engineering (2-3 hours)
Apply techniques from the notebook in rounds:
1. Handle missing values properly
2. Log-transform skewed numerics
3. Create interaction features (area × quality, etc.)
4. Encode categoricals (ordinal for ordered, one-hot for nominal)
5. Domain-specific features (e.g., total square footage, age of house)
6. Bin continuous variables
7. Target encode high-cardinality categoricals

After each round: re-evaluate with cross-validation.

### Phase 3: Polish (30 min)
- Build final sklearn Pipeline
- Generate final predictions
- Submit to Kaggle
- Write up findings

## How to Run
```bash
# Download data from Kaggle (requires kaggle CLI)
# kaggle competitions download -c house-prices-advanced-regression-techniques
# unzip house-prices-advanced-regression-techniques.zip -d data/

jupyter lab notebook.ipynb
```
