# Claude's Plan

## Applied AI Engineering Self-Study Implementation Plan

## Context

Building a self-study curriculum for Applied AI Engineering across 8 modules and about 34 weeks. The learner is a beginner with basic Python, can commit 15-20 hours per week, and will follow the modules sequentially. AWS access and LLM API keys are already available. The goal is to progress from Python fundamentals to production-grade agentic AI systems.

## 1. Repo Structure

```text
applied-ai-engineer-upgrade/
├── README.md                     # Portfolio page + curriculum map
├── PROGRESS.md                   # Weekly check-in log
├── .gitignore                    # Ignore .env, data/*, checkpoints, etc.
├── .env.example                  # API key template (never commit .env)
├── environment.yml               # Conda env (Python 3.11)
├── pyproject.toml                # Makes shared/ importable
├── requirements-base.txt         # Core libs for all modules
│
├── shared/                       # Reusable utilities
│   ├── __init__.py
│   ├── data_utils.py
│   ├── viz_utils.py
│   ├── eval_utils.py
│   ├── llm_utils.py
│   └── config.py                 # Loads .env
│
├── module-1-foundations/
│   ├── README.md
│   ├── requirements.txt
│   ├── notebooks/                # Concept notebooks (theory + code)
│   ├── mini-projects/            # Small exercises
│   └── projects/                 # Main deliverables
│       ├── p1-sales-pipeline/
│       ├── p2-cohort-retention/
│       ├── p3-feature-engineering/
│       └── p4-sql-revenue-dashboard/
│
├── module-2-ml-deep-learning/    # Same pattern
├── module-3-mlops-production/
├── module-4-capstone-a/
├── module-5-llm-rag/
├── module-6-agent-frameworks/
├── module-7-finetuning-production/
├── module-8-capstone-b/
│
└── templates/                    # Reusable project/notebook templates
```

Each project is self-contained with its own `README`, `src/`, `data/`, and `outputs/`.

## 2. Week 0 Setup

- Initialize git and create `.gitignore`.
- Create the Conda environment:

  ```bash
  conda create -n ai-engineer python=3.11
  ```

- Install base libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `jupyterlab`, `python-dotenv`.
- Set up `.env` with API keys and keep `.env.example` as the template.
- Create `pyproject.toml` and the `shared/` package, then run:

  ```bash
  pip install -e .
  ```

- Verify AWS CLI:

  ```bash
  aws sts get-caller-identity
  ```

- Verify Docker:

  ```bash
  docker run hello-world
  ```

- Create the repo skeleton for all modules, READMEs, and templates.

## 3. Weekly Rhythm

| Day | Activity | Hours |
| --- | --- | --- |
| Mon | Concept Notebook A (read + code along) | 3-4 |
| Tue | Concept Notebook B (or deeper dive) | 3-4 |
| Wed | Mini-project (apply concepts) | 3-4 |
| Thu | Main project work | 3-4 |
| Fri | Main project + README + commit | 3-4 |

## 4. Module-by-Module Breakdown

### Module 1: Foundations and Data Engineering (Weeks 1-4)

**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `sqlalchemy`, `scikit-learn`, `openpyxl`

| Week | Concepts | Mini-Project | Main Project |
| --- | --- | --- | --- |
| 1 | Python OOP, advanced Pandas | OOP inventory system | P1: Sales Pipeline (start) |
| 2 | SQL (joins, CTEs, windows), EDA | 15 SQL challenges | P1: Sales Pipeline (finish) |
| 3 | Feature engineering, scaling, encoding | Feature transforms | P2: Cohort Retention + P3: Feature Engineering Competition |
| 4 | Git practices, business visualization | - | P4: SQL Revenue Dashboard |

**Resources:** Corey Schafer, Kaggle Learn, SQLBolt, Mode Analytics SQL

### Module 2: ML and Deep Learning (Weeks 5-9)

**Libraries:** `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `torch`, `torchvision`, `transformers`, `prophet`

| Week | Concepts | Mini-Project | Main Project |
| --- | --- | --- | --- |
| 5 | Linear/logistic regression, trees, boosting | House price regression | P1: Churn Model (start) |
| 6 | SHAP/LIME interpretability | SHAP deep dive | P1: Churn Model (finish) |
| 7 | Neural nets (PyTorch), CNNs, transfer learning | MNIST from scratch | P2: Image Classifier |
| 8 | RNNs/LSTMs, Transformers, BERT/GPT | Text classification comparison | P3: NLP Sentiment |
| 9 | Time series (ARIMA/Prophet), clustering/PCA | - | P4: Time Series + P5: Segmentation |

**Resources:** StatQuest, fast.ai, PyTorch 60-Minute Blitz, Hugging Face NLP Course, Jay Alammar

### Module 3: MLOps and Production (Weeks 10-14)

**Libraries:** `boto3`, `sagemaker`, `mlflow`, `fastapi`, `uvicorn`, `docker`, `evidently`, `streamlit`

| Week | Concepts | Mini-Project | Main Project |
| --- | --- | --- | --- |
| 10 | AWS (S3/IAM/Lambda), Boto3 | S3 data pipeline | P1: Cloud Pipeline |
| 11 | SageMaker training/endpoints | - | P2: Distributed Training |
| 12 | Docker, FastAPI | FastAPI calculator | P3: Dockerized API |
| 13 | MLflow, A/B testing, drift detection | - | P4: MLOps Pipeline |
| 14 | Monitoring, ROI calculation | - | P5: ROI Presentation |

**Resources:** AWS Free Tier, Made With ML, FastAPI docs, MLflow tutorials

### Module 4: Capstone A (Weeks 15-17)

**Track options:** Predictive Maintenance or Customer Intelligence

- Week 15: Architecture, data pipeline, and MLflow setup
- Week 16: Model training, SHAP, deployment to SageMaker or Docker
- Week 17: Monitoring, README, presentation deck, and demo

### Module 5: LLM Foundations and RAG (Weeks 18-20)

**Libraries:** `openai`, `anthropic`, `langchain`, `chromadb`, `tiktoken`, `sentence-transformers`, `faiss-cpu`

| Week | Concepts | Mini-Project | Main Project |
| --- | --- | --- | --- |
| 18 | LLM APIs, token economics, prompt engineering | Prompt library (10+ prompts) | P1: Prompt Engineering Challenge |
| 19 | Embeddings, similarity, vector DBs | Semantic search engine | P2: RAG Prototype |
| 20 | RAG patterns, chunking, evaluation | - | P3: Production RAG API |

**Resources:** OpenAI Cookbook, Anthropic prompt guide, DeepLearning.AI short courses, ChromaDB docs

### Module 6: Agent Frameworks (Weeks 21-25)

**Libraries:** `langchain`, `langgraph`, `llama-index`, `autogen-agentchat`, `crewai`, `tavily-python`

| Week | Concepts | Mini-Project | Main Project |
| --- | --- | --- | --- |
| 21 | Agent architectures, ReAct, tool use | ReAct agent from scratch | P1: Agent Architecture |
| 22 | LangChain, LangGraph | - | P2: LangChain Research Assistant |
| 23 | LlamaIndex | - | P3: Multi-Doc Knowledge System |
| 24 | AutoGen | - | P4: Multi-Agent Collaboration |
| 25 | CrewAI | - | P5: Production Content Crew |

**Resources:** Lilian Weng's agent blog, LangChain docs, DeepLearning.AI CrewAI course

### Module 7: Fine-Tuning and Production (Weeks 26-30)

**Libraries:** `peft`, `trl`, `bitsandbytes`, `datasets`, `ray[serve]`, `locust`

| Week | Concepts | Mini-Project | Main Project |
| --- | --- | --- | --- |
| 26 | LoRA/QLoRA, PEFT, dataset prep | Fine-tune classifier | P1: Fine-Tuning with LoRA |
| 27 | AWS Bedrock, guardrails | - | P2: Bedrock Deployment |
| 28 | Ray Serve, CI/CD, GitHub Actions | - | P3: CI/CD Pipeline |
| 29 | LLM security, OWASP Top 10, red teaming | - | P4: Security Red Team Drill |
| 30 | Cost optimization, caching, model routing | - | P5: Cost Optimization Report |

**Resources:** Hugging Face PEFT docs, Maxime Labonne LLM Course, OWASP LLM Top 10

### Module 8: Capstone B (Weeks 31-34)

**Track options:** Enterprise Assistant, Autonomous Research, or Business Operations

- Week 31: Architecture, CI/CD, core agent
- Week 32: Full implementation (tools, RAG, API, Docker)
- Week 33: Security hardening, load testing, monitoring
- Week 34: Documentation, demo, retrospective

## 5. Budget Estimate

Estimated total: **$150-300**

- AWS: Free Tier should cover most of Modules 1-2. SageMaker/Bedrock may cost about `$20-50` per module. Set billing alerts.
- OpenAI/Anthropic: roughly `$10-20` per month during Modules 5-8. Use cheaper models for development.
- Cost control tips: cache API responses, shut down endpoints quickly, and use spot instances where appropriate.

## 6. Progress Tracking

- `PROGRESS.md`: weekly check-in with hours, key takeaway, struggles, and next focus
- Per-project `README`: objective, approach, key learnings, results, and retrospective
- Git discipline: daily commits, conventional commit messages, and `git tag module-N-complete`
- Root `README.md`: update the portfolio page after each module

## 7. Verification

### After Setup (Week 0)

```bash
conda activate ai-engineer && python -c "import pandas; print(pandas.__version__)"
python -c "from shared.config import *"
aws sts get-caller-identity
docker run hello-world
git log --oneline
```

### After Each Module

- All project notebooks run end-to-end without errors.
- Each project has a `README` with results.
- `PROGRESS.md` is updated with reflections.
- A git tag is created for module completion.

## 8. Key Resources

| Topic | Resource |
| --- | --- |
| Python OOP | Corey Schafer YouTube |
| Pandas / SQL / Feature Engineering | Kaggle Learn courses |
| SQL Practice | SQLBolt, Mode Analytics |
| ML Fundamentals | StatQuest YouTube |
| Deep Learning | fast.ai Practical DL |
| PyTorch | Official 60-Minute Blitz |
| NLP / Transformers | Hugging Face NLP Course |
| MLOps | Made With ML (Goku Mohandas) |
| LLM Prompting | DeepLearning.AI short courses |
| Agents | Lilian Weng's blog |
| Fine-Tuning | Maxime Labonne LLM Course (GitHub) |
| Security | OWASP LLM Top 10 |
| Paid (~$35) | Chip Huyen, *Designing ML Systems* |

## Implementation Order

1. Execute Week 0 setup: create the repo skeleton, Conda env, `.gitignore`, and `shared` package.
2. Start Module 1, Week 1 with the Python OOP concept notebook.
3. Continue sequentially through all 34 weeks.
