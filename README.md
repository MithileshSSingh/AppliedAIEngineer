# Applied AI Engineering — Self-Study Curriculum

A project-centric, 34-week self-study program going from Python fundamentals to production agentic AI systems.

## Curriculum Overview

| Module | Topic | Weeks | Projects | Status |
|--------|-------|-------|----------|--------|
| 1 | [Foundations & Data Engineering](module-1-foundations/) | 1-4 | Sales Pipeline, Cohort Retention, Feature Eng, SQL Dashboard | [ ] |
| 2 | [ML & Deep Learning](module-2-ml-deep-learning/) | 5-9 | Churn Model, Image Classifier, NLP Sentiment, Time Series, Segmentation | [ ] |
| 3 | [MLOps & Production](module-3-mlops-production/) | 10-14 | Cloud Pipeline, Distributed Training, Dockerized API, MLOps Pipeline, ROI | [ ] |
| 4 | [Capstone A: End-to-End ML](module-4-capstone-a/) | 15-17 | Full ML system with deployment & monitoring | [ ] |
| 5 | [LLM Foundations & RAG](module-5-llm-rag/) | 18-20 | Prompt Engineering, RAG Prototype, Production RAG API | [ ] |
| 6 | [Agent Frameworks](module-6-agent-frameworks/) | 21-25 | Agent Architecture, LangChain, LlamaIndex, AutoGen, CrewAI | [ ] |
| 7 | [Fine-Tuning & Production](module-7-finetuning-production/) | 26-30 | LoRA Fine-Tuning, Bedrock, CI/CD, Red Team, Cost Report | [ ] |
| 8 | [Capstone B: Agentic AI](module-8-capstone-b/) | 31-34 | Production agentic AI system | [ ] |

## Getting Started

```bash
# 1. Create conda environment
conda create -n ai-engineer python=3.11 -y
conda activate ai-engineer

# 2. Install base dependencies
pip install -r requirements-base.txt

# 3. Install shared utilities (editable mode)
pip install -e .

# 4. Copy .env.example to .env and add your API keys
cp .env.example .env

# 5. Start with Module 1
cd module-1-foundations
pip install -r requirements.txt
jupyter lab
```

## Weekly Rhythm (15-20 hrs)

| Day | Activity |
|-----|----------|
| Mon | Concept Notebook A — theory + code examples |
| Tue | Concept Notebook B — deeper exploration |
| Wed | Mini-Project — hands-on practice |
| Thu | Main Project — build the deliverable |
| Fri | Main Project — finish, document, commit |

## Project Highlights

*Updated as modules are completed.*

---

Built with dedication, one week at a time.
