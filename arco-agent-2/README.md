# ARCO Agent 2.0 — Starter Project

ARCO Agent 2.0 is a prototype for AI-powered retail marketing strategy generation and sales/profit optimization.
**All included data and policy rules are synthetic examples, not Alshaya data or actual company policies.**

## Architecture (first milestone)

- **Frontend:** React dashboard loaded from CDN — no Node.js/build step required for this starter.
- **Backend:** FastAPI (Python).
- **Data:** Synthetic CSV files generated locally.
- **Optimization:** Deterministic financial calculations and candidate-offer comparison.
- **Later phases:** demand prediction, customer segmentation, Groq marketing generation, RAG, and LangGraph.

## Requirements

- Python 3.10–3.12 recommended.
- Internet access in the browser to load React/Babel CDN scripts.
- Node.js is not required for this starter frontend.

## 1. Create and activate a virtual environment (Windows CMD)

```bat
cd arco-agent-2
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.11 is not installed, use `py -0p` to see available Python versions, then create the venv with one of them supported by your installed packages.

## 2. Generate synthetic data

From the project root:

```bat
python -m scripts.generate_synthetic_data
```

Generated CSVs are written to `data/synthetic/`.

## 3. Start the API

In the first terminal, from the project root with the virtual environment activated:

```bat
uvicorn backend.main:app --reload
```

Check:
- http://127.0.0.1:8000/api/health
- http://127.0.0.1:8000/docs

## 4. Start the React dashboard (second terminal)

Open a second terminal, activate the same environment, and run from the project root:

```bat
python -m http.server 5500 --directory frontend
```

Open http://127.0.0.1:5500

Keep both terminals running.

## 5. What works in this starter

- Synthetic products, customers, stores, transactions, inventory, campaigns, and demo brand rules.
- Summary KPIs from generated data.
- A basic offer comparison endpoint that applies stock, discount, margin, and budget constraints.
- React dashboard for API health, KPIs, and sample strategy comparison.

## Important modeling note

The current optimizer is a **first-step demonstration**, not a trained prediction system. Its example demand curve is deliberately transparent. We will replace it with a trained demand model after the data generator and financial logic are checked. It also uses simplified product-level calculations; multi-product campaigns and bundles will be added later.

## Next milestones

1. Validate the generated data and financial engine.
2. Add RFM customer segmentation and demand model.
3. Add campaign-response estimation and a proper no-campaign baseline.
4. Add strategy generation with Groq (using an environment variable for the API key).
5. Add RAG over synthetic demo policies.
6. Add LangGraph orchestration and human approval workflow.
