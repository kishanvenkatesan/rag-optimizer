# RAG Optimizer Tool 🧠📚

A production style light weight Retrieval-Augmented Generation (RAG) experimentation platform with:

- 🔎 Live RAG API (FastAPI)
- 📊 Optimization dashboard (Streamlit)
- ⚡ Two-stage optimization (fast retrieval screening -> full RAG evaluation)
- 🔁 Hot-swappable local corpora
- 🏆 Experiment tracking & leaderboard

This repository is designed to be cloned and run locally in minutes.

---

# 🔧 Installation (Recommended: Conda)

This project uses NumPy, SciPy, PyTorch, and Transformers.  
To avoid binary compatibility issues (especially on Windows), use the provided Conda environment.

### Create the environment

```
conda env create -f environment.yml
conda activate rag-optimizer
```

This installs:
- Python 3.10
- NumPy (pinned < 2.0)
- SciPy / scikit-learn (compatible versions)
- PyTorch (CPU build)
- Transformers stack
- FastAPI
- Streamlit

---

# 🚀 Running the Application

You need to run two processes:
1. The FastAPI backend
2. The Streamlit dashboard

## 1. Prepare the dataset

For the sake of testing, try out the included Wikipedia Corpus from Huggingface (rag-mini-wikipedia)

To build a small local Wikipedia corpus + QA evaluation set (It is already available in the data folder):
```
python scripts/prepare_wikipedia_corpus.py
```

This creates:
```
data/
  wiki_docs.jsonl
  wiki_eval.jsonl
```

You can try out your own corpus by modifying `prepare_wikipedia_corpus.py` script to generate a suitable corpus of the JSONL format:
```json
{"id": 0, "text": "Document text here"}
```

## 2. Start the FastAPI Backend

From the project root, run:
```
python -m uvicorn api.main:app --reload --port 9000
```

If successful, you'll see:
```
Uvicorn running on http://127.0.0.1:9000
```

### Useful API URLs:

Swagger UI:
```
http://127.0.0.1:9000/docs
```

Health Check:
```
http://127.0.0.1:9000/health
```

## 3. Start the Dashboard (Streamlit)

Open a second terminal (same environment active):
```
streamlit run ui/app.py
```

Streamlit will show a local URL, typically:
```
http://localhost:8501
```

Open it in your browser

# 🔎 Using the App

![alt text](image.png)

## 💬 Live RAG

- Ask questions against the current corpus
- Inspect retrieved contexts
- View reranker scores
- Adjust `k` and reranker settings

## 🔁 Change Corpus Without Restarting

From the UI sidebar:

- Set a new `docs_path`
- Click **Reload corpus in API**
- Live RAG immediately uses the new corpus

## 🧪 Optimize RAG Configuration

Use the Optimize tab to tune:

- chunk_size
- overlap
- top-k
- reranker (on/off)

**Two-Stage Strategy**

To avoid slow “full RAG for every config”, the optimizer uses:
- Stage 1:
    Retrieval-only ranking across many configs (fast)
- Stage 2
    Full RAG (generation + LLM judge) only on top configs (expensive)

This makes large datasets practical on CPU and matches real-world evaluation workflows.

Results are saved to:
```
experiments/run_YYYYMMDD_HHMMSS.json
```

## 🏆 Leaderboard

The Leaderboard tab:

- Lists all experiment runs
- Shows best configs
- Displays combined score:
```
combined = 0.7 * judge + 0.2 * f1 + 0.1 * recall
```
- Provides copy-ready configuration snippet for production

# 🧠 Design Principles

- Local-first corpus (no external DB required)
- Modular retriever / reranker / generator
- Instruction-tuned generator (Flan-T5)
- Long-context optional
- Two-stage optimization for scalability
- UI controls the entire workflow

# ⚠️ Troubleshooting

### 1. Numpy/Scipy Errors

Use:
```
conda env create -f environment.yml
```

**Do not install NumPy 2.x manually**

### 2. Torch GPU

The provided environment uses CPU-only PyTorch.
If you want GPU support, modify: ` cpuonly` in `environment.yml` to a CUDA-compatible configuration.

### 3. `ModuleNotFoundError: autoreg`

Run Streamlit from project root OR ensure `ui/app.py` adds repo root to `sys.path`

# Updates I am currently working on:

- Background optimization jobs (non-blocking UI)
- Better charts + config drilldown
- Docker Compose for one-command startup
- Hosted deployment (HF Spaces + Render)