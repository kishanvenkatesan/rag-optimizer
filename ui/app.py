import json
import time
import sys
import requests
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from autoreg.data import load_qa_dataset, load_docs_corpus
from autoreg.optimizer import two_stage_search  


EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

DEFAULT_API_URL = "http://127.0.0.1:9000"

st.set_page_config(
    page_title="RAG Optimizer Tool",
    page_icon="🧠",
    layout="wide",
)

# ---------- Helpers ----------

def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_run(run_id: str, payload: dict):
    out = EXPERIMENTS_DIR / f"run_{run_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out

def list_runs():
    return sorted(EXPERIMENTS_DIR.glob("run_*.json"), reverse=True)

def load_run(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def results_to_df(results_list: list) -> pd.DataFrame:
    """
    results_list: list of dicts with config + metrics
    """
    rows = []
    for r in results_list:
        cfg = r.get("config", {})
        m = r.get("metrics", {})
        rows.append({
            "chunk_size": cfg.get("chunk_size"),
            "overlap": cfg.get("overlap"),
            "k": cfg.get("k"),
            "use_reranker": cfg.get("use_reranker", False),
            "top_n": cfg.get("top_n", None),
            "avg_judge": m.get("avg_judge_score", None),
            "avg_f1": m.get("avg_f1_answer", None),
            "avg_recall": m.get("avg_recall_at_k", None),
            "combined": m.get("combined_score", None),
        })
    df = pd.DataFrame(rows)
    # best-first
    if "combined" in df.columns and df["combined"].notna().any():
        df = df.sort_values(["combined", "avg_judge", "avg_f1", "avg_recall"], ascending=False)
    return df

def compute_combined(avg_judge, avg_f1, avg_recall, wj=0.7, wf=0.2, wr=0.1):
    if avg_judge is None: avg_judge = 0.0
    if avg_f1 is None: avg_f1 = 0.0
    if avg_recall is None: avg_recall = 0.0
    return (wj * avg_judge) + (wf * avg_f1) + (wr * avg_recall)

def run_two_stage_job(
    dataset_path: str,
    docs_path: str,
    search_space: dict,
    stage1_max_eval: int,
    stage2_max_eval: int,
    top_n_configs: int,
) -> dict:
    """
    Runs optimizer locally (in the Streamlit process).
    Writes a run artifact into experiments/.
    """
    t0 = time.time()

    dataset = load_qa_dataset(dataset_path)
    docs = load_docs_corpus(docs_path)

    # Run two-stage
    results = two_stage_search(
        dataset=dataset,
        docs_corpus=docs,
        search_space=search_space,
        stage1_max_eval=stage1_max_eval,
        stage2_max_eval=stage2_max_eval,
        top_n_configs=top_n_configs,
    )

    # Pack results
    packed = []
    for r in results:
        cfg = r.config
        metrics = {
            "avg_recall_at_k": float(r.avg_recall_at_k),
            "avg_f1_answer": float(r.avg_f1_answer),
            "avg_judge_score": float(r.avg_judge_score),
        }
        metrics["combined_score"] = float(compute_combined(
            metrics["avg_judge_score"], metrics["avg_f1_answer"], metrics["avg_recall_at_k"]
        ))
        packed.append({
            "config": {
                "chunk_size": int(cfg.chunk_size),
                "overlap": int(cfg.overlap),
                "k": int(cfg.k),
                "use_reranker": bool(getattr(cfg, "use_reranker", False)),
                "top_n": int(getattr(cfg, "top_n", search_space.get("top_n", 20))),
            },
            "metrics": metrics
        })

    run_id = _now_id()
    payload = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": dataset_path,
        "docs_path": docs_path,
        "search_space": search_space,
        "stage1_max_eval": stage1_max_eval,
        "stage2_max_eval": stage2_max_eval,
        "top_n_configs": top_n_configs,
        "duration_sec": round(time.time() - t0, 2),
        "results": packed,
    }
    out_path = save_run(run_id, payload)
    payload["_saved_to"] = str(out_path)
    return payload


# ---------- Sidebar ----------

st.sidebar.title("RAG Optimizer Tool")
st.sidebar.caption("Production-style RAG tuning + demo UI")

api_url = st.sidebar.text_input("API URL (FastAPI RAG service)", DEFAULT_API_URL)

st.sidebar.divider()
st.sidebar.subheader("Data paths")
dataset_path = st.sidebar.text_input("Eval dataset (QA) JSONL", "data/wiki_eval.jsonl")
docs_path = st.sidebar.text_input("Corpus (docs) JSONL", "data/wiki_docs.jsonl")

st.sidebar.divider()
st.sidebar.subheader("Serving corpus control")

if st.sidebar.button("Reload corpus in API", type="primary"):
    try:
        res = requests.post(
            f"{api_url}/reload_corpus",
            json={"docs_path": docs_path},
            timeout=600
        )
        st.sidebar.write(res.json())
    except Exception as e:
        st.sidebar.error(f"Reload failed: {e}")

if st.sidebar.button("API health check"):
    try:
        res = requests.get(f"{api_url}/health", timeout=30)
        st.sidebar.write(res.json())
    except Exception as e:
        st.sidebar.error(f"Health check failed: {e}")

st.sidebar.divider()
st.sidebar.subheader("Optimization settings")

chunk_sizes = st.sidebar.multiselect("Chunk sizes", [128, 256, 512, 768], default=[256, 512])
overlaps = st.sidebar.multiselect("Overlaps", [0, 16, 32, 64, 128], default=[32, 64])
ks = st.sidebar.multiselect("Top-k", [1, 3, 5, 8], default=[1, 3, 5])
use_reranker_opts = st.sidebar.multiselect("Reranker", [False, True], default=[False, True])
top_n = st.sidebar.slider("Retrieve top-N before rerank", 5, 50, 20, step=5)

st.sidebar.caption("Two-stage evaluation sizes (keep small to finish fast)")
stage1_max_eval = st.sidebar.slider("Stage 1 (retrieval-only) eval size", 50, 500, 200, step=50)
stage2_max_eval = st.sidebar.slider("Stage 2 (full RAG) eval size", 20, 200, 60, step=10)
top_n_configs = st.sidebar.slider("Configs to promote to Stage 2", 2, 10, 5, step=1)

search_space = {
    "chunk_size": chunk_sizes,
    "overlap": overlaps,
    "k": ks,
    "use_reranker": use_reranker_opts,
    "top_n": top_n,
}

# ---------- Main UI ----------

st.title("🧠 RAG Optimizer Dashboard")
st.caption("Live RAG + Two-stage optimization + Leaderboard")

tab_live, tab_opt, tab_lb = st.tabs(["💬 Live RAG", "🧪 Optimize", "🏆 Leaderboard"])

# ===== Tab 1: Live RAG =====
with tab_live:
    colA, colB = st.columns([2, 1], gap="large")
    with colA:
        st.subheader("Ask the knowledge base")
        q = st.text_input("Question", "Do women live longer than men?")
        k_live = st.slider("k (contexts used)", 1, 8, 3)
        if st.button("Ask", type="primary"):
            import requests
            try:
                t0 = time.time()
                res = requests.get(f"{api_url}/query", params={"q": q, "k": k_live}, timeout=120)
                latency = round((time.time() - t0) * 1000, 1)
                if res.status_code != 200:
                    st.error(f"API error {res.status_code}: {res.text}")
                else:
                    body = res.json()
                    st.success(f"Answer generated in {latency} ms")
                    st.markdown("### Answer")
                    st.write(body.get("answer", ""))

                    st.markdown("### Retrieved contexts")
                    chunks = body.get("retrieved_chunks", [])
                    doc_ids = body.get("retrieved_doc_ids", [])
                    scores = body.get("reranker_scores", None)

                    for i, txt in enumerate(chunks):
                        title = f"Context {i+1}"
                        if i < len(doc_ids):
                            title += f" (doc_id={doc_ids[i]})"
                        if scores and i < len(scores):
                            title += f" | rerank={scores[i]:.3f}"
                        with st.expander(title, expanded=(i == 0)):
                            st.write(txt)

            except Exception as e:
                st.error(f"Failed to call API: {e}")

    with colB:
        st.subheader("Health check")
        st.write("Make sure your FastAPI service is running.")
        st.code("uvicorn api.main:app --reload --port 9000")
        st.write("Swagger:")
        st.code(f"{api_url}/docs")

# ===== Tab 2: Optimize =====
with tab_opt:
    st.subheader("Two-stage optimization (fast → accurate)")
    st.write(
        "Stage 1 ranks configs with retrieval metrics (fast). "
        "Stage 2 runs full RAG (generation + judge) only on the top configs."
    )

    c1, c2, c3 = st.columns([1, 1, 1], gap="large")
    with c1:
        st.metric("Chunk sizes", len(chunk_sizes))
    with c2:
        st.metric("Overlaps", len(overlaps))
    with c3:
        st.metric("k values", len(ks))

    total_cfg = len(chunk_sizes) * len(overlaps) * len(ks) * len(use_reranker_opts)
    st.info(f"Candidate configs (Stage 1): **{total_cfg}** | Promoted to Stage 2: **{top_n_configs}**")

    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None

    run_btn = st.button("Run optimization now", type="primary")
    if run_btn:
        with st.status("Running two-stage optimization…", expanded=True) as status:
            st.write("Loading dataset + corpus…")
            st.write(f"QA: {dataset_path}")
            st.write(f"Docs: {docs_path}")
            st.write("Search space:")
            st.json(search_space)

            st.write("Stage 1: retrieval-only…")
            st.write("Stage 2: full RAG on top configs… (this is the slow part)")

            try:
                payload = run_two_stage_job(
                    dataset_path=dataset_path,
                    docs_path=docs_path,
                    search_space=search_space,
                    stage1_max_eval=stage1_max_eval,
                    stage2_max_eval=stage2_max_eval,
                    top_n_configs=top_n_configs,
                )
                st.session_state["last_run"] = payload
                status.update(label="Optimization complete ✅", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Optimization failed ❌", state="error", expanded=True)
                st.exception(e)

    if st.session_state["last_run"]:
        run = st.session_state["last_run"]
        st.success(f"Saved run: {run.get('_saved_to', '')} | duration: {run.get('duration_sec')}s")
        df = results_to_df(run["results"])
        st.dataframe(df, use_container_width=True)

        # Simple chart: combined score by config row
        if df["combined"].notna().any():
            fig = plt.figure()
            plt.plot(list(range(1, len(df) + 1)), df["combined"].tolist())
            plt.title("Combined score (sorted best → worst)")
            plt.xlabel("Rank")
            plt.ylabel("Combined score")
            st.pyplot(fig, clear_figure=True)

# ===== Tab 3: Leaderboard =====
with tab_lb:
    st.subheader("Experiment leaderboard")
    run_files = list_runs()
    if not run_files:
        st.warning("No runs found yet. Go to Optimize tab and run an optimization.")
    else:
        run_labels = [p.name for p in run_files]
        selected = st.selectbox("Select a run", run_labels, index=0)
        selected_path = EXPERIMENTS_DIR / selected
        run = load_run(selected_path)

        meta_cols = st.columns(4)
        meta_cols[0].metric("Run ID", run.get("run_id", ""))
        meta_cols[1].metric("Duration (s)", run.get("duration_sec", ""))
        meta_cols[2].metric("Stage1 eval", run.get("stage1_max_eval", ""))
        meta_cols[3].metric("Stage2 eval", run.get("stage2_max_eval", ""))

        df = results_to_df(run.get("results", []))
        st.dataframe(df, use_container_width=True)

        if not df.empty:
            st.markdown("### Best config")
            best = df.iloc[0].to_dict()
            st.json(best)
            st.markdown("### Apply to API (manual)")
            st.code(
                f"# In api/rag_service.py set defaults:\n"
                f"DEFAULT_CHUNK_SIZE = {int(best['chunk_size'])}\n"
                f"DEFAULT_OVERLAP = {int(best['overlap'])}\n"
                f"DEFAULT_TOP_K = {int(best['k'])}\n"
                f"DEFAULT_TOP_N = {int(best.get('top_n') or 20)}\n"
                f"USE_RERANKER = {bool(best.get('use_reranker'))}\n"
            )
