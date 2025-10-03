# BEIR Retrieval Pipeline — README

This repository contains the BEIR retrieval + rerank + LTR pipeline (the `main.py` script you provided). The README below explains, step-by-step, how to set up, run, evaluate, and prepare your submission as a **public GitHub repository** for your assignment.

---

## TL;DR (Quick submission checklist)

1. Create a public GitHub repository and push your project files.
2. Include these files in the repo: `main.py`, `README.md` (this file), `requirements.txt`, `LICENSE`, `.gitignore`, `results/` (optional), and any helper scripts or notebooks.
3. Provide clear run instructions (see **Run & Evaluate** section).
4. Make sure `README.md` lists exact steps to reproduce and the expected output files (e.g., `cache_beir/results/<dataset>_results.json`).
5. Paste the public GitHub URL into your assignment submission.

---

## Recommended repository structure

```
beir-retrieval-pipeline/
├─ main.py
├─ requirements.txt
├─ README.md        # this file
├─ LICENSE
├─ .gitignore
├─ scripts/
│  ├─ run_local.sh
│  └─ run_gpu.sh
├─ notebooks/       # optional: quick exploratory notebooks
├─ results/         # (optional) final JSON results you want to include
└─ cache_beir/      # (do NOT commit large caches; add to .gitignore)
```

---

## 1) Prepare your code for submission

1. Put the full pipeline (`main.py`) at the repo root (you already have it).
2. Add a short wrapper script `scripts/run_quick_test.sh` that modifies `CONFIG` for a small smoke test (see example below). This makes CI and graders able to run a fast check.
3. Add a `requirements.txt` with the exact Python packages used (example below).
4. Add a `.gitignore` to avoid committing caches, virtual environments, or large files.
5. Add a license (for example, `MIT`) if required by the assignment.

**Example `scripts/run_quick_test.sh`**

```bash
#!/usr/bin/env bash
# Quick smoke run: reduced corpus/queries for grader testing
python - <<'PY'
from main import CONFIG
CONFIG['max_docs'] = 200
CONFIG['max_queries'] = 20
CONFIG['datasets'] = ['fiqa']
CONFIG['encode_batch_size'] = 64
CONFIG['rerank_batch_size'] = 32
CONFIG['retrieval_k'] = 100
CONFIG['rerank_k'] = 30
CONFIG['device'] = 'cpu'  # safe default for CI/grader
from main import main as entry
entry()
PY
```

---

## 2) `requirements.txt` (example)

Put this at the repo root. You can pin versions if you like; the following is a minimal example:

```
python>=3.8
numpy
torch
sentence-transformers
cross-encoder
beir
faiss-cpu      # or faiss-gpu if you want GPU support
rank_bm25
lightgbm
tqdm

# Optional / helpers
scikit-learn
pandas
```

To freeze exact versions after successful testing:

```bash
pip freeze > requirements.txt
```

---

## 3) `.gitignore` (example)

```
__pycache__/
*.pyc
venv/
.env
cache_beir/
*.index
*.npy
.DS_Store
results/
```

> **Note:** Do not commit large caches (FAISS indices, embeddings, dataset zips). Keep `cache_beir/` ignored.

---

## 4) Create a public GitHub repository and push your code

### Option A — CLI (recommended)

1. Create the repo on GitHub (or use `gh` CLI):

   * `gh repo create <your-username>/<repo-name> --public --source=. --remote=origin`
2. Push locally if not using `gh`:

```bash
git init
git add .
git commit -m "Initial commit: BEIR retrieval pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### Option B — GitHub website

1. Create a new repository on [https://github.com](https://github.com) (click New → Repository).
2. Follow the web instructions to push an existing repository from the command line (they will show the exact `git remote add` and `git push` commands).

After pushing, verify that the repo is **public** and the README renders on GitHub.

---

## 5) Run & Evaluate (reproduce results locally)

### Prerequisites

* Python 3.8+ installed.
* `pip` available (or `conda`/`mamba` if using conda env).
* Optional but recommended: a CUDA-capable GPU and a matching `torch` with CUDA installed.

### Setup (local)

```bash
python -m venv venv
source venv/bin/activate     # macOS / Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

> If you have a GPU, install `torch` with CUDA support per PyTorch official instructions. If not, install `torch` for CPU-only.

### Run the full pipeline

```bash
python main.py
```

This will perform:

* Download BEIR dataset automatically (into `datasets/` via `beir.util.download_and_unzip`)
* Build / cache embeddings
* Build FAISS index
* Run dense retrieval, BM25, RRF fusion
* Rerank with cross-encoder
* Train & apply LTR (LightGBM)
* Evaluate and save `cache_beir/results/<dataset>_results.json`

**If you want a quick test** (recommended before full run):

* Edit `CONFIG` near the top of `main.py` and set `max_docs` and `max_queries` to small numbers (e.g., `max_docs=500`, `max_queries=50`) and `device='cpu'`. Then run `python main.py` to validate the pipeline runs end-to-end.

### Where to find results

After completion, results are stored at:

```
cache_beir/results/<dataset>_results.json
```

and per-run printed evaluation is shown on the console.

---

## 6) Reproducibility & notes for graders

* Put a short section in `README.md` titled **Reproducibility** describing which `CONFIG` fields you changed (if any), hardware used (e.g., GPU - NVIDIA GTX 1080 Ti), and a sample command used to run tests.
* Include one or two sample output `.json` files in `/results` so graders can inspect the final numbers quickly (optional, but useful).
* If you rely on GPU, note it explicitly; if grader does not have GPU, include `run_quick_test.sh` that forces `device='cpu'` and reduces sizes.

---

## 7) (Optional) Add a GitHub Actions workflow for a smoke test

Create `.github/workflows/smoke-test.yml` with a job that runs your `scripts/run_quick_test.sh` using `ubuntu-latest`. Keep the job light (small `max_docs/queries`) so it finishes quickly.

**Minimal CI idea**

* Checkout
* Set up Python
* Install `pip install -r requirements.txt`
* Run `scripts/run_quick_test.sh`

This helps graders automatically validate that the repository runs.

---

## 8) What to include in your README for the assignment submission

A grader-friendly README contains:

* Short project description (1–2 sentences)
* How to run (commands)
* How to reproduce the evaluation (where to find result files)
* Hardware & environment notes
* Any small changes you made to `main.py` for testing
* Public GitHub link (top of the README welcomes graders)

---

## 9) Troubleshooting tips (common issues)

* **faiss import errors**: Use `pip install faiss-cpu` on CPU machines. For GPU, install `faiss-gpu` that matches your CUDA.
* **CUDA / Torch mismatch**: Make sure the installed `torch` matches your CUDA version. If not using GPU, install CPU-only `torch`.
* **Out-of-memory when reranking**: reduce `rerank_k` and `rerank_batch_size` in `CONFIG`.
* **Slow encoding**: reduce `encode_batch_size` (or increase if GPU has capacity) and use `max_docs/max_queries` for testing.
* **Large caches**: add `cache_beir/` to `.gitignore` (don’t commit large .npy or index files).

---

## 10) Final Submission (what to paste into the assignment portal)

1. Public GitHub repository URL (must be accessible without authorization)
2. Short note: which command to run for a smoke test.
3. Optional: link to a sample `results/<dataset>_results.json` file in the repo.

---

## Need help automating these repo files?

If you want, I can:

* generate a ready-to-commit `requirements.txt`, `.gitignore`, `scripts/run_quick_test.sh`, and an example GitHub Actions workflow file.
* or generate a polished `README.md` (this file already does that) and prepare a `LICENSE`.

If you'd like me to create any of those files for you now, tell me which ones and I will generate their contents.
