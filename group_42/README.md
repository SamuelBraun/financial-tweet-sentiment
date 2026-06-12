# Group 42 — Financial Tweet Sentiment Classification

Text Mining, Spring 2025/2026 — NOVA IMS

3-class sentiment classification on financial tweets (Bearish / Bullish / Neutral)
with a systematic comparison of 32 feature × classifier combinations under 5-fold
stratified cross-validation. Best pipeline reaches **0.794 macro-F1**.

## Headline results

| Method | Macro-F1 |
|---|---|
| Bag-of-Words + LogReg | 0.720 |
| TF-IDF + SVM | 0.740 |
| Word2Vec mean-pool + XGBoost | 0.658 |
| DistilBERT (frozen) + SVM | 0.705 |
| FinBERT (frozen) + XGBoost | 0.779 |
| **Twitter-RoBERTa (frozen) + XGBoost** | **0.794** |
| Zero-shot Twitter-RoBERTa head (no training) | 0.587 |
| Flan-T5-large few-shot (no training) | 0.485 |

Full ranking, per-class analysis and statistical-significance tests in
[report/report_42.pdf](report/report_42.pdf).

## Layout

```
group_42/
├── notebooks/
│   ├── tm_tests_42.ipynb         # Full experimentation
│   └── tm_final_42.ipynb         # Final pipeline → pred_42.csv (<20 min on CPU)
├── src/
│   ├── preprocessing.py          # Composable cleaning + 3 named pipelines
│   ├── features.py               # BoW, TF-IDF, Word2Vec, Transformer extractors
│   ├── models.py                 # Classifier factories + few-shot decoder
│   ├── evaluation.py             # Metrics, CV, plots
│   └── agent.py                  # LangChain agent (extra work)
├── data/                         # train.csv, test.csv (gitignored, synced out-of-band)
├── outputs/
│   ├── pred_42.csv               # Final predictions
│   ├── final_results_*.csv       # Results matrix
│   ├── agent_transcripts.md      # Agent traces
│   └── decoder_cache_*.json      # Pre-computed Flan-T5 predictions
├── figures/                      # 300 dpi PNGs used by the report
├── models/                       # Cached embeddings (gitignored)
└── report/
    ├── report_42.tex
    └── report_42.pdf             # 15 pages
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install nbstripout && nbstripout --install
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['stopwords','wordnet','punkt','punkt_tab']]"
# Place train.csv and test.csv into data/
```

## Run

```bash
jupyter notebook notebooks/tm_tests_42.ipynb     # full experimentation (~15 min on MPS)
jupyter notebook notebooks/tm_final_42.ipynb     # produces outputs/pred_42.csv (<20 min on CPU)
```

The decoder cells use cached predictions in `outputs/decoder_cache_*.json`, so they
finish in seconds. To regenerate from scratch, delete the cache and rerun — first run
takes ~35 min on MPS for Flan-T5-base.

## Reproducibility

- `random_state=42` everywhere (splits, CV, classifiers, embeddings).
- All Transformer embeddings cached to `models/*.npy` on first run.
- Final pipeline (`tm_final_42.ipynb`) runs end-to-end with **Restart & Run All** in
  under 20 minutes on a CPU laptop.

## Team

| Name | Student ID |
|------|-----------|
| Jan Luis Thier               | 20250352 |
| Samuel Braun                 | 20250355 |
| Lukas Belser                 | 20250338 |
| Margarida Estrada Quintino   | 20250411 |
