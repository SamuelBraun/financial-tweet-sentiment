# Financial Tweet Sentiment Classification

Three-class sentiment classification on financial tweets (Bearish / Bullish / Neutral),
built as a systematic comparison rather than a single model. Course project for Text
Mining, M.Sc. Data Science and Advanced Analytics, NOVA IMS, 2025/2026.

**32 feature × classifier combinations** were benchmarked under 5-fold stratified
cross-validation, spanning classical bag-of-words through static embeddings to frozen
transformer encoders.

**Best result: 0.794 macro F1** — Twitter-RoBERTa embeddings with XGBoost.

## Results

| Representation | Classifier | Macro F1 |
|---|---|---|
| **Twitter-RoBERTa (frozen)** | **XGBoost** | **0.794** |
| FinBERT (frozen) | XGBoost | 0.779 |
| TF-IDF | SVM | 0.740 |
| Bag-of-words | Logistic regression | 0.720 |
| DistilBERT (frozen) | SVM | 0.705 |
| Word2Vec mean-pool | XGBoost | 0.658 |
| Twitter-RoBERTa zero-shot head | none | 0.587 |
| Flan-T5-large few-shot | none | 0.485 |

Two findings worth the comparison: a domain-matched encoder (Twitter-RoBERTa, trained
on tweets) beats a domain-matched-by-topic one (FinBERT, trained on financial text),
and both zero-shot and few-shot decoding fall well behind a trained classifier on frozen
embeddings. Full ranking, per-class analysis and significance tests are in
[report/report_42.pdf](report/report_42.pdf).

## Layout

```
notebooks/tm_tests_42.ipynb    Full experimentation across all combinations
notebooks/tm_final_42.ipynb    Final pipeline → outputs/pred_42.csv
src/preprocessing.py           Composable cleaning steps + 3 named pipelines
src/features.py                BoW, TF-IDF, Word2Vec and transformer extractors
src/models.py                  Classifier factories + few-shot decoder
src/evaluation.py              Metrics, cross-validation, plots
src/agent.py                   LangChain agent layer (additional work)
outputs/                       Result matrices, predictions, agent transcripts
figures/                       300 dpi figures used in the report
report/                        Report (PDF + LaTeX source)
```

The labelled and held-out CSVs are course data and are **not** included here; place
`train.csv` and `test.csv` under `data/` to run the notebooks.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['stopwords','wordnet','punkt','punkt_tab']]"
# place train.csv and test.csv into data/
jupyter notebook notebooks/tm_final_42.ipynb    # end-to-end in under 20 min on CPU
```

`random_state=42` throughout (splits, cross-validation, classifiers, embeddings).
Transformer embeddings are cached to `models/*.npy` on first run. Decoder cells read
pre-computed predictions from `outputs/decoder_cache_*.json`; delete the cache to
regenerate.

The agent layer needs an API key — copy `.env.example` to `.env` and fill it in.

## Team

Group project (4 people): Jan Luis Thier, Lukas Belser, Margarida Estrada Quintino,
Samuel Braun.
