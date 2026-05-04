# Group 42 — Financial Tweet Sentiment Classification

**Course**: Text Mining, Spring 2025/2026 — NOVA IMS

## Team Members

| Name | Student ID | Sections |
|------|-----------|----------|
| TBD  | TBD       | TBD      |
| TBD  | TBD       | TBD      |
| TBD  | TBD       | TBD      |
| TBD  | TBD       | TBD      |

## Task

Classify financial tweets into three sentiment classes:
- **0 — Bearish**: negative market sentiment
- **1 — Bullish**: positive market sentiment
- **2 — Neutral**: no clear directional sentiment

## Repository Structure

```
group_42/
├── notebooks/
│   ├── tm_tests_42.ipynb    # Full experimentation notebook
│   └── tm_final_42.ipynb    # Final pipeline (runs end-to-end < 20 min)
├── src/                     # Reusable modules
│   ├── preprocessing.py     # Text cleaning & normalization
│   ├── features.py          # Feature extraction (BoW, W2V, Transformers)
│   ├── models.py            # Classifier factories
│   ├── evaluation.py        # Metrics & visualization
│   └── agent.py             # LangChain agentic workflow (extra credit)
├── outputs/
│   └── pred_42.csv          # Final test predictions
├── figures/                 # Saved plots (300 dpi PNGs)
├── models/                  # Cached models & embeddings (gitignored)
└── report/
    └── report_42.pdf        # Final report (≤ 15 pages)
```

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install nbstripout (prevents notebook merge conflicts)
pip install nbstripout
nbstripout --install

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger')"

# 5. Place data files
# Copy train.csv and test.csv into data/
```

## Running

### Experimentation notebook
```bash
jupyter notebook notebooks/tm_tests_42.ipynb
```

### Final pipeline (produces predictions)
```bash
jupyter notebook notebooks/tm_final_42.ipynb
# Run all cells — completes in < 20 minutes on CPU
# Output: outputs/pred_42.csv
```

### Agentic workflow (extra credit)
```bash
# Set up .env with API keys first (see .env.example)
python -m src.agent
```

## Notes

- **CPU-only**: All models are designed to run on CPU laptops
- **Reproducibility**: `random_state=42` is used everywhere
- **Cached embeddings**: Transformer embeddings are cached to `models/*.npy` — first run takes ~30 min, subsequent runs use cache
- Group number: 42
