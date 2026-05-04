# Report Outline — Group 42
## Financial Tweet Sentiment Classification

**Target**: ≤ 15 pages, PDF format.

---

## 1. Data Exploration (~2 pages)
- **Dataset overview**: 9,938 financial tweets, 3 classes (Bearish/Bullish/Neutral)
- **Class distribution**: bar chart + imbalance ratios → motivates `class_weight='balanced'` and stratified splits
- **Tweet length analysis**: character and token distributions by class (boxplot/violin) → informs max_length=128 choice for Transformers
- **Domain-specific features**: prevalence of $TICKER cashtags, @mentions, URLs, hashtags, emojis → justifies targeted preprocessing (e.g., replacing tickers with `<TICKER>` placeholder)
- **Top tokens per class**: bar charts showing discriminative unigrams → early signal on feature importance
- **Word clouds**: visual per-class vocabulary differences
- **Vocabulary statistics**: total tokens, unique tokens, type-token ratio, OOV rate
- **Duplicates check**: exact and near-duplicates → cleaned if significant
- **Label-leak verification**: confirmed that text never contains literal label

## 2. Data Preprocessing (~2 pages)
- **Pipeline design**: composable functions in `src/preprocessing.py` for reproducibility
- **Techniques implemented** (≥4):
  1. Regex-based cleaning (URLs, @mentions, hashtags, $TICKERS → `<TICKER>`, numbers → `<NUM>`/`<PCT>`, emojis → text via demojize)
  2. Lowercasing + NFKC Unicode normalization
  3. Tokenization comparison: TweetTokenizer (default) vs whitespace — justify TweetTokenizer for tweet-specific handling (emoticons, contractions)
  4. Financial-aware stopword removal — explain WHY negations (no, not, nor) and directional words (up, down) are kept
  5. Stemming (Porter) vs Lemmatization (WordNet) — compared downstream
- **Named pipelines**:
  - `pp_minimal`: light cleaning, preserves structure for BoW/W2V
  - `pp_aggressive`: full normalization with lemmatization, for classical ML
  - `pp_transformer`: minimal for pretrained models (they handle tokenization)
- **Benchmark**: both pipelines tested with same baseline classifier (LogReg + TF-IDF) to justify chosen default with numbers
- **Decision table**: which pipeline feeds which feature extractor, and why

## 3. Feature Engineering (~3 pages)
### 3a. Bag-of-Words (mandatory)
- CountVectorizer (unigrams) as baseline
- TfidfVectorizer with bigrams (`ngram_range=(1,2)`, `sublinear_tf=True`)
- Vocabulary size report and top features by class (LogReg coefficients)

### 3b. Word2Vec (mandatory)
- Custom Word2Vec (skip-gram, 100d, trained on corpus)
- Pretrained GloVe Twitter 100d (domain-aligned)
- Aggregation: mean pooling (baseline) vs TF-IDF-weighted mean pooling (variation)
- Comparison table of W2V variants

### 3c. Transformer Encoders (mandatory + EXTRA WORK)
- **Mandatory**: DistilBERT (distilbert-base-uncased) — frozen encoder, mean-pooled embeddings
- **EXTRA WORK**: FinBERT (ProsusAI/finbert) — finance-domain encoder
- **EXTRA WORK**: Twitter-RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest) — tweet-domain encoder
- Strategy: frozen feature extraction → sklearn classifier head (CPU-feasible)
- Caching strategy for embeddings (`.npy` files)

## 4. Classification Models (~3 pages)
### 4a. Traditional ML (mandatory, ≥2 variations)
- Logistic Regression (`class_weight='balanced'`, C tuned via grid search [0.01, 0.1, 1, 10])
- Linear SVM (LinearSVC, balanced)
- XGBoost (300 estimators, multi:softprob)
- Random Forest (balanced, 300 estimators)
- All evaluated via 5-fold StratifiedKFold CV → mean ± std macro-F1

### 4b. Transformer Fine-tuning (mandatory, ≥2 variations)
- DistilBERT + linear head (encoder frozen)
- Second checkpoint (FinBERT or Twitter-RoBERTa) + linear head
- Fallback documented: if fine-tuning too slow → frozen embeddings + LogReg head

### 4c. EXTRA WORK — Decoder Model
- Flan-T5-small (local) or API-based (Claude/GPT-4o-mini)
- Few-shot prompt: 2 examples per class
- Deterministic output parsing
- Comparison with encoder-based approaches

## 5. Evaluation and Results (~3 pages)
- **Results matrix**: rows = (preprocessing × features), cols = classifiers, cells = macro-F1 ± std
- **Per-model analysis**:
  - Accuracy, Precision (macro + per-class), Recall (macro + per-class), F1 (macro + per-class)
  - Confusion matrices for top models
- **Interpretive analysis**:
  - Which class is hardest (likely Neutral — catch-all class)
  - Bearish/Bullish confusion patterns and investor implications
  - Impact of class weighting on dominant class bias
  - Justification of macro-F1 as primary metric
- **Best pipeline selection**: chosen based on validation macro-F1 + cross-fold consistency

## 6. EXTRA WORK — Agentic Workflow (~1 page)
- Architecture: LangChain agent with tool-calling capabilities
- Tools: model routing, ensemble classification, evaluation comparison, explainability
- 3 example conversation transcripts showing non-trivial decisions
- Agent trace logs (from `outputs/agent_trace.jsonl`)

## 7. Conclusion & Limitations (~0.5 page)
- Summary of best approach and why it works
- Key findings from the systematic comparison
- Limitations: CPU-only constraint, dataset size, class imbalance
- Future work: full fine-tuning with GPU, ensemble methods, temporal analysis

---

**Page budget**: 2 + 2 + 3 + 3 + 3 + 1 + 0.5 = 14.5 pages ✓ (within 15-page limit)
