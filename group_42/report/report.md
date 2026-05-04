# Financial Tweet Sentiment Classification

**Group 42 — Text Mining 2025/2026, NOVA IMS**

---

## 1. Introduction

This report presents a systematic approach to classifying financial tweets as **Bearish** (0), **Bullish** (1), or **Neutral** (2). Financial sentiment analysis on social media has practical applications in algorithmic trading, market surveillance, and investor sentiment tracking. The challenge lies in the brevity and informality of tweets, domain-specific vocabulary (cashtags, financial jargon), and inherent class imbalance where neutral commentary dominates.

We evaluate **32 combinations** of feature extraction methods and classifiers using 5-fold stratified cross-validation, with macro-F1 as the primary metric. Our best pipeline — **Twitter-RoBERTa embeddings + XGBoost** — achieves a macro-F1 of **0.794**.

---

## 2. Data Exploration

### 2.1 Dataset Overview

The dataset consists of **9,543 training tweets** and **2,388 test tweets**, each labeled with one of three sentiment classes.

| Class | Count | Proportion |
|-------|-------|------------|
| Bearish (0) | 1,442 | 15.1% |
| Bullish (1) | 1,923 | 20.1% |
| Neutral (2) | 6,178 | 64.7% |

**Table 1**: Training set class distribution. The dataset exhibits significant class imbalance — Neutral tweets outnumber Bearish tweets by a factor of 4.3x.

The imbalance motivates two key design decisions: (1) using `class_weight='balanced'` in all classifiers to prevent majority-class bias, and (2) adopting macro-F1 as the primary metric rather than accuracy, since a naive all-Neutral classifier would achieve 64.7% accuracy while failing completely on minority classes.

*See Figure 1 (class_distribution.png) for the visual distribution.*

### 2.2 Tweet Characteristics

Tweet lengths range from 2 to 48 tokens (median: 18 tokens), well within Twitter's constraints. All three classes show similar length distributions, indicating that length alone is not a discriminative feature. This also justifies using `max_length=128` for Transformer tokenization — no tweet approaches this limit.

*See Figure 2 (tweet_length_distribution.png).*

### 2.3 Domain-Specific Features

Financial tweets contain distinctive markers:

- **$TICKER cashtags** (e.g., $AAPL, $TSLA): Present in ~60% of tweets across all classes. These carry contextual signal but overfitting to specific tickers would not generalize, so we replace them with a `<TICKER>` placeholder.
- **@mentions**: Present in ~15% of tweets; removed during preprocessing as they carry no sentiment signal.
- **URLs**: Present in ~20% of tweets; removed as they point to external content unavailable to the classifier.
- **Hashtags**: The `#` symbol is stripped but the word is retained (e.g., `#bullish` becomes `bullish`), preserving the semantic content.

*See Figure 5 (domain_features.png).*

### 2.4 Discriminative Vocabulary

Analysis of top tokens per class reveals clear thematic separation:

- **Bearish**: "cut", "downgrade", "sell", "loss", "short", "crash", "bear"
- **Bullish**: "buy", "upgrade", "breakout", "moon", "bull", "long", "growth"
- **Neutral**: "earnings", "report", "market", "stock", "today", "trade"

The Neutral class vocabulary overlaps significantly with both Bearish and Bullish, confirming that Neutral is the hardest class to discriminate — it acts as a catch-all for informational tweets without explicit sentiment.

*See Figures 3-4 (top_tokens_per_class.png, wordclouds_per_class.png).*

### 2.5 Data Quality

- **Exact duplicates**: A small number exist (<1%) and are retained, as removing them could bias the class distribution.
- **Label leakage check**: Words like "bearish" and "bullish" appear naturally in financial discourse — they are domain vocabulary, not data leakage. A model that learns to associate these words with their respective classes is learning a legitimate pattern.

---

## 3. Data Preprocessing

### 3.1 Design Philosophy

We implemented **composable preprocessing functions** in `src/preprocessing.py`, allowing us to mix and match cleaning steps into named pipelines. This modular design ensures reproducibility and enables systematic comparison.

### 3.2 Preprocessing Functions

We implement the following cleaning steps (each as a standalone function):

1. **HTML unescape** — Decode `&amp;`, `&lt;`, etc.
2. **URL removal** — Strip `http://...` and `www.` links.
3. **Mention removal** — Strip `@username` handles.
4. **Hashtag handling** — Strip `#` but keep the word (e.g., `#bullish` → `bullish`). An alternative "drop" mode was tested but performed worse as it discards potentially informative hashtag words.
5. **Ticker normalization** — Replace `$AAPL`, `$TSLA`, etc. with `<TICKER>`. This preserves the signal that a ticker was mentioned without overfitting to specific stock names.
6. **Number normalization** — Replace percentages with `<PCT>` and other numbers with `<NUM>`, preserving the signal that numerical claims exist.
7. **Emoji demojization** — Convert emojis to textual descriptions (e.g., `🚀` → `rocket`) using the `emoji` library, preserving sentiment signal.
8. **Unicode normalization** — NFKC normalization for consistent character encoding.
9. **Lowercasing** — Standard case folding.
10. **Punctuation collapsing** — `!!!` → `!`, whitespace normalization.
11. **Financial-aware stopword removal** — We customize the NLTK English stopword list by **keeping** negations (`no`, `not`, `nor`, `against`) and directional words (`up`, `down`, `over`, `under`, `more`, `less`). These words are critical for financial sentiment — "not going up" means the opposite of "going up".
12. **Lemmatization** (WordNet) and **Stemming** (Porter) — Tested as optional token-level operations.

### 3.3 Named Pipelines

| Pipeline | Steps | Use Case |
|----------|-------|----------|
| `pp_minimal` | Steps 1-10 | BoW, TF-IDF, Word2Vec |
| `pp_aggressive` | Steps 1-12 (+ stopwords, lemma) | Tested, dropped |
| `pp_transformer` | Steps 1-7, 10 (no lowercase/normalize) | Transformer inputs |

### 3.4 Pipeline Comparison

We benchmarked `pp_minimal` vs. `pp_aggressive` using the same classifier (LogReg + TF-IDF) to isolate the effect of preprocessing:

| Pipeline | Macro-F1 |
|----------|----------|
| `pp_minimal` | **0.736** |
| `pp_aggressive` (lemma) | 0.718 |
| `pp_aggressive` (stem) | 0.709 |

**Result**: `pp_minimal` consistently outperforms `pp_aggressive`. The aggressive pipeline's stopword removal and lemmatization destroy discriminative signal. Specifically:
- Removing stopwords drops negations like "not" that flip sentiment.
- Lemmatizing collapses temporal distinctions ("falling" → "fall") that carry urgency information.
- Number normalization (`<NUM>`) removes specific price/percentage values that provide context.

**Decision**: `pp_minimal` is used for all traditional ML experiments. `pp_transformer` is used for Transformer feature extraction, where pretrained tokenizers handle subword splitting and the model benefits from seeing near-original text.

---

## 4. Feature Engineering

We implement and compare **eight feature extraction methods** across three families.

### 4.1 Bag-of-Words Features (Mandatory)

**CountVectorizer (BoW)**: Unigram counts with a vocabulary cap of 10,000 features. Serves as the simplest baseline.

**TF-IDF**: Our optimized configuration uses:
- Bigrams (`ngram_range=(1,2)`) to capture phrases like "going up" and "sell off"
- Sublinear TF scaling (`sublinear_tf=True`) to dampen the impact of very frequent terms
- Maximum 20,000 features with `min_df=2` (appear in at least 2 documents) and `max_df=0.95` (appear in at most 95% of documents)

TF-IDF with bigrams outperforms BoW by 1-3 percentage points across all classifiers, confirming that word order carries sentiment signal even in a bag-of-words framework.

*See Figure 7 (top_tfidf_features.png) for the top discriminative features per class.*

### 4.2 Word Embeddings (Mandatory)

**Custom Word2Vec**: Skip-gram model trained on the financial tweet corpus (100 dimensions, window=5, min_count=2, 20 epochs). Document representations are computed via mean pooling of token embeddings.

**Pretrained GloVe Twitter 100d**: The `glove-twitter-100` embeddings from the Gensim library, pretrained on 2 billion tweets. These provide general Twitter vocabulary coverage but lack financial domain specificity.

**TF-IDF-Weighted Pooling**: As a variation on mean pooling, we weight each token's embedding by its TF-IDF score before averaging. This gives more importance to rare, discriminative words and less to common ones.

**Results**: All Word2Vec variants perform significantly worse than BoW/TF-IDF (macro-F1: 0.59-0.66 vs. 0.72-0.74). The primary reason is that **mean pooling loses word order information**, which is critical for sentiment. "Stock going up" and "stock going down" would produce similar mean vectors despite opposite sentiments. Additionally, the 100-dimensional embeddings cannot capture the same granularity as 20,000-dimensional TF-IDF vectors for this task.

### 4.3 Transformer Encoders (Mandatory + Extra Work)

**Strategy**: We use pretrained Transformer encoders as **frozen feature extractors**. For each tweet, we:
1. Tokenize with the model's pretrained tokenizer
2. Pass through the frozen encoder
3. Mean-pool the last hidden states (768-dimensional) with attention mask
4. Use the resulting vector as input to a downstream sklearn classifier

This is CPU-feasible because we extract embeddings once and cache them as `.npy` files, avoiding repeated forward passes.

**Models evaluated**:

| Encoder | Source | Domain | Params |
|---------|--------|--------|--------|
| DistilBERT | `distilbert-base-uncased` | General | 66M |
| FinBERT | `ProsusAI/finbert` | Finance (SEC filings) | 110M |
| Twitter-RoBERTa | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Twitter + sentiment | 125M |

FinBERT and Twitter-RoBERTa are **extra work** beyond the mandatory DistilBERT.

**Key finding**: Domain alignment is more important than model size. DistilBERT (general-purpose, 66M params) underperforms TF-IDF, while Twitter-RoBERTa (tweet-domain, sentiment-tuned, 125M params) achieves the best results overall. FinBERT falls between — it understands financial language but was trained on formal SEC filings, not informal tweets.

---

## 5. Classification Models

### 5.1 Classifiers Evaluated

| Classifier | Key Hyperparameters | Rationale |
|-----------|-------------------|-----------|
| Logistic Regression | `class_weight='balanced'`, C=1.0, multinomial | Strong linear baseline, interpretable coefficients |
| LinearSVC | `class_weight='balanced'`, C=1.0 | Maximum-margin classifier, good for high-dimensional sparse data |
| XGBoost | 300 estimators, max_depth=6, softprob | Gradient boosting for non-linear decision boundaries |
| Random Forest | 300 estimators, `class_weight='balanced'` | Ensemble baseline, robust to noise |

All classifiers use `class_weight='balanced'` (or equivalent) to handle class imbalance by upweighting minority classes.

### 5.2 Evaluation Protocol

- **5-fold Stratified K-Fold Cross-Validation** preserving class proportions in each fold
- **Primary metric**: Macro-F1 (equally weights all three classes)
- **Reproducibility**: `random_state=42` for all stochastic operations

---

## 6. Results and Analysis

### 6.1 Full Results Matrix

**Table 2**: Macro-F1 scores (mean ± std) across all 32 feature × classifier combinations, evaluated via 5-fold stratified CV.

| Features | LogReg | SVM | XGBoost | RandomForest |
|----------|--------|-----|---------|-------------|
| BoW | 0.720±0.011 | 0.708±0.010 | 0.691±0.013 | 0.664±0.012 |
| TF-IDF | 0.736±0.006 | **0.740±0.013** | 0.670±0.011 | 0.654±0.014 |
| W2V-mean | 0.600±0.013 | 0.603±0.012 | 0.650±0.006 | 0.595±0.011 |
| W2V-tfidf | 0.595±0.013 | 0.610±0.015 | 0.658±0.016 | 0.611±0.011 |
| GloVe-mean | 0.599±0.017 | 0.617±0.014 | 0.636±0.009 | 0.494±0.004 |
| DistilBERT | 0.702±0.009 | 0.705±0.008 | 0.674±0.012 | 0.517±0.007 |
| FinBERT* | 0.736±0.007 | 0.751±0.009 | 0.779±0.003 | 0.734±0.002 |
| Twitter-RoBERTa* | 0.755±0.011 | 0.764±0.008 | **0.794±0.006** | 0.742±0.008 |

*\* Extra work*

### 6.2 Key Findings

**Feature hierarchy** (most to least effective):
1. **Twitter-RoBERTa** (best: 0.794) — Pretrained on 58M tweets with sentiment labels; perfect domain alignment for our task.
2. **FinBERT** (best: 0.779) — Financial domain knowledge helps, but was trained on formal SEC filings rather than informal tweets.
3. **TF-IDF** (best: 0.740) — Surprisingly competitive. Bigram TF-IDF captures local word order patterns that discriminate sentiment.
4. **BoW** (best: 0.720) — Solid baseline, but loses the bigram advantage.
5. **DistilBERT** (best: 0.705) — General-purpose encoder *underperforms* TF-IDF. This demonstrates that domain alignment matters more than model sophistication.
6. **Word2Vec variants** (best: 0.658) — Mean pooling destroys word order, critical for sentiment.

**Classifier patterns**:
- **XGBoost excels with dense features** (Transformer embeddings): It can learn non-linear decision boundaries in the 768-dimensional embedding space. However, XGBoost struggles with sparse features (BoW/TF-IDF) where it must split on individual token presence/absence.
- **Linear models (LogReg, SVM) excel with sparse features** (BoW/TF-IDF): Regularization handles high-dimensional sparse inputs well. The linear decision boundary is sufficient when features are already semantically meaningful (TF-IDF weights).
- **Random Forest consistently underperforms**: It struggles with both sparse high-dimensional features (too many random splits) and dense features (less effective than boosting for learning complex boundaries).

**Stability**: All standard deviations are below 0.017, indicating stable, reproducible results across folds. The best model (Twitter-RoBERTa + XGBoost) has one of the lowest standard deviations (±0.006), suggesting consistent performance regardless of the specific data split.

### 6.3 Per-Class Analysis

Analysis of per-class F1 scores for the top models reveals:

- **Neutral** is consistently the easiest class (F1 > 0.87) due to its large support and distinctive vocabulary of factual/informational language.
- **Bearish** and **Bullish** are harder (F1: 0.65-0.75) because they share domain vocabulary and differ primarily in sentiment polarity, which requires understanding context.
- **Bearish** is slightly harder than Bullish across most models — bearish tweets sometimes use hedging language ("might drop", "could fall") that resembles neutral commentary.

*See Figure (per_class_f1_top5.png).*

### 6.4 Error Analysis

Examining misclassified tweets from the best model reveals common failure patterns:

1. **Neutral ↔ Bearish confusion**: Tweets reporting negative news factually (e.g., "Stock down 5% after earnings miss") are genuinely ambiguous — the factual reporting is Neutral but the content is inherently negative.

2. **Neutral ↔ Bullish confusion**: Similarly, factual reports of positive events (e.g., "$AAPL reports record revenue") blur the line between informational and positive sentiment.

3. **Bearish ↔ Bullish confusion** (rarest): Occurs primarily with sarcasm or contrarian positions (e.g., "Great, another day of losses" — sarcastic bearish classified as bullish due to "great").

These patterns suggest that the primary challenge is the **fundamental ambiguity between factual reporting and sentiment expression** in financial discourse.

### 6.5 Learning Curve

The learning curve (Figure: learning_curve.png) shows model performance as a function of training set size for Twitter-RoBERTa + XGBoost. Validation F1 grows rapidly from 10% to 40% of the data, then plateaus, suggesting the model is approaching its ceiling for this feature representation. The persistent train-validation gap indicates mild overfitting that more data alone may not resolve — architectural improvements (e.g., full fine-tuning) would be needed.

### 6.6 Statistical Significance

Paired t-tests on the 5-fold CV scores confirm that the performance differences between top models are statistically meaningful:

- Twitter-RoBERTa + XGBoost vs. FinBERT + XGBoost: Significant improvement (p < 0.05)
- Twitter-RoBERTa + XGBoost vs. TF-IDF + SVM: Significant improvement (p < 0.05)
- FinBERT + XGBoost vs. TF-IDF + SVM: Significant improvement (p < 0.05)

This validates that our model ranking is not an artifact of random variation.

### 6.7 Confusion Matrices

*See Figures 8-11 (cm_twitter_roberta.png, cm_finbert.png, cm_distilbert.png, cm_best_traditional.png).*

The confusion matrices visually confirm the per-class analysis: the dominant error mode is confusion between Neutral and the sentiment classes (Bearish/Bullish), while direct Bearish ↔ Bullish misclassification is rare.

---

## 7. Extra Work

### 7.1 Domain-Specific Transformer Encoders (+1.0 pts)

We evaluate two additional pretrained encoders beyond the mandatory DistilBERT:

- **FinBERT** (`ProsusAI/finbert`): Trained on financial communications (SEC filings, analyst reports). Achieves 0.779 macro-F1, a 7.4 percentage point improvement over DistilBERT, demonstrating the value of financial domain pretraining.

- **Twitter-RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`): Trained on 58 million tweets with sentiment labels. Achieves 0.794 macro-F1, the best result overall. The dual domain alignment (Twitter + sentiment) makes this the ideal feature extractor for our task.

### 7.2 Agentic Workflow (+Extra)

We implement a **LangChain-based agent** (`src/agent.py`) that orchestrates the classification pipeline with five tools:

1. **classify_tweet** — Classify a single tweet with a specified model
2. **route_model** — Intelligent model routing based on tweet characteristics (presence of cashtags → FinBERT, short tweets with emojis → Twitter-RoBERTa)
3. **ensemble_classify** — Run all models and return majority-vote prediction with agreement score
4. **compare_models** — Display formatted comparison of all model metrics
5. **explain_prediction** — Show top contributing features for a prediction (coefficient-based, works with linear models)

The agent demonstrates non-trivial decision-making: it inspects tweet content to choose the most appropriate model, reconciles disagreements via ensemble voting, and provides human-interpretable explanations. Example transcripts are provided in `outputs/agent_transcripts.md`.

### 7.3 Decoder Model (Attempted)

We implemented few-shot classification using **Flan-T5-small** (`src/models.py: decoder_classify_batch`). The prompt provides 2 examples per class and asks the model to output a single digit (0, 1, or 2). However, CPU inference proved infeasible: ~2-5 seconds per tweet × 2,388 test tweets would require over 3 hours. This is documented as a limitation — GPU access would make this viable.

---

## 8. Final Pipeline and Predictions

The production pipeline (`tm_final_42.ipynb`) trains the best model on the **full training set** (9,543 tweets) and generates predictions for all 2,388 test tweets:

1. Load and preprocess all data with `pp_transformer`
2. Extract Twitter-RoBERTa embeddings (cached as `.npy` files)
3. Train XGBoost on all 9,543 training embeddings
4. Predict on 2,388 test tweets → `pred_42.csv`

**Prediction distribution**: Bearish: 309 (12.9%), Bullish: 392 (16.4%), Neutral: 1,687 (70.6%) — consistent with training set proportions, suggesting the model is not over- or under-predicting any class.

---

## 9. Conclusion

### 9.1 Summary

Our systematic evaluation of 32 feature × classifier combinations reveals that **domain-specific pretrained Transformer encoders** combined with **gradient-boosted classifiers** achieve the best results for financial tweet sentiment classification. The winning pipeline — Twitter-RoBERTa + XGBoost — achieves **0.794 macro-F1**, significantly outperforming traditional approaches.

### 9.2 Key Takeaways

1. **Domain alignment > model sophistication**: A general-purpose DistilBERT (66M params) underperforms simple TF-IDF bigrams. A domain-aligned Twitter-RoBERTa (125M params) exceeds both. This demonstrates that *what* a model was trained on matters more than *how large* it is.

2. **Feature extraction strategy matters more than classifier choice**: The gap between feature families (0.60 for W2V vs. 0.79 for Twitter-RoBERTa) far exceeds the gap between classifiers within a family (typically 0.02-0.05).

3. **Preprocessing should be conservative**: Aggressive normalization (stopword removal, stemming) destroys sentiment signal. Financial text benefits from keeping negations and directional words intact.

4. **Class imbalance handling is essential**: Without balanced class weights, classifiers default to predicting the majority Neutral class, yielding high accuracy but poor minority-class recall.

### 9.3 Limitations

- **CPU-only constraint**: Full Transformer fine-tuning was infeasible; frozen embeddings + classifier head is a compromise.
- **No hyperparameter tuning**: We used default/recommended hyperparameters. Grid search could improve results by 1-3%.
- **Single dataset**: Results may not generalize to other financial text domains (e.g., earnings calls, SEC filings).
- **Decoder model infeasible**: Flan-T5 few-shot classification was too slow on CPU for evaluation.

### 9.4 Future Work

- **Full fine-tuning with GPU**: Fine-tuning Twitter-RoBERTa's encoder (not just the head) would likely improve results significantly.
- **Ensemble methods**: Combining predictions from multiple feature families could leverage complementary strengths.
- **Temporal analysis**: Financial sentiment may have temporal patterns (e.g., market hours vs. after-hours) worth exploiting.
- **Hyperparameter optimization**: Bayesian optimization (e.g., Optuna) across the full pipeline.

---

## References

- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models. *arXiv:1908.10063*.
- Barbieri, F., et al. (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. *arXiv:2010.12421*.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.

---

*Report generated for Group 42, Text Mining 2025/2026, NOVA IMS.*
*Code repository: [GitHub](https://github.com/SamuelBraun/MSDAA-Text-Mining)*
