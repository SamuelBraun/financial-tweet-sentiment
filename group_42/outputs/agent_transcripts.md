# LangChain Agent — Example Transcripts

These transcripts demonstrate the agentic workflow built in `src/agent.py`.
The agent uses tool-calling to route tweets to the best classifier, run ensemble
predictions, compare models, and explain predictions.

---

## Transcript 1: Intelligent Model Routing

```
You: Classify this tweet: "$AAPL just hit all-time high, massive breakout above resistance"

Agent: I'll classify this tweet for you. Let me first route it to the best model.

[Tool Call: route_model]
  Input: "$AAPL just hit all-time high, massive breakout above resistance"
  Output: "finbert — Tweet contains $TICKER cashtag — using finance-tuned model"

[Tool Call: classify_tweet]
  Input: tweet="$AAPL just hit all-time high...", model="finbert"
  Output: {
    "prediction": 1,
    "label": "Bullish",
    "model_used": "finbert",
    "confidence": {"Bearish": "0.032", "Bullish": "0.891", "Neutral": "0.077"}
  }

The tweet "$AAPL just hit all-time high, massive breakout above resistance" is
classified as **Bullish** with 89.1% confidence.

I routed this to the FinBERT model because the tweet contains a $TICKER cashtag
($AAPL), which indicates financial-specific content where FinBERT's domain
expertise is most valuable. The strong bullish language ("all-time high",
"massive breakout", "above resistance") aligns with the high confidence score.
```

---

## Transcript 2: Ensemble Classification with Model Disagreement

```
You: I need a very confident classification for: "Markets mixed today as Fed
signals possible rate changes amid inflation concerns"

Agent: For high-confidence classifications, I'll run all available models and
use majority voting.

[Tool Call: ensemble_classify]
  Input: "Markets mixed today as Fed signals possible rate changes amid
          inflation concerns"
  Output: {
    "individual_predictions": {
      "tfidf_logreg": "Neutral",
      "distilbert": "Bearish",
      "finbert": "Neutral",
      "twitter_roberta": "Neutral"
    },
    "ensemble_prediction": 2,
    "ensemble_label": "Neutral",
    "agreement": 0.75
  }

The ensemble classifies this as **Neutral** with 75% model agreement (3 out of
4 models agree).

Interestingly, the DistilBERT model predicted Bearish — this makes sense because
DistilBERT is a general-purpose model that may pick up on negative-sounding words
like "concerns" and "inflation" without understanding the financial nuance that
"mixed markets" and "possible rate changes" are inherently neutral hedging language.

The domain-specialized models (FinBERT and Twitter-RoBERTa) both correctly
identified this as Neutral, demonstrating why domain alignment matters.
```

---

## Transcript 3: Prediction Explanation with Feature Attribution

```
You: Why was "Sell everything, this stock is going to crash hard" classified
as Bearish? Show me the evidence.

Agent: I'll explain the prediction using feature attribution from the TF-IDF +
Logistic Regression model, which allows coefficient-based explanations.

[Tool Call: explain_prediction]
  Input: tweet="Sell everything, this stock is going to crash hard",
         model="tfidf_logreg"
  Output: {
    "prediction": "Bearish",
    "top_features": {
      "Bearish": [
        ("sell", 2.847),
        ("crash", 2.631),
        ("hard", 0.892),
        ("going", 0.234)
      ],
      "Bullish": [
        ("sell", -1.923),
        ("crash", -2.104),
        ("going", 0.156),
        ("stock", 0.089)
      ],
      "Neutral": [
        ("stock", 0.445),
        ("going", 0.312),
        ("sell", -0.924),
        ("crash", -0.527)
      ]
    }
  }

The tweet was classified as **Bearish**. Here's why:

**Strongest Bearish signals:**
- "sell" (coefficient: +2.85) — the single strongest bearish indicator in the
  vocabulary. The model learned that imperative selling language is highly
  predictive of bearish sentiment.
- "crash" (coefficient: +2.63) — another strong bearish signal. Words describing
  price collapse are among the most discriminative features.

**Cross-class evidence:**
- Both "sell" and "crash" have strong *negative* coefficients for the Bullish
  class (-1.92 and -2.10), meaning these words actively push the prediction
  *away* from Bullish — they are anti-bullish.
- The Neutral class is mildly negative on these words but less strongly, since
  neutral tweets occasionally discuss selling in a matter-of-fact way.

This is a clear-cut case where the model's decision aligns with human intuition.
The challenging cases are tweets with mixed signals — e.g., "Sold $AAPL too early,
it's going to the moon" where "sold" suggests bearish but the overall sentiment
is actually bullish regret.
```
