"""
EXTRA WORK — Agentic Workflow for Financial Tweet Classification.

LangChain agent that orchestrates the classification pipeline with:
1. Model routing — chooses classifier based on tweet characteristics
2. Ensemble/comparison — runs multiple classifiers and reconciles
3. Evaluation — compares models on validation metrics
4. Explainability — shows top contributing features for a prediction

Run: python -m src.agent
"""

import os
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import joblib

# Configure logging for agent traces
TRACE_PATH = Path(__file__).resolve().parent.parent / "outputs" / "agent_trace.jsonl"


def log_trace(tool_name: str, input_data: str, output_data: str):
    """Log tool invocation to JSONL trace file."""
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "input": input_data[:500],
        "output": output_data[:500],
    }
    with open(TRACE_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ===========================================================================
# Tool implementations
# ===========================================================================

class ClassificationTools:
    """Container for trained models and tools the agent can invoke."""

    def __init__(self, models_dir: Optional[str] = None):
        from src import MODELS_DIR
        self.models_dir = Path(models_dir or MODELS_DIR)
        self._loaded_models = {}
        self._vectorizers = {}

    def _load_pipeline(self, name: str):
        """Load a cached (vectorizer, model) pair."""
        if name in self._loaded_models:
            return self._loaded_models[name]

        model_path = self.models_dir / f"pipeline_{name}.joblib"
        if not model_path.exists():
            return None

        pipeline = joblib.load(model_path)
        self._loaded_models[name] = pipeline
        return pipeline

    def classify_tweet(self, tweet: str, model_name: str = "best") -> dict:
        """Classify a single tweet with a specified model.

        Returns dict with prediction, confidence, and model used.
        """
        from src.preprocessing import pp_minimal, pp_transformer

        pipeline = self._load_pipeline(model_name)
        if pipeline is None:
            return {"error": f"Model '{model_name}' not found. "
                    f"Available: {self.list_models()}"}

        vectorizer = pipeline.get("vectorizer")
        model = pipeline["model"]
        preprocess_fn = pipeline.get("preprocess_fn", pp_minimal)

        # Preprocess
        processed = preprocess_fn(tweet)

        # Feature extraction
        if vectorizer is not None:
            features = vectorizer.transform([processed])
        else:
            # Transformer embeddings — use cached or compute
            from src.features import TransformerFeaturizer
            featurizer = pipeline.get("featurizer")
            if featurizer:
                features = featurizer.transform([processed])
            else:
                features = vectorizer.transform([processed])

        # Predict
        pred = model.predict(features)[0]
        label_map = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

        result = {
            "tweet": tweet[:100],
            "prediction": int(pred),
            "label": label_map[int(pred)],
            "model_used": model_name,
        }

        # Confidence if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            result["confidence"] = {label_map[i]: f"{p:.3f}" for i, p in enumerate(proba)}

        log_trace("classify_tweet", tweet, json.dumps(result))
        return result

    def list_models(self) -> list:
        """List available trained model pipelines."""
        models = [p.stem.replace("pipeline_", "")
                  for p in self.models_dir.glob("pipeline_*.joblib")]
        return models or ["No models found — run the training notebook first."]

    def route_model(self, tweet: str) -> str:
        """Inspect tweet characteristics and choose the best model.

        Routing logic:
        - If tweet contains $TICKER cashtags → prefer finbert pipeline
        - If tweet is short (< 50 chars) with emojis → prefer twitter-roberta
        - If tweet is long/technical → prefer distilbert
        - Default → best overall model
        """
        has_ticker = bool(re.search(r"\$[A-Za-z]{1,5}\b", tweet))
        has_emoji = bool(re.search(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]", tweet))
        is_short = len(tweet) < 50

        available = self.list_models()
        if isinstance(available, list) and "No models" in available[0]:
            return "best"

        if has_ticker and "finbert" in available:
            choice = "finbert"
            reason = "Tweet contains $TICKER cashtag — using finance-tuned model"
        elif is_short and has_emoji and "twitter_roberta" in available:
            choice = "twitter_roberta"
            reason = "Short tweet with emojis — using Twitter-specialized model"
        elif "best" in available:
            choice = "best"
            reason = "Using best overall model"
        else:
            choice = available[0]
            reason = f"Using default available model: {choice}"

        log_trace("route_model", tweet, f"{choice}: {reason}")
        return choice

    def ensemble_classify(self, tweet: str) -> dict:
        """Run all available models and reconcile via majority vote."""
        available = self.list_models()
        if isinstance(available, list) and "No models" in available[0]:
            return {"error": "No trained models available."}

        predictions = {}
        for model_name in available:
            result = self.classify_tweet(tweet, model_name)
            if "error" not in result:
                predictions[model_name] = result["prediction"]

        if not predictions:
            return {"error": "All model predictions failed."}

        # Majority vote
        votes = list(predictions.values())
        from collections import Counter
        majority = Counter(votes).most_common(1)[0][0]
        label_map = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

        result = {
            "tweet": tweet[:100],
            "individual_predictions": {k: label_map[v] for k, v in predictions.items()},
            "ensemble_prediction": int(majority),
            "ensemble_label": label_map[majority],
            "agreement": sum(1 for v in votes if v == majority) / len(votes),
        }

        log_trace("ensemble_classify", tweet, json.dumps(result))
        return result

    def compare_models(self) -> str:
        """Load and format cached evaluation results for all models."""
        results_path = self.models_dir.parent / "outputs" / "all_results.json"
        if not results_path.exists():
            return ("No evaluation results found. Run the experimentation "
                    "notebook first to generate comparison metrics.")

        with open(results_path) as f:
            results = json.load(f)

        lines = ["Model Comparison (Validation Set)", "=" * 50]
        for r in results:
            line = (f"{r.get('preprocessing', '?')} + {r.get('feature', '?')} + "
                    f"{r.get('model', '?')}: Macro-F1 = {r.get('macro_f1', 0):.4f}")
            if r.get("std"):
                line += f" ± {r['std']:.4f}"
            lines.append(line)

        output = "\n".join(lines)
        log_trace("compare_models", "comparison request", output[:500])
        return output

    def explain_prediction(self, tweet: str, model_name: str = "tfidf_logreg") -> dict:
        """Show top contributing tokens for a prediction.

        Uses Logistic Regression coefficients to identify which tokens
        pushed the prediction toward each class.
        """
        from src.preprocessing import pp_minimal

        pipeline = self._load_pipeline(model_name)
        if pipeline is None:
            return {"error": f"Model '{model_name}' not found."}

        vectorizer = pipeline.get("vectorizer")
        model = pipeline["model"]

        if not hasattr(model, "coef_"):
            return {"error": f"Model '{model_name}' does not support "
                    "coefficient-based explanation. Use a linear model."}

        processed = pp_minimal(tweet)
        features = vectorizer.transform([processed])
        pred = model.predict(features)[0]

        # Get feature contributions
        feature_names = vectorizer.get_feature_names_out()
        nonzero = features.nonzero()[1]

        label_map = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
        contributions = {}

        for cls_idx in range(model.coef_.shape[0]):
            cls_contribs = []
            for feat_idx in nonzero:
                cls_contribs.append((
                    feature_names[feat_idx],
                    float(model.coef_[cls_idx, feat_idx])
                ))
            cls_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            contributions[label_map[cls_idx]] = cls_contribs[:10]

        result = {
            "tweet": tweet[:100],
            "prediction": label_map[int(pred)],
            "top_features": contributions,
        }

        log_trace("explain_prediction", tweet, json.dumps(result, default=str))
        return result


# ===========================================================================
# LangChain Agent setup
# ===========================================================================

def create_agent():
    """Create a LangChain agent with classification tools.

    Requires OPENAI_API_KEY or ANTHROPIC_API_KEY in .env.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from langchain.tools import StructuredTool
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    tools_container = ClassificationTools()

    # Define tools
    classify_tool = StructuredTool.from_function(
        func=lambda tweet, model="auto": (
            tools_container.classify_tweet(
                tweet,
                tools_container.route_model(tweet) if model == "auto" else model
            )
        ),
        name="classify_tweet",
        description=(
            "Classify a financial tweet as Bearish (0), Bullish (1), or Neutral (2). "
            "Set model='auto' for intelligent routing, or specify a model name."
        ),
    )

    ensemble_tool = StructuredTool.from_function(
        func=lambda tweet: tools_container.ensemble_classify(tweet),
        name="ensemble_classify",
        description=(
            "Run ALL available models on a tweet and return majority-vote prediction. "
            "Use when high confidence is needed or when models disagree."
        ),
    )

    compare_tool = StructuredTool.from_function(
        func=lambda: tools_container.compare_models(),
        name="compare_models",
        description=(
            "Compare all trained models' performance metrics on the validation set. "
            "Returns a formatted table of Macro-F1 scores."
        ),
    )

    explain_tool = StructuredTool.from_function(
        func=lambda tweet, model="tfidf_logreg": tools_container.explain_prediction(tweet, model),
        name="explain_prediction",
        description=(
            "Explain WHY a tweet was classified a certain way by showing the top "
            "contributing tokens and their coefficients. Only works with linear models."
        ),
    )

    list_tool = StructuredTool.from_function(
        func=lambda: tools_container.list_models(),
        name="list_models",
        description="List all available trained model pipelines.",
    )

    tools = [classify_tool, ensemble_tool, compare_tool, explain_tool, list_tool]

    # Choose LLM based on available API key
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    elif os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        raise EnvironmentError(
            "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
        )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a financial tweet sentiment analysis assistant. You have access "
         "to multiple trained ML models for classifying tweets as Bearish, Bullish, "
         "or Neutral. Use your tools to classify tweets, compare models, explain "
         "predictions, and help users understand the classification pipeline. "
         "When classifying, use intelligent model routing by default — explain "
         "which model you chose and why."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    return executor


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    """Interactive chat loop for the classification agent."""
    print("=" * 60)
    print("Financial Tweet Sentiment Classifier — Agent Interface")
    print("=" * 60)
    print("Type a tweet to classify, or ask questions about models.")
    print("Commands: 'quit' to exit, 'compare' to compare models.")
    print("=" * 60)

    try:
        agent = create_agent()
    except EnvironmentError as e:
        print(f"\nError: {e}")
        print("Create a .env file with your API key. See .env.example.")
        return

    chat_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            result = agent.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
            print(f"\nAgent: {result['output']}")

            # Update chat history
            chat_history.append(("human", user_input))
            chat_history.append(("ai", result["output"]))

        except Exception as e:
            print(f"\nError: {e}")
            print("Try rephrasing your question or check your API key.")


if __name__ == "__main__":
    main()
