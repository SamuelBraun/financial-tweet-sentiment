"""
Classification models for financial tweet sentiment analysis.

Provides a factory function get_model() and training helpers for
traditional ML classifiers and Transformer head-only fine-tuning.
"""

import os
from typing import Optional, Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from src import RANDOM_STATE

# Lazy import for XGBoost (may not be installed)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


# ===========================================================================
# Model factory
# ===========================================================================

_MODEL_REGISTRY = {}


def register_model(name):
    """Decorator to register a model constructor."""
    def wrapper(fn):
        _MODEL_REGISTRY[name] = fn
        return fn
    return wrapper


@register_model("logreg")
def _logreg(**kwargs):
    defaults = dict(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
    )
    defaults.update(kwargs)
    return LogisticRegression(**defaults)


@register_model("svm")
def _svm(**kwargs):
    defaults = dict(
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
        C=1.0,
    )
    defaults.update(kwargs)
    return LinearSVC(**defaults)


@register_model("xgboost")
def _xgboost(**kwargs):
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")
    defaults = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


@register_model("random_forest")
def _random_forest(**kwargs):
    defaults = dict(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


def get_model(name: str, **kwargs):
    """Factory function to create a classifier by name.

    Supported names: 'logreg', 'svm', 'xgboost', 'random_forest'.

    Args:
        name: Model identifier.
        **kwargs: Overrides for model hyperparameters.
    Returns:
        Configured sklearn-compatible classifier.
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name](**kwargs)


# ===========================================================================
# Hyperparameter grid search helper
# ===========================================================================

def get_param_grid(name: str) -> Dict[str, list]:
    """Return default parameter grid for grid search.

    Args:
        name: Model identifier.
    Returns:
        Dict of param_name -> list of values.
    """
    grids = {
        "logreg": {"C": [0.01, 0.1, 1.0, 10.0]},
        "svm": {"C": [0.01, 0.1, 1.0, 10.0]},
        "xgboost": {
            "n_estimators": [100, 300],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
        },
        "random_forest": {
            "n_estimators": [100, 300],
            "max_depth": [10, 20, None],
        },
    }
    return grids.get(name, {})


# ===========================================================================
# Transformer head-only fine-tuning
# ===========================================================================

def train_transformer_classifier(
    checkpoint: str,
    train_texts, train_labels,
    val_texts, val_labels,
    num_labels: int = 3,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    freeze_encoder: bool = True,
    output_dir: Optional[str] = None,
):
    """Fine-tune a Transformer model with a classification head.

    By default, the encoder is frozen and only the classification head
    is trained — this is the CPU-friendly approach.

    Args:
        checkpoint: HuggingFace model identifier.
        train_texts: Training texts (preprocessed with pp_transformer).
        train_labels: Training labels (0, 1, 2).
        val_texts: Validation texts.
        val_labels: Validation labels.
        num_labels: Number of output classes.
        epochs: Training epochs.
        batch_size: Per-device batch size.
        learning_rate: Learning rate for head.
        max_length: Max token length.
        freeze_encoder: If True, freeze all encoder parameters.
        output_dir: Directory for checkpoints.

    Returns:
        (trainer, predictions_on_val)
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset

    torch.set_num_threads(os.cpu_count() or 4)

    if output_dir is None:
        from src import MODELS_DIR
        output_dir = str(MODELS_DIR / f"ft_{checkpoint.replace('/', '_')}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels
    )

    # Freeze encoder layers if requested
    if freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name and "pre_classifier" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Frozen encoder: {trainable:,} / {total:,} parameters trainable "
              f"({100*trainable/total:.1f}%)")

    # Create datasets
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_ds = Dataset.from_dict({"text": list(train_texts), "label": list(train_labels)})
    val_ds = Dataset.from_dict({"text": list(val_texts), "label": list(val_labels)})

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Training arguments (CPU-optimized)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,  # CPU — no fp16
        seed=RANDOM_STATE,
        logging_steps=50,
        report_to="none",
    )

    # Metrics callback
    from sklearn.metrics import f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"macro_f1": macro_f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Predictions on validation
    preds_output = trainer.predict(val_ds)
    val_preds = np.argmax(preds_output.predictions, axis=-1)

    return trainer, val_preds


# ===========================================================================
# Decoder classification (extra credit)
# ===========================================================================

def build_fewshot_prompt(tweet: str, examples: list) -> str:
    """Build a few-shot classification prompt for a decoder model.

    Args:
        tweet: The tweet to classify.
        examples: List of (text, label) tuples (2 per class recommended).

    Returns:
        Formatted prompt string.
    """
    label_map = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    lines = [
        "Classify the following financial tweet as Bearish (0), "
        "Bullish (1), or Neutral (2). Respond with ONLY the number.\n"
    ]
    for ex_text, ex_label in examples:
        lines.append(f"Tweet: {ex_text}")
        lines.append(f"Label: {ex_label}\n")

    lines.append(f"Tweet: {tweet}")
    lines.append("Label:")
    return "\n".join(lines)


def decoder_classify_batch(
    tweets: list,
    examples: list,
    model_name: str = "flan-t5-small",
    cache_path: Optional[str] = None,
) -> list:
    """Classify tweets using a decoder/encoder-decoder model.

    Supports local Flan-T5 or API-based (via environment variables).

    Args:
        tweets: List of tweet texts.
        examples: Few-shot examples [(text, label), ...].
        model_name: 'flan-t5-small', 'flan-t5-base', or 'api'.
        cache_path: Path to JSON cache file.

    Returns:
        List of predicted labels (0, 1, 2).
    """
    import json
    import re

    if cache_path is None:
        from src import OUTPUTS_DIR
        cache_path = str(OUTPUTS_DIR / "decoder_cache.json")

    # Load cache
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)

    predictions = []
    uncached_indices = []
    uncached_tweets = []

    for i, tweet in enumerate(tweets):
        if tweet in cache:
            predictions.append(cache[tweet])
        else:
            predictions.append(None)
            uncached_indices.append(i)
            uncached_tweets.append(tweet)

    if not uncached_tweets:
        return predictions

    if model_name.startswith("flan-t5"):
        # Local Flan-T5
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        full_name = f"google/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(full_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(full_name)
        model.eval()

        for idx, tweet in zip(uncached_indices, uncached_tweets):
            prompt = build_fewshot_prompt(tweet, examples)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=512)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=5)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Parse label
            match = re.search(r"[012]", response)
            label = int(match.group()) if match else 2  # default Neutral
            predictions[idx] = label
            cache[tweet] = label

    elif model_name == "api":
        # API-based classification (OpenAI or Anthropic)
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key found in .env. Set OPENAI_API_KEY or "
                             "ANTHROPIC_API_KEY.")

        if os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        for idx, tweet in zip(uncached_indices, uncached_tweets):
            prompt = build_fewshot_prompt(tweet, examples)
            response = llm.invoke(prompt).content.strip()

            match = re.search(r"[012]", response)
            label = int(match.group()) if match else 2
            predictions[idx] = label
            cache[tweet] = label

    # Save cache
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    return predictions
