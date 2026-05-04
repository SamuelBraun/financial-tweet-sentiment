"""
Feature engineering module for financial tweet sentiment classification.

Provides sklearn-compatible feature extractors for:
- Bag-of-Words / TF-IDF
- Word2Vec (custom + pretrained GloVe)
- Transformer encoder embeddings (frozen, used as feature extractors)
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from src import MODELS_DIR, RANDOM_STATE

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===========================================================================
# 4a. Bag-of-Words family
# ===========================================================================

class BowFeaturizer(BaseEstimator, TransformerMixin):
    """CountVectorizer wrapper with sklearn-compatible interface."""

    def __init__(self, **kwargs):
        defaults = dict(max_features=10000)
        defaults.update(kwargs)
        self.vectorizer = CountVectorizer(**defaults)
        self.name = "BoW"

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)

    def fit_transform(self, X, y=None):
        return self.vectorizer.fit_transform(X)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()


class TfidfFeaturizer(BaseEstimator, TransformerMixin):
    """TfidfVectorizer wrapper with tweet-optimized defaults."""

    def __init__(self, **kwargs):
        defaults = dict(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=20000,
        )
        defaults.update(kwargs)
        self.vectorizer = TfidfVectorizer(**defaults)
        self.name = "TF-IDF"

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)

    def fit_transform(self, X, y=None):
        return self.vectorizer.fit_transform(X)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()


# ===========================================================================
# 4b. Word2Vec family
# ===========================================================================

class Word2VecFeaturizer(BaseEstimator, TransformerMixin):
    """Word2Vec document embeddings via mean pooling.

    Supports both custom-trained and pretrained (GloVe) models.

    Args:
        mode: 'custom' — train on the corpus, 'glove' — use pretrained.
        vector_size: Embedding dimension (only for custom mode).
        pooling: 'mean' or 'tfidf_weighted'.
        cache_path: Path to save/load trained model.
    """

    def __init__(self, mode: str = "custom", vector_size: int = 100,
                 window: int = 5, min_count: int = 2, epochs: int = 20,
                 pooling: str = "mean", cache_path: Optional[str] = None):
        self.mode = mode
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.pooling = pooling
        self.cache_path = cache_path or str(MODELS_DIR / "w2v_custom.bin")
        self.model = None
        self.tfidf = None
        self.name = f"W2V-{mode}-{pooling}"

    def fit(self, X, y=None):
        """Train or load Word2Vec model.

        Args:
            X: List of token lists (already tokenized texts).
        """
        from gensim.models import Word2Vec
        import gensim.downloader as api

        if self.mode == "glove":
            cache = Path(self.cache_path).parent / "glove-twitter-100.model"
            if cache.exists():
                self.model = api.load("glove-twitter-100")
            else:
                print("Downloading GloVe Twitter embeddings (first time only)...")
                self.model = api.load("glove-twitter-100")
            self.vector_size = self.model.vector_size
        else:
            cache = Path(self.cache_path)
            if cache.exists():
                print(f"Loading cached Word2Vec from {cache}")
                self.model = Word2Vec.load(str(cache))
            else:
                print("Training Word2Vec model...")
                self.model = Word2Vec(
                    sentences=X,
                    vector_size=self.vector_size,
                    window=self.window,
                    min_count=self.min_count,
                    sg=1,  # skip-gram
                    epochs=self.epochs,
                    seed=RANDOM_STATE,
                    workers=os.cpu_count() or 4,
                )
                cache.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(cache))
                print(f"Saved Word2Vec to {cache}")

        # Fit TF-IDF weights for weighted pooling
        if self.pooling == "tfidf_weighted":
            corpus_strings = [" ".join(tokens) for tokens in X]
            self.tfidf = TfidfVectorizer()
            self.tfidf.fit(corpus_strings)

        return self

    def _get_wv(self):
        """Get the word vectors object from the model."""
        if hasattr(self.model, "wv"):
            return self.model.wv
        return self.model  # pretrained KeyedVectors

    def _doc_vector_mean(self, tokens: List[str]) -> np.ndarray:
        """Mean pooling of token embeddings."""
        wv = self._get_wv()
        vectors = [wv[t] for t in tokens if t in wv]
        if not vectors:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)

    def _doc_vector_tfidf(self, tokens: List[str]) -> np.ndarray:
        """TF-IDF weighted mean pooling."""
        wv = self._get_wv()
        vocab = self.tfidf.vocabulary_
        idf = self.tfidf.idf_

        vectors, weights = [], []
        for t in tokens:
            if t in wv and t in vocab:
                vectors.append(wv[t])
                weights.append(idf[vocab[t]])

        if not vectors:
            return np.zeros(self.vector_size)

        weights = np.array(weights)
        weighted = np.average(vectors, axis=0, weights=weights)
        return weighted

    def transform(self, X) -> np.ndarray:
        """Transform token lists to document embeddings.

        Args:
            X: List of token lists.
        Returns:
            np.ndarray of shape (n_docs, vector_size).
        """
        if self.pooling == "tfidf_weighted" and self.tfidf is not None:
            pool_fn = self._doc_vector_tfidf
        else:
            pool_fn = self._doc_vector_mean

        embeddings = np.array([pool_fn(tokens) for tokens in X])
        return embeddings

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


# ===========================================================================
# 4c. Transformer Encoder family (frozen feature extractor)
# ===========================================================================

class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    """Extract frozen Transformer embeddings for use as features.

    Uses mean pooling of the last hidden states (with attention mask)
    to produce fixed-length document vectors. Embeddings are cached
    to disk as .npy files to avoid recomputation on CPU.

    Args:
        checkpoint: HuggingFace model name/path.
        max_length: Maximum token length (128 is plenty for tweets).
        batch_size: Inference batch size (8-16 for CPU).
        cache_name: Filename stem for the cached .npy file.
    """

    def __init__(self, checkpoint: str = "distilbert-base-uncased",
                 max_length: int = 128, batch_size: int = 16,
                 cache_name: Optional[str] = None):
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_name = cache_name or checkpoint.replace("/", "_")
        self.tokenizer = None
        self.model = None
        self.name = f"Transformer-{self.cache_name}"

    def _load_model(self):
        """Lazy-load model and tokenizer."""
        if self.model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModel

        # Set CPU threads
        torch.set_num_threads(os.cpu_count() or 4)

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModel.from_pretrained(self.checkpoint)
        self.model.eval()

    def _extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract mean-pooled embeddings from the model."""
        import torch

        self._load_model()
        all_embeddings = []

        for i in tqdm(range(0, len(texts), self.batch_size),
                      desc=f"Extracting {self.cache_name} embeddings"):
            batch = texts[i:i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self.model(**encoded)

            # Mean pooling with attention mask
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embeddings.append(pooled.numpy())

        return np.vstack(all_embeddings)

    def fit(self, X, y=None):
        """No-op for frozen encoder — model is pretrained."""
        return self

    def transform(self, X, cache_suffix: str = "") -> np.ndarray:
        """Extract or load cached embeddings.

        Args:
            X: List of preprocessed text strings.
            cache_suffix: Added to cache filename (e.g., '_train', '_val').
        """
        cache_path = MODELS_DIR / f"embeddings_{self.cache_name}{cache_suffix}.npy"

        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)

        print(f"Computing embeddings with {self.checkpoint} "
              f"({len(X)} texts, batch_size={self.batch_size})...")
        embeddings = self._extract_embeddings(list(X))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"Cached embeddings to {cache_path}")

        return embeddings

    def fit_transform(self, X, y=None, cache_suffix: str = "_train") -> np.ndarray:
        return self.fit(X, y).transform(X, cache_suffix=cache_suffix)


# ===========================================================================
# Utilities
# ===========================================================================

def get_top_features_by_class(vectorizer, classifier, class_names: List[str],
                              n: int = 15) -> dict:
    """Extract top-N features per class from a linear classifier.

    Works with LogisticRegression, LinearSVC, or any model with .coef_.

    Returns:
        dict mapping class_name -> list of (feature_name, coefficient).
    """
    feature_names = vectorizer.get_feature_names_out()
    top_features = {}

    if hasattr(classifier, "coef_"):
        for i, cls_name in enumerate(class_names):
            coef = classifier.coef_[i] if classifier.coef_.ndim > 1 else classifier.coef_
            top_idx = np.argsort(coef)[-n:][::-1]
            top_features[cls_name] = [
                (feature_names[j], coef[j]) for j in top_idx
            ]

    return top_features
