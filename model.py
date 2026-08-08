"""
model.py
Loads a pretrained sentiment classifier and exposes simple batch-predict
and evaluation helpers.

Uses cardiffnlp/twitter-roberta-base-sentiment-latest, a RoBERTa model
fine-tuned on ~124M tweets — well-suited to short, informal text
(tweets, reviews) with 3 classes: negative / neutral / positive.
"""

from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABEL_MAP = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}


@lru_cache(maxsize=1)
def load_classifier():
    """Load once and cache — reused across Streamlit reruns via st.cache_resource in app.py."""
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=None,  # return scores for all 3 classes, not just the top one
        truncation=True,
        max_length=512,
    )


def predict_batch(texts: List[str]) -> pd.DataFrame:
    """
    Run sentiment prediction over a list of texts.
    Returns a DataFrame with columns: text, label, confidence, neg, neu, pos
    """
    classifier = load_classifier()
    raw_results = classifier(list(texts), batch_size=16)

    rows = []
    for text, scores in zip(texts, raw_results):
        score_dict = {s["label"]: s["score"] for s in scores}
        top_label = max(score_dict, key=score_dict.get)
        rows.append(
            {
                "text": text,
                "label": LABEL_MAP.get(top_label, top_label),
                "confidence": score_dict[top_label],
                "negative": score_dict.get("negative", 0.0),
                "neutral": score_dict.get("neutral", 0.0),
                "positive": score_dict.get("positive", 0.0),
            }
        )
    return pd.DataFrame(rows)


def compute_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]):
    """Returns a confusion matrix as a numpy array, for the given label order."""
    return confusion_matrix(y_true, y_pred, labels=labels)
