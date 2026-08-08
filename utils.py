"""
utils.py
Text cleanup, word cloud generation, and small formatting helpers for the dashboard.
"""

import re
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import STOPWORDS, WordCloud

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_EXTRA_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light cleanup before display / word cloud — NOT applied before model inference,
    since the sentiment model expects natural (even messy) text."""
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = _EXTRA_WS_RE.sub(" ", text).strip()
    return text


def parse_input_text(raw_input: str) -> list[str]:
    """Split a textarea's raw input into individual entries (one per line)."""
    return [line.strip() for line in raw_input.splitlines() if line.strip()]


def generate_wordcloud_image(texts: list[str]):
    """Returns a PNG image buffer of a word cloud built from the given texts."""
    joined = " ".join(clean_text(t) for t in texts)
    stopwords = set(STOPWORDS) | {"amp", "rt"}

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords,
        colormap="viridis",
    ).generate(joined or "no data")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def sentiment_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a two-column DataFrame: label, count — fixed order for consistent chart colors."""
    counts = df["label"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    result = counts.rename_axis("label").reset_index(name="count")
    return result
