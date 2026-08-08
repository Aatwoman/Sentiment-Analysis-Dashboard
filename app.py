"""
app.py
Streamlit dashboard: paste text (tweets/reviews) -> BERT-based sentiment
classification -> confidence bars, distribution chart, word cloud, and an
optional confusion matrix if the user provides ground-truth labels.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from model import compute_confusion_matrix, load_classifier, predict_batch
from utils import generate_wordcloud_image, parse_input_text, sentiment_counts

st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="💬", layout="wide")

st.title("💬 Sentiment Analysis Dashboard")
st.caption("Paste tweets, reviews, or any short text (one per line) to classify sentiment with a pretrained BERT model.")


@st.cache_resource(show_spinner="Loading sentiment model (first run only)...")
def get_classifier():
    return load_classifier()


get_classifier()  # warm the cache on first load

tab_analyze, tab_about = st.tabs(["Analyze", "About this model"])

with tab_analyze:
    col_input, col_options = st.columns([3, 1])

    with col_input:
        raw_text = st.text_area(
            "Paste text — one entry per line",
            height=200,
            placeholder="I absolutely love this product!\nShipping took forever, not happy.\nIt's fine, nothing special.",
        )

    with col_options:
        st.markdown("**Optional: ground truth**")
        raw_labels = st.text_area(
            "True labels (one per line: Positive / Neutral / Negative)",
            height=200,
            help="If provided, a confusion matrix will be shown below.",
        )

    run = st.button("Analyze sentiment", type="primary")

    if run:
        texts = parse_input_text(raw_text)
        if not texts:
            st.error("Paste at least one line of text.")
        else:
            with st.spinner(f"Classifying {len(texts)} entries..."):
                df = predict_batch(texts)

            st.subheader("Results")
            st.dataframe(
                df[["text", "label", "confidence", "negative", "neutral", "positive"]].style.format(
                    {"confidence": "{:.1%}", "negative": "{:.1%}", "neutral": "{:.1%}", "positive": "{:.1%}"}
                ),
                use_container_width=True,
            )

            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Sentiment distribution")
                dist = sentiment_counts(df)
                fig = px.bar(
                    dist, x="label", y="count", color="label",
                    color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"},
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("Per-entry confidence")
                melted = df.melt(
                    id_vars=["text"], value_vars=["negative", "neutral", "positive"],
                    var_name="class", value_name="score",
                )
                melted["short_text"] = melted["text"].str.slice(0, 30) + "..."
                fig2 = px.bar(
                    melted, x="score", y="short_text", color="class", orientation="h",
                    color_discrete_map={"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"},
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Word cloud")
            wc_image = generate_wordcloud_image(texts)
            st.image(wc_image, use_column_width=True)

            true_labels = parse_input_text(raw_labels)
            if true_labels:
                if len(true_labels) != len(texts):
                    st.warning(
                        f"Provided {len(true_labels)} labels for {len(texts)} texts — "
                        "confusion matrix needs one label per line, matching the input count."
                    )
                else:
                    st.subheader("Confusion matrix")
                    labels_order = ["Positive", "Neutral", "Negative"]
                    cm = compute_confusion_matrix(true_labels, df["label"].tolist(), labels_order)
                    cm_fig = px.imshow(
                        cm, x=labels_order, y=labels_order,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        text_auto=True, color_continuous_scale="Blues",
                    )
                    st.plotly_chart(cm_fig, use_container_width=True)

with tab_about:
    st.markdown(
        """
        **Model:** [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

        A RoBERTa-base model fine-tuned on ~124M tweets for 3-class sentiment
        classification (negative / neutral / positive). Well suited to short,
        informal text such as tweets and product reviews.

        **Pipeline:** raw text → tokenizer → RoBERTa → softmax over 3 classes → top label + confidence.
        """
    )
