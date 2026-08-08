# 💬 Sentiment Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green)

Paste tweets, reviews, or any short text and get instant sentiment classification from a pretrained RoBERTa model — with confidence bars, a distribution chart, a word cloud, and an optional confusion matrix if you supply ground-truth labels.

## Demo

> _Add a screenshot here: `docs/demo.png`_
>
> _Suggested shot: the Analyze tab with ~5 sample reviews classified, showing the results table, distribution bar chart, and word cloud._

## Features

- Batch sentiment classification (negative / neutral / positive) using `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Per-entry confidence breakdown across all 3 classes, not just the top label
- Sentiment distribution bar chart and horizontal per-entry confidence chart (Plotly)
- Auto-generated word cloud from the input text
- Optional confusion matrix when ground-truth labels are supplied — useful for quickly eyeballing model accuracy on your own labeled sample
- Model loaded once and cached across reruns (`st.cache_resource`)

## Project structure

```
sentiment-dashboard/
├── app.py                # Streamlit UI
├── model.py                # Model loading, batch prediction, confusion matrix
├── utils.py                  # Text cleanup, word cloud generation
├── sample_data/
│   └── reviews.csv            # 5 labeled example reviews to try the confusion matrix
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/sentiment-dashboard.git
cd sentiment-dashboard
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

First run downloads the model (~500MB) from the Hugging Face Hub; it's cached locally afterward.

## Tech stack

`HuggingFace Transformers` · `PyTorch` · `scikit-learn` · `Plotly` · `Streamlit` · `WordCloud`

## Possible extensions

- Add CSV upload for bulk analysis instead of pasting text
- Swap in a domain-fine-tuned model (e.g. financial or medical sentiment) via a dropdown
- Add aspect-based sentiment (which part of the review is positive/negative)
- Cache predictions by text hash to avoid re-running identical inputs

---

### Resume bullet points

- Built a sentiment analysis dashboard using a pretrained BERT-based model (RoBERTa) to classify text into 3 sentiment classes with per-class confidence scores
- Designed interactive Plotly visualizations (distribution charts, per-entry confidence breakdown, confusion matrix) for both aggregate and individual-level model inspection
- Implemented model caching in Streamlit to avoid reloading a 500MB transformer model on every UI interaction

### Recruiter talking points

- **What it demonstrates:** applying a pretrained transformer model to a practical NLP task, plus building the visualization layer to make model outputs actually interpretable.
- **Design decisions worth discussing:** why a tweet-tuned model over a generic sentiment model; why softmax probabilities are shown for all 3 classes instead of just the top prediction (helps spot low-confidence/ambiguous cases).
- **What you'd improve at scale:** batch inference optimization (dynamic batching, GPU serving), a proper eval set instead of ad hoc ground-truth entry, model versioning if swapping between multiple sentiment models.
