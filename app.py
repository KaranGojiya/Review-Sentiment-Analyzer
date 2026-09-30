import json
import pickle

import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Review Sentiment Analyzer",
    page_icon="⭐",
    layout="wide",
)

# =============================================================
# Files (change these names if yours are different)
# =============================================================
MODEL_PATH = "model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
CONFIG_PATH = "config.json"

MAX_CHARS = 5000

# Numbers shown in the "Model performance" section.
# Validation accuracy comes from the training log (epoch 5).
MODEL_METRICS = {
    "Validation accuracy": "76.1%",
    "Star classes": "5",
    "Vocabulary used": "10,000 words",
    "Parameters": "1.42 M",
}

EXAMPLE_REVIEWS = [
    "Excellent app, love it! Fast delivery and great prices. Highly recommend.",
    "The product looks okay, but the quality is not as expected. Delivery was late and the packaging was damaged. It works, but I am not satisfied with the overall experience.",
    "The app keeps crashing every time I try to pay. Very disappointed, want a refund.",
    "The app is easy to use and the prices are good, but delivery takes too much time. Customer support also needs improvement. Overall, it is okay.",
]

SENTIMENT_EMOJI = {"Negative": "😞", "Neutral": "😐", "Positive": "😊"}


# =============================================================
# Load model, tokenizer and config
# =============================================================
@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model, tokenizer, config


try:
    model, tokenizer, config = load_assets()
except Exception as exc:
    st.error(
        "The model could not be loaded. Check that model.h5, tokenizer.pkl and "
        "config.json are in the app folder, and that requirements.txt matches "
        "the TensorFlow version used for training."
    )
    st.exception(exc)
    st.stop()

MAX_LEN = config["MAX_LEN"]
NUM_CLASSES = config["num_classes"]
STAR_LABELS = [config["label_mapping"][str(i)] for i in range(NUM_CLASSES)]
OOV_INDEX = tokenizer.word_index.get(tokenizer.oov_token, 1)


# =============================================================
# Prediction
# =============================================================
def analyze(text):
    # 1. Turn words into numbers with the saved tokenizer
    sequence = tokenizer.texts_to_sequences([text])[0]

    # 2. Pad/cut to MAX_LEN. The model was trained with padding AT THE END ("post").
    padded = tf.keras.utils.pad_sequences(
        [sequence], maxlen=MAX_LEN, padding="post", truncating="post"
    )

    # 3. Predict the probability of each star rating
    probs = model.predict(padded, verbose=0)[0]

    top = int(probs.argmax())
    negative = float(probs[0] + probs[1])
    neutral = float(probs[2])
    positive = float(probs[3] + probs[4])
    sentiment = ["Negative", "Neutral", "Positive"][
        [negative, neutral, positive].index(max(negative, neutral, positive))
    ]

    return {
        "text": text,
        "probs": [float(p) for p in probs],
        "stars": top + 1,
        "label": STAR_LABELS[top],
        "confidence": float(probs[top] * 100),
        "expected": float(sum((i + 1) * p for i, p in enumerate(probs))),
        "sentiment": sentiment,
        "sentiment_pct": max(negative, neutral, positive) * 100,
        "n_words": len(sequence),
        "n_unknown": sequence.count(OOV_INDEX),
        "truncated": len(sequence) > MAX_LEN,
    }


def use_example(text):
    st.session_state["review"] = text


# =============================================================
# Session state
# =============================================================
st.session_state.setdefault("review", "")
st.session_state.setdefault("result", None)

# =============================================================
# Sidebar
# =============================================================
with st.sidebar:
    st.header("How it works")
    st.markdown(
        f"""
        1. The review is split into words and converted to numbers
           (a Keras tokenizer with a {tokenizer.num_words:,}-word vocabulary).
        2. It is padded or cut to {MAX_LEN} words.
        3. A **stacked GRU** network reads the sequence and outputs a
           probability for each star rating.
        """
    )
    st.subheader("Good to know")
    st.markdown(
        """
        - 1-star and 5-star reviews are easier to predict than 2, 3 and 4 stars.
        - The model only reads the first 100 words.
        - It works best on English app or product reviews.
        """
    )

# =============================================================
# Header and input
# =============================================================
st.title("⭐ Review Sentiment Analyzer")
st.caption(
    "Paste a customer review and a stacked GRU model predicts its star rating "
    "and overall sentiment."
)
st.divider()

st.text_area(
    "Customer review",
    key="review",
    placeholder="Example: The app is great, delivery was fast...",
    height=140,
    max_chars=MAX_CHARS,
)

st.caption("Try an example:")
example_cols = st.columns(len(EXAMPLE_REVIEWS))
for i, (col, text) in enumerate(zip(example_cols, EXAMPLE_REVIEWS)):
    col.button(text, on_click=use_example, args=(text,), key=f"example_{i}")

if st.button("🔍 Analyze review", type="primary"):
    cleaned = " ".join(st.session_state["review"].split())
    if cleaned == "":
        st.warning("Please enter a review first.")
    else:
        st.session_state["result"] = analyze(cleaned)

# =============================================================
# Results
# =============================================================
result = st.session_state["result"]

if result is None:
    st.info("The prediction will appear here after you analyze a review.")
else:
    st.divider()

    known_words = result["n_words"] - result["n_unknown"]
    if result["n_words"] == 0 or known_words == 0:
        st.warning(
            "None of the words in this review are in the model's vocabulary, so "
            "the prediction below is not reliable. Try a longer review in English."
        )

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.subheader("Prediction")
    
        rounded_expected = round(result["expected"])
        stars_text = "★" * rounded_expected + "☆" * (5 - rounded_expected)
    
        message = (
            f"{SENTIMENT_EMOJI[result['sentiment']]} **{result['sentiment']}** review, "
            f"estimated rating **{result['expected']:.2f}/5** {stars_text}"
        )
    
        if result["sentiment"] == "Positive":
            st.success(message)
        elif result["sentiment"] == "Negative":
            st.error(message)
        else:
            st.info(message)
    
        m1, m2, m3 = st.columns(3)
    
        m1.metric(
            "Estimated Rating",
            f"{result['expected']:.2f} / 5"
        )
    
        m2.metric(
            "Most Likely Class",
            result["label"]
        )
    
        m3.metric(
            "Class Confidence",
            f"{result['confidence']:.1f}%"
        )
    
        st.metric(
            f"Overall sentiment: {result['sentiment']}",
            f"{result['sentiment_pct']:.1f}%",
            help="Negative = 1-2 stars, Neutral = 3 stars, Positive = 4-5 stars."
        )

    with right:
        st.subheader("Probability of each rating")
        chart_data = pd.DataFrame(
            {"Probability (%)": [p * 100 for p in result["probs"]]},
            index=STAR_LABELS,
        )
        st.bar_chart(chart_data, height=280)

    with st.expander("How the model read this review"):
        st.write(f"Words read: **{result['n_words']}**")
        st.write(
            f"Words not in the model's vocabulary: **{result['n_unknown']}** "
            "(treated as unknown)"
        )
        if result["truncated"]:
            st.write(f"Only the first {MAX_LEN} words were used.")

# =============================================================
# Model details
# =============================================================
st.divider()
st.subheader("Model performance")
st.caption("Validation accuracy on exact 5-class star prediction.")

cols = st.columns(len(MODEL_METRICS))
for col, (name, value) in zip(cols, MODEL_METRICS.items()):
    col.metric(name, value)

with st.expander("Model architecture"):
    st.code(
        "Embedding (10,000 words x 128)\n"
        "GRU (128 units, returns sequences)\n"
        "GRU (64 units)\n"
        "Dropout (0.4)\n"
        "Dense (64, ReLU)\n"
        "Dropout (0.3)\n"
        "Dense (5, softmax)",
        language="text",
    )

# =============================================================
# Footer
# =============================================================
st.divider()
st.markdown(
    "Developed by **Karan Gojiya** | [GitHub](https://github.com/KaranGojiya) | "
    "[LinkedIn](https://www.linkedin.com/in/karan-gojiya)"
)
