  # Review Sentiment Analyzer

A deep learning web app that reads a customer review and predicts its **star rating (1 to 5)** and overall **sentiment** (negative, neutral or positive), using a stacked GRU network.

**[Live Demo](https://review-sentiment-analyzer-1.streamlit.app/)**

## Features

- Predicts a 1 to 5 star rating from free review text
- Shows the confidence and the probability of every star rating in a chart
- Groups the result into overall sentiment: Negative (1-2 stars), Neutral (3 stars), Positive (4-5 stars)
- Expected rating, the average star rating weighted by probability
- Shows how many words the model recognized and warns when none are in its vocabulary
- One-click example reviews

## How it works

1. **Tokenize:** a saved Keras `Tokenizer` (10,000-word vocabulary, `<OOV>` token for unknown words) converts the review into numbers.
2. **Pad:** each review is padded or cut to 100 words. The model was trained with `padding="post"` and `truncating="post"`, so the app uses the same settings. Using the Keras defaults (padding at the start) gives wrong predictions.
3. **Predict:** a stacked GRU network outputs a probability for each of the 5 star ratings.

## Model architecture

| Layer | Details |
|---|---|
| Embedding | 10,000 words, 128 dimensions |
| GRU | 128 units, returns sequences |
| GRU | 64 units |
| Dropout | 0.4 |
| Dense | 64 units, ReLU |
| Dropout | 0.3 |
| Dense | 5 units, softmax |

1.42 million parameters. Trained with the Adam optimizer and sparse categorical cross-entropy loss.

## Results

| Metric | Result |
|---|---|
| Validation accuracy (5-class, exact star) | 76.07% |
| Training accuracy | 76.25% |

Measured at epoch 5 of training. Training and validation accuracy are close, so the model is not overfitting.

**Dataset:** [add dataset name and number of reviews]

## Tech stack

Python, TensorFlow / Keras, Streamlit, Pandas, NumPy

## Project structure

```
review-sentiment-analyzer/
├── app.py            # Streamlit app
├── model.h5          # Trained stacked GRU model
├── tokenizer.pkl     # Fitted Keras tokenizer
├── config.json       # MAX_LEN, vocabulary size and label mapping
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/KaranGojiya/[repo-name].git
cd [repo-name]
pip install -r requirements.txt
streamlit run app.py
```

Use Python 3.11 or 3.12, since TensorFlow does not yet support the newest Python versions.

## Limitations

- Predicting the exact star rating is hard. 1-star and 5-star reviews are easier than 2, 3 and 4 stars, so the overall sentiment is usually more reliable than the exact number of stars.
- Only the first 100 words of a review are used.
- The vocabulary is limited to 10,000 words. Other words are treated as unknown.
- Works best on English reviews similar to the training data.

## Future improvements

- Show a confusion matrix and per-star precision and recall
- Handle class imbalance with class weights
- Compare against an LSTM and a bidirectional GRU
- Batch mode: upload a CSV of reviews and download the predictions

## Author

**Karan Gojiya**, B.Tech CSE, IIIT Bhopal

[LinkedIn](https://www.linkedin.com/in/karan-gojiya) · [GitHub](https://github.com/KaranGojiya)
