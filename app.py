
import os
import joblib
import numpy as np
import torch

from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 1. Flask Setup

app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_FOLDER = os.path.join(
    BASE_DIR,
    "fake_news_hybrid"
)

BERT_FOLDER = os.path.join(
    MODEL_FOLDER,
    "bert_model"
)

MAX_LENGTH = 256


# 2. Device

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# 3. Load BERT

tokenizer = AutoTokenizer.from_pretrained(
    BERT_FOLDER
)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    BERT_FOLDER
)

bert_model.to(device)
bert_model.eval()


# 4. Load Logistic Regression and TF-IDF

tfidf = joblib.load(
    os.path.join(
        MODEL_FOLDER,
        "tfidf_vectorizer.joblib"
    )
)

lr_model = joblib.load(
    os.path.join(
        MODEL_FOLDER,
        "logistic_regression.joblib"
    )
)


# 5. Load Ensemble Weights

ensemble_config = joblib.load(
    os.path.join(
        MODEL_FOLDER,
        "ensemble_config.joblib"
    )
)

BERT_WEIGHT = ensemble_config[
    "bert_weight"
]

LR_WEIGHT = ensemble_config[
    "lr_weight"
]


# 6. Hybrid Prediction Function

def predict_news(title, article_text):

    encoding = tokenizer(
        str(title),
        str(article_text),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoding[
        "input_ids"
    ].to(device)

    attention_mask = encoding[
        "attention_mask"
    ].to(device)

    with torch.no_grad():

        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        bert_probs = torch.softmax(
            outputs.logits,
            dim=1
        )[0].cpu().numpy()


    combined_text = (
        str(title).strip()
        + " "
        + str(article_text).strip()
    ).strip()

    lr_features = tfidf.transform(
        [combined_text]
    )

    lr_probs = lr_model.predict_proba(
        lr_features
    )[0]


    final_probs = (
        BERT_WEIGHT * bert_probs
        +
        LR_WEIGHT * lr_probs
    )

    prediction = int(
        np.argmax(final_probs)
    )

    confidence = float(
        final_probs[prediction]
    )


    if prediction == 0: #I changed this to 0 because the model was predicting 0 for real news and 1 for fake news, so I changed it to match that in the deployment
        result = "REAL NEWS"

    else:
        result = "FAKE NEWS"


    return {
        "prediction": result,
        "confidence": round(
            confidence * 100,
            2
        )
    }


# 7. Prediction API

@app.route("/")
def home():
    return render_template("index.html")

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "No data received."
            }), 400


        title = data.get(
            "title",
            ""
        ).strip()

        article = data.get(
            "article",
            ""
        ).strip()


        if not title and not article:

            return jsonify({
                "error":
                "Please provide a title or article."
            }), 400


        result = predict_news(
            title,
            article
        )


        return jsonify(result)


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# 8. Health Route

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "healthy",
        "model":
        "BERT + Logistic Regression"
    })


# 9. Run Flask

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
