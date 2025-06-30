from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import FastText
import joblib
from utils import preprocessing, encode_text, summarize_article, rechercher_wikipedia, recuperer_article_wikipedia
import scipy
from nltk.tokenize import sent_tokenize

# Config
MAX_LEN = 100
LABELS = ["World 🌍", "Sports 🏀", "Business 🏦", "Sci/Tech 🧬"]

# Chargement des modèles
model = load_model("DL_Models/model_DL_10_120000.keras")
fasttext_model = FastText.load("DL_Models/fasttext_model.bin")
word_index = joblib.load("DL_Models/word_index.pkl")
# Chargement du modèle ML
pipeline_ml = joblib.load("ML_Models/model_pipeline_ML_LogisticRegression.joblib")
# Chargement du Sumarizer
summarizer = joblib.load("ML_Sumarizer/summarize_pipeline_XGBoost.pkl")
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    label = None
    confidence = None
    label_ml_str = None
    confidence_ml = None
    summary = None
    head = None
    article = None
    summarized_article = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("text", "")
        action = request.form.get("action")

        if input_text.strip():
            if action == "classify":
                # Prétraitement
                text_clean = preprocessing(input_text)

                tokens = text_clean.split()
                encoded = encode_text(tokens, word_index)
                padded = pad_sequences([encoded], maxlen=MAX_LEN, padding='post')

                # Prédiction DL
                probs = model.predict(padded)[0]
                pred_class = int(np.argmax(probs))
                prediction = pred_class
                label = LABELS[pred_class]
                confidence = round(float(np.max(probs)) * 100, 2)

                # Prédiction ML
                label_ml = pipeline_ml.predict([text_clean])[0]
                label_ml_str = LABELS[label_ml]

                decision_scores = pipeline_ml.decision_function([text_clean])[0]
                probs_ml = scipy.special.softmax(decision_scores)
                confidence_ml = round(float(np.max(probs_ml)) * 100, 2)

            elif action == "summarize":
                summary = summarize_article(input_text, summarizer, preprocessing, top_k=3)

            elif action == "wiki":
                head = rechercher_wikipedia(input_text)
                article = recuperer_article_wikipedia(input_text)
                summarized_article = summarize_article(article, summarizer, preprocessing, top_k=5)


    return render_template("index.html",
                           prediction=prediction,
                           label=label,
                           confidence=confidence,
                           label_ml=label_ml_str,
                           confidence_ml=confidence_ml,
                           input_text=input_text,
                           summary=summary,
                           head=head,
                           article=article,
                           summarized_article=summarized_article,
    )

if __name__ == "__main__":
    app.run(debug=True)