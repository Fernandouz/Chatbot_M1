import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy

STOPWORDS = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

tag_map = {
    "J": wordnet.ADJ,    # Adjectif
    "V": wordnet.VERB,   # Verbe
    "N": wordnet.NOUN,   # Nom
    "R": wordnet.ADV     # Adverbe
}

def clean_text(text: str) -> list:
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in STOPWORDS]
    return tokens

def lemmatik(seq):
    tag = pos_tag(seq)
    lenma = [
        lemmatizer.lemmatize(word, tag_map.get(tag[0], wordnet.NOUN))  # Défaut : nom
        for word, tag in tag
    ]
    return " ".join(lenma)

def preprocessing(seq):
    prep = clean_text(seq)
    cleaned = lemmatik(prep)
    return cleaned

def encode_text(tokens, word_index):
    return [word_index[word] for word in tokens if word in word_index]

## SUMMARIZE

def summarize_tf_idf(text, n_sentences=3):

    sentences = sent_tokenize(text)

    # texte trop court on renvoi le texte direct

    if len(sentences) <= n_sentences:
        return text

    # 2. Calcule TF-IDF par phrase (chaque phrase = document)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)

    # 3. Score : moyenne TF-IDF de chaque phrase
    sentence_scores = np.asarray(tfidf_matrix.mean(axis=1)).ravel()

    # 4. Indices des meilleures phrases
    top_idx = sentence_scores.argsort()[-n_sentences:][::-1]
    top_idx_sorted = sorted(top_idx)  # pour garder l'ordre dans le texte

    # 5. Résumé
    summary = " ".join([sentences[i] for i in top_idx_sorted])
    return summary

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

def summarize_tf_idf_with_scores(text, n_sentences=3, position_weight=0.5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.tokenize import sent_tokenize
    import numpy as np

    sentences = sent_tokenize(text)

    if len(sentences) <= n_sentences:
        return text, [], [], [], []

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=1)).ravel()

    n = len(sentences)
    # position_scores = np.array([(1 - (i / n)) for i in range(n)])
    position_scores = np.exp(-np.arange(n))

    # Similarité de chaque phrase avec le document global (moyenne TF-IDF)
    doc_vector = np.asarray(tfidf_matrix.mean(axis=0)).ravel().reshape(1, -1)
    cosine_sim = cosine_similarity(tfidf_matrix, doc_vector)
    cosine_similarity_scores = cosine_sim.ravel()

    # final_scores = (1 - position_weight) * tfidf_scores + position_weight * position_scores
    # final_scores = (
    # 0.6 * normalize(tfidf_scores) +
    # 0.3 * normalize(position_scores) +
    # 0.1 * normalize(cosine_similarity_scores)
    # )
    final_scores = (
    0.7 * normalize(tfidf_scores) +
    0.2 * normalize(position_scores) +
    0.1 * normalize(cosine_similarity_scores)
    )
    top_idx = final_scores.argsort()[-n_sentences:][::-1]
    top_idx_sorted = sorted(top_idx)

    summary = " ".join([sentences[i] for i in top_idx_sorted])
    return summary, sentences, tfidf_scores, position_scores, cosine_similarity_scores, final_scores

def summarize_article(article: str, summarizer, preprocessing, top_k: int = 5) -> list:
    # Découper et nettoyer les phrases
    sentences = sent_tokenize(article)
    cleaned_sentences = [preprocessing(s) for s in sentences]

    # Vectorisation
    vectorizer = summarizer.named_steps['vectorizer']
    clf = summarizer.named_steps['classifier']
    X_vec = vectorizer.transform(cleaned_sentences)

    # Prédiction des probabilités de pertinence
    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(X_vec)[:, 1]
    elif hasattr(clf, "decision_function"):
        decision_scores = clf.decision_function(X_vec)
        probas = scipy.special.softmax(np.vstack([-decision_scores, decision_scores]), axis=0)[1]
    else:
        raise ValueError("Le modèle ne supporte ni predict_proba ni decision_function")

    # Sélection des top_k phrases les plus probables
    top_k = min(top_k, len(sentences))  # éviter erreur si top_k > nb phrases
    top_k_idx = np.argsort(probas)[-top_k:][::-1]
    summary_sentences = [sentences[i] for i in sorted(top_k_idx)]  # Tri pour garder l'ordre

    return summary_sentences

### WIKIPEDIA

import requests

def rechercher_wikipedia(terme):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{terme}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        titre = data.get("title", "Sans titre")
        extrait = data.get("extract", "Pas de résumé trouvé.")
        url_article = data.get("content_urls", {}).get("desktop", {}).get("page", "#")

        return extrait



    except requests.exceptions.HTTPError:
        return f"Aucun article trouvé pour « {terme} »."
    except Exception as e:
        return f"Erreur inattendue : {e}"

def recuperer_article_wikipedia(titre):
    """
    Récupère le contenu complet brut (texte seul) d'un article Wikipédia.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": titre,
        "exintro": True,
        "explaintext": True,
        "redirects": 1  # Pour suivre les redirections
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        pages = data["query"]["pages"]
        page = next(iter(pages.values()))

        if "extract" in page and page["extract"].strip():
            return page["extract"]
        else:
            return ""

    except Exception as e:
        return f"Erreur lors de la récupération de l'article : {e}"