"""
nlp_utils.py
------------
Bütün NLP təhlil funksiyaları burada saxlanılır ki, həm terminal
proqramı (main.py), həm də veb proqramı (app.py) eyni kodu
istifadə etsin (kod təkrarlanmasın).
"""

import base64
import io
import string
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # şəkilləri ekran açmadan yaratmaq üçün
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud


# ----------------------------------------------------------------------
# TOKENIZATION
# ----------------------------------------------------------------------
def tokenize(text: str):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return sentences, words


# ----------------------------------------------------------------------
# TƏMİZLƏMƏ (stopwords + durğu işarələri)
# ----------------------------------------------------------------------
def clean_words(words):
    stop_words = set(stopwords.words("english"))
    return [
        w.lower()
        for w in words
        if w.lower() not in stop_words and w not in string.punctuation
    ]


# ----------------------------------------------------------------------
# POS TAGGING
# ----------------------------------------------------------------------
def pos_tagging(words):
    return nltk.pos_tag(words)


# ----------------------------------------------------------------------
# NAMED ENTITY RECOGNITION (NER)
# ----------------------------------------------------------------------
def named_entities(words):
    tagged = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tagged)
    entities = []
    for subtree in tree:
        if hasattr(subtree, "label"):
            entity_name = " ".join(c[0] for c in subtree)
            entities.append((entity_name, subtree.label()))
    return entities


# ----------------------------------------------------------------------
# SENTIMENT ANALYSIS (VADER)
# ----------------------------------------------------------------------
def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    if scores["compound"] >= 0.05:
        label = "Müsbət (Positive)"
    elif scores["compound"] <= -0.05:
        label = "Mənfi (Negative)"
    else:
        label = "Neytral (Neutral)"
    return scores, label


# ----------------------------------------------------------------------
# QRAFİKLƏRİ BASE64 ŞƏKİLDƏ QAYTARMAQ (fayla yazmadan, birbaşa HTML-ə
# yerləşdirmək üçün - veb proqramında istifadə olunur)
# ----------------------------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def word_frequency_chart_base64(cleaned_words, top_n=10):
    freq = Counter(cleaned_words)
    most_common = freq.most_common(top_n)
    if not most_common:
        return freq, None

    words, counts = zip(*most_common)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(words, counts, color="#4C72B0")
    ax.set_title(f"Ən çox işlənən {top_n} söz")
    ax.set_xlabel("Sözlər")
    ax.set_ylabel("Tezlik")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    return freq, fig_to_base64(fig)


def wordcloud_base64(cleaned_words):
    text_for_cloud = " ".join(cleaned_words)
    if not text_for_cloud.strip():
        return None
    wc = WordCloud(width=900, height=500, background_color="white").generate(text_for_cloud)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ----------------------------------------------------------------------
# HAMISINI BİR YERƏ TOPLAYAN FUNKSIYA
# ----------------------------------------------------------------------
def full_analysis(text: str, top_n: int = 10):
    """Verilmiş mətn üzərində bütün NLP təhlilini aparır və
    nəticələri lüğət (dict) şəklində qaytarır."""

    sentences, words = tokenize(text)
    cleaned = clean_words(words)

    freq, freq_chart_b64 = word_frequency_chart_base64(cleaned, top_n=top_n)
    wc_b64 = wordcloud_base64(cleaned)
    tagged = pos_tagging(words)
    entities = named_entities(words)
    scores, sentiment_label = sentiment_analysis(text)

    return {
        "char_count": len(text),
        "sentence_count": len(sentences),
        "word_count": len(words),
        "cleaned_word_count": len(cleaned),
        "top_words": freq.most_common(top_n),
        "freq_chart_b64": freq_chart_b64,
        "wordcloud_b64": wc_b64,
        "pos_tags": tagged[:20],
        "entities": entities,
        "sentiment_scores": scores,
        "sentiment_label": sentiment_label,
    }