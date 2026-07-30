"""
Kiçik NLP (Natural Language Processing) Layihəsi - Terminal versiyası
-----------------------------------------------------------------------
Bu proqram sample_text.txt faylını oxuyub üzərində NLP təhlili aparır
və nəticələri həm terminalda çap edir, həm də output/ qovluğuna
iki qrafik şəkil (.png) yazır.

Veb versiyası üçün: app.py faylına baxın (python app.py).
"""

import base64

from nlp_utils import full_analysis


def save_b64_image(b64_string, out_path):
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64_string))


def main():
    print("=" * 60)
    print("NLP LAYİHƏSİ - Mətn Analizi")
    print("=" * 60)

    with open("sample_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\nMətn oxundu ({len(text)} simvol).\n")

    result = full_analysis(text)

    print(f"-> Cümlə sayı: {result['sentence_count']}")
    print(f"-> Söz sayı (token): {result['word_count']}\n")
    print(f"-> Stopword təmizləmədən sonra qalan söz sayı: {result['cleaned_word_count']}\n")

    print("-> Ən çox işlənən sözlər:")
    for word, count in result["top_words"][:5]:
        print(f"   {word}: {count}")

    if result["freq_chart_b64"]:
        save_b64_image(result["freq_chart_b64"], "output/word_frequency.png")
        print("   (Qrafik saxlanıldı: output/word_frequency.png)\n")

    print("-> İlk 10 sözün nitq hissəsi (POS tag):")
    for word, tag in result["pos_tags"][:10]:
        print(f"   {word:15s} -> {tag}")
    print()

    print("-> Tapılan xüsusi adlar (Named Entities):")
    if result["entities"]:
        for name, label in result["entities"]:
            print(f"   {name} -> {label}")
    else:
        print("   Heç bir xüsusi ad tapılmadı.")
    print()

    scores = result["sentiment_scores"]
    print("-> Hiss analizi (Sentiment Analysis):")
    print(f"   Müsbət: {scores['pos']:.2f} | Mənfi: {scores['neg']:.2f} | "
          f"Neytral: {scores['neu']:.2f} | Ümumi bal: {scores['compound']:.2f}")
    print(f"   Nəticə: {result['sentiment_label']}\n")

    if result["wordcloud_b64"]:
        save_b64_image(result["wordcloud_b64"], "output/wordcloud.png")
        print("-> Word Cloud şəkli saxlanıldı: output/wordcloud.png")

    print("\n" + "=" * 60)
    print("Analiz tamamlandı! Nəticələri 'output/' qovluğunda tapa bilərsiniz.")
    print("=" * 60)


if __name__ == "__main__":
    main()