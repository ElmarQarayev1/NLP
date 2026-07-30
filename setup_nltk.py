"""
Bu skripti layihəni işə salmazdan ƏVVƏL, YALNIZ BİR DƏFƏ çalışdırın.
NLTK kitabxanasının işləməsi üçün lazım olan dil məlumat bazalarını
(tokenizer, stopwords, POS tagger, sentiment lexicon və s.) internetdən
yükləyir. Növbəti dəfələr işlətməyə ehtiyac yoxdur.

İşə salmaq: python setup_nltk.py
"""

import nltk

PACKAGES = [
    "punkt",                          # cümlə/söz bölmə (tokenization)
    "punkt_tab",
    "stopwords",                      # "the", "is", "and" kimi mənasız sözlər
    "averaged_perceptron_tagger",     # POS (nitq hissəsi) təyini
    "averaged_perceptron_tagger_eng",
    "vader_lexicon",                  # hiss (sentiment) analizi üçün lüğət
    "maxent_ne_chunker",              # adlandırılmış varlıqların tanınması (NER)
    "maxent_ne_chunker_tab",
    "words",
]

print("NLTK data paketləri yüklənir, bir az vaxt apara bilər...\n")
for pkg in PACKAGES:
    try:
        nltk.download(pkg)
        print(f"[OK] {pkg}")
    except Exception as e:
        print(f"[XƏTA] {pkg}: {e}")

print("\nHazırdır! İndi 'python main.py' əmri ilə layihəni işə sala bilərsiniz.")