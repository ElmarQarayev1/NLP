"""
app.py
------
Layihənin VEB versiyası. İstifadəçi brauzerdə mətn daxil edir,
"Analiz et" düyməsini basır və nəticələri (qrafiklər daxil)
birbaşa səhifədə görür.

İşə salmaq:
    python app.py

Sonra brauzerdə açın:
    http://127.0.0.1:5000
"""

from flask import Flask, render_template, request

from nlp_utils import full_analysis

app = Flask(__name__)

# Sadə nümunə mətn - forma ilk açılanda köməkçi olsun deyə
EXAMPLE_TEXT = (
    "Last summer, Sarah decided to travel across Europe with her best friend Emma. "
    "They started their journey in Paris, where they visited the Eiffel Tower and "
    "enjoyed delicious French pastries every morning. The weather was absolutely "
    "wonderful, and the city felt magical at night."
)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    text = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if not text:
            error = "Zəhmət olmasa analiz üçün mətn daxil edin."
        elif len(text) < 20:
            error = "Mətn çox qısadır — mənalı nəticə üçün ən azı bir neçə cümlə yazın."
        else:
            try:
                result = full_analysis(text)
            except Exception as e:
                error = f"Analiz zamanı xəta baş verdi: {e}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        text=text,
        example_text=EXAMPLE_TEXT,
    )


if __name__ == "__main__":
    app.run(debug=True)