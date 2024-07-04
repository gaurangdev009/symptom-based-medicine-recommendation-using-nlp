from flask import Flask, render_template, request, redirect, session
import os
import joblib
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

app = Flask(__name__)

app.secret_key = os.urandom(24)
# Model saved with Keras model.save()
MODEL_PATH = "model/passmodel.pkl"

TOKENIZER_PATH = "model/tfidfvectorizer.pkl"

DATA_PATH = "data/drugsComTrain.csv"

# loading vectorizer
vectorizer = joblib.load(TOKENIZER_PATH)
# loading model
model = joblib.load(MODEL_PATH)

# getting stopwords
stop = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


@app.route("/")
def login():
	
    return render_template("index.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/login")
def first():
    return render_template("form_edited.html")


@app.route("/home2")
def home2():
    return render_template("index.html")


@app.route("/index")
def index():
    if "user_id" in session:

        return render_template("home.html")
    else:
        return redirect("/")


# ______________________--------------------------->
# home to other page route---------->


@app.route("/home_about")
def home_to_about():
    return render_template("about.html")


@app.route("/home_SE")
def seasonal():
    return render_template("seasonal.html")


@app.route("/home_doc")
def doc():
    return render_template("doctors.html")


@app.route("/home_bt")
def brain():
    return render_template("bt.html")


@app.route("/home_contact")
def contact():
    return render_template("contact.html")


@app.route("/home_home")
def home_to_home():
    return render_template("index.html")


# ---------------------------------------------------------------->
# ----------------------- about to other page -------------------->


@app.route("/about_home")
def about_to_home():
    return render_template("index.html")


@app.route("/about_SE")
def about_to_se():
    return render_template("seasonal.html")


@app.route("/about_DOC")
def about_to_doc():
    return render_template("doctors.html")


@app.route("/about_contact")
def about_to_contact():
    return render_template("contact.html")


@app.route("/about_DR")
def about_to_dr():
    return render_template("form_edited.html")


@app.route("/about_BT")
def about_to_bt():
    return render_template("bt.html")


@app.route("/about_about")
def about_to_about():
    return render_template("about.html")


# --------------------------------------------------------------->
# --------------- seasonal to other ----------------------------->


@app.route("/seasonal_seasonal")
def seasonal_to_seasonal():
    return render_template("seasonal.html")


@app.route("/seasonal_home")
def seasonal_to_home():
    return render_template("index.html")


@app.route("/seasonal_about")
def seasonal_to_about():
    return render_template("about.html")


@app.route("/seasonal_DOC")
def seasonal_to_doc():
    return render_template("doctors.html")


@app.route("/seasonal_DR")
def seasonal_to_dr():
    return render_template("form_edited.html")


@app.route("/seasonal_BT")
def seasonal_to_bt():
    return render_template("bt.html")


@app.route("/seasonal_contact")
def seasonal_to_contact():
    return render_template("contact.html")


@app.route("/seasonal_details")
def seasonal_to_detail():
    return render_template("seasonaldetails.html")


# -------------------------------------------------------------->
# -------------------- doctors to others ----------------------->


@app.route("/DOC_DOC")
def doctor_to_doctor():
    return render_template("doctors.html")


@app.route("/DOC_home")
def doctor_to_home():
    return render_template("index.html")


@app.route("/DOC_about")
def doctor_to_about():
    return render_template("about.html")


@app.route("/DOC_DR")
def doctor_to_dr():
    return render_template("form_edited.html")


@app.route("/DOC_BT")
def doctor_to_bt():
    return render_template("bt.html")


@app.route("/DOC_contact")
def doctor_to_contact():
    return render_template("contact.html")


@app.route("/DOC_SE")
def doctor_to_se():
    return render_template("Seasonal.html")


# ------------------------------------------------------->
# --------------------- BT to home --------------------------->


@app.route("/BT_home")
def BT_to_home():
    return render_template("index.html")


# ------------------------------------------------------------->
# ---------------------- contact to others ------------------------->


@app.route("/contact_contact")
def contact_to_contact():
    return render_template("contact.html")


@app.route("/contact_home")
def contact_to_home():
    return render_template("index.html")


@app.route("/contact_about")
def contact_to_about():
    return render_template("about.html")


@app.route("/contact_SE")
def contact_to_se():
    return render_template("seasonal.html")


@app.route("/contact_BT")
def contact_to_bt():
    return render_template("BT.html")


@app.route("/contact_DR")
def contact_to_dr():
    return render_template("form_edited.html")


@app.route("/contact_Doc")
def contact_to_doc():
    return render_template("doctors.html")


@app.route("/login_validation", methods=["POST"])
def login_validation():
    username = request.form.get("username")
    password = request.form.get("password")

    session["user_id"] = username
    session["domain"] = password

    if username == "admin@gmail.com" and password == "admin":

        return render_template("home.html")
        # return render_template('login_new.html', data=payload)

    else:

        err = "Kindly Enter valid User ID/ Password"
        return render_template("login_new.html", lbl=err)

    return ""


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        raw_text = request.form["rawtext"]

        if raw_text != "":
            clean_text = cleanText(raw_text)
            clean_lst = [clean_text]

            tfidf_vect = vectorizer.transform(clean_lst)
            prediction = model.predict(tfidf_vect)
            predicted_cond = prediction[0]
            df = pd.read_csv(DATA_PATH)
            top_drugs = top_drugs_extractor(predicted_cond, df)

            return render_template(
                "predict.html",
                rawtext=raw_text,
                result=predicted_cond,
                top_drugs=top_drugs,
            )
        else:
            raw_text = "There is no text to select"


def cleanText(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    # 2. Make a space
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return " ".join(lemmitize_words)


def top_drugs_extractor(condition, df):
    df_top = df[(df["rating"] >= 9) & (df["usefulCount"] >= 100)].sort_values(
        by=["rating", "usefulCount"], ascending=[False, False]
    )
    drug_lst = df_top[df_top["condition"] == condition]["drugName"].head(3).tolist()
    return drug_lst


if __name__ == "__main__":

    app.run(debug=True, host="localhost", port=8080)
