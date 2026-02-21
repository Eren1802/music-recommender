from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("top50.csv").fillna("")
df["features"] = df["artist"] + " " + df["composer"] + " " + df["album"]

vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(df["features"])
similarity = cosine_similarity(matrix)

def recommend(song_name):
    song_name = song_name.lower().strip()
    df["track_lower"] = df["track"].str.lower().str.strip()

    if song_name not in df["track_lower"].values:
        return ["Song not found"]

    idx = df[df["track_lower"] == song_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    return [df.iloc[i[0]]["track"] for i in scores]

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        song = request.form["song"]
        recommendations = recommend(song)
    return render_template("index.html", recs=recommendations)

if __name__ == "__main__":
    app.run(debug=True)