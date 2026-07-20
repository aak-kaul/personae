"""
Personae — Flask web application.
Upload a customer CSV -> automated segmentation -> interactive PCA scatter
+ LLM-style grounded personas.

Run:
    pip install -r requirements.txt
    python app.py
    open http://127.0.0.1:5000
"""
import io
import os
import json
import traceback

import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, redirect, url_for,
                   flash, session)

import personae_pipeline as pipe
import persona_generator as pg

app = Flask(__name__)
app.secret_key = "personae-demo-key"
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "CC_GENERAL.csv")

# very small in-memory cache so results survive the render round-trip
_LAST = {}


def analyze(df, k=None):
    result = pipe.run(df, k=k)
    personas = pg.generate_all(result)
    personas_by_seg = {p["segment"]: p for p in personas}

    # build scatter payload
    coords = result["coords"]
    labels = result["labels"]
    scatter = [{"x": round(float(coords[i, 0]), 3),
                "y": round(float(coords[i, 1]), 3),
                "s": int(labels[i])} for i in range(len(labels))]

    # segment table columns: use the well-known credit-card headline when that
    # schema is present, otherwise the features that most distinguish the
    # segments in whatever data was uploaded (largest spread of segment means).
    cc_headline = [f for f in ["PURCHASES", "CASH_ADVANCE", "BALANCE",
                               "PURCHASES_FREQUENCY", "CREDIT_LIMIT", "PAYMENTS"]
                   if f in result["means"].columns]
    if len(cc_headline) >= 4:
        headline = cc_headline
    else:
        spread = (result["means"].std(ddof=0)
                  / result["overall_std"].replace(0, 1e-9))
        headline = list(spread.sort_values(ascending=False).head(6).index)
    seg_table = []
    for seg in result["means"].index:
        row = {"segment": int(seg), "n": int(result["sizes"][seg]),
               "pct": round(100 * result["sizes"][seg] / result["sizes"].sum(), 1),
               "name": personas_by_seg[seg]["name"],
               "color": personas_by_seg[seg]["color"]}
        for f in headline:
            row[f] = pg._fmt_num(result["means"].loc[seg, f])
        seg_table.append(row)

    return {
        "n_rows": result["n_rows"],
        "n_features": len(result["features"]),
        "chosen_k": result["chosen_k"],
        "best_k": result["best_k"],
        "pca_n90": result["pca_n90"],
        "pca_cum10": round(result["pca_cum10"], 1),
        "pca_pc12": round(result["pca_pc12"], 1),
        "missing": result["missing"],
        "sweep": result["sweep"],
        "personas": personas,
        "scatter": scatter,
        "seg_table": seg_table,
        "headline": headline,
    }


@app.route("/")
def index():
    return render_template("index.html", has_sample=os.path.exists(SAMPLE_PATH))


@app.route("/analyze", methods=["POST"])
def do_analyze():
    try:
        use_sample = request.form.get("use_sample") == "1"
        k = request.form.get("k")
        k = int(k) if k and k.isdigit() else None

        if use_sample:
            if not os.path.exists(SAMPLE_PATH):
                flash("Sample data not found on server.")
                return redirect(url_for("index"))
            df = pd.read_csv(SAMPLE_PATH)
        else:
            file = request.files.get("file")
            if not file or file.filename == "":
                flash("Please choose a CSV file to upload.")
                return redirect(url_for("index"))
            df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8", "ignore")))

        out = analyze(df, k=k)
        _LAST["out"] = out
        return render_template("results.html", d=out,
                               scatter_json=json.dumps(out["scatter"]),
                               sweep_json=json.dumps(out["sweep"]))
    except Exception as e:
        traceback.print_exc()
        flash(f"Could not analyze that file: {e}")
        return redirect(url_for("index"))


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
