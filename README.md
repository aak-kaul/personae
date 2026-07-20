# Personae — Automated Customer Segmentation with LLM-Generated Personas

A web application that discovers natural customer groups from a raw behavioral
dataset and describes each group as an actionable marketing persona. Upload any
numeric customer CSV and get labeled segments, an interactive PCA map, and a
grounded persona (name, description, key stats, and a marketing play) per group.

**Pipeline:** median imputation → standardization → PCA → K-means (k chosen by
elbow + silhouette) → persona generation. Works on the credit-card demo dataset
*and* generalizes to any numeric schema (it drops ID columns and clusters on
whatever numeric features are present).

Built by Jiajun Huang, Aakriti Kaul, and Nicholas Kaplun for the MSAI Machine
Learning (Summer 2026) final project.

---

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000  (use --port if 5000 is taken on macOS AirPlay)
```

Click **"Try it with the sample dataset"** or upload your own CSV.

---

## Deploy

The same repository deploys to either platform below. For a scikit-learn app,
**Render is the more reliable choice**; Vercel also works but can hit its
serverless size limit because scientific libraries are large.

### Option A — Render (recommended, runs Flask natively)

1. Push this repo to GitHub (see steps in the project chat/notes).
2. Go to [render.com](https://render.com) → **New → Web Service** → connect the repo.
3. Settings: **Build Command** `pip install -r requirements.txt`,
   **Start Command** `gunicorn app:app`. Environment: **Python 3**. Instance: **Free**.
4. Click **Create Web Service**. Render builds and gives you a public
   `https://<name>.onrender.com` URL.

### Option B — Vercel

1. Push this repo to GitHub.
2. Go to [vercel.com](https://vercel.com) → **Add New → Project** → import the repo.
3. Framework preset: **Other**. Leave build settings default — `vercel.json`
   already configures the Python build. Click **Deploy**.
4. Vercel gives you a public `https://<name>.vercel.app` URL.

> If the Vercel build fails citing a size/`250mb` limit, that is the known
> scikit-learn constraint — deploy on Render (Option A) instead.

---

## Project structure

```
app.py                 Flask routes and orchestration
personae_pipeline.py   preprocess → PCA → K-means (+ k selection)
persona_generator.py   segment profile → grounded persona (credit-card + generic modes)
templates/             index + results pages
static/                stylesheet
sample_data/           the Kaggle Credit Card dataset
vercel.json            Vercel Python build config
Procfile               start command for Render / Railway / Heroku
requirements.txt       pinned dependencies
```

## Data

Credit Card Dataset for Clustering — A. Bhasin, Kaggle, 2019.
<https://www.kaggle.com/datasets/arjunbhasin2013/ccdata>
