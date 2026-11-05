# BiblioMatch Reader App

This is a separate reader-facing app; the lab walkthrough remains available in `10k_app.py`.

## Run locally

```powershell
pip install -r requirements-book-lovers.txt
streamlit run book_lovers_app.py
```

## Models in the app

- **Content TF-IDF:** ranks titles by cosine similarity to selected favorites using the saved lab artifacts.
- **KNN collaborative:** finds similar readers in the ratings matrix and recommends books from their ratings. Rate books on My shelf to give a profile a reading history.
- **SVD matrix factorization:** uses the Task 6 style 100,000-rating sample to predict scores for a linked Goodbooks reader.
- **Popularity regression:** runs the lab's LinearRegression features (rating count, review count, publication year) to rank predicted average ratings.
- **NeuMF deep learning:** loads the saved PyTorch GMF + MLP checkpoint and ranks unrated books for a linked Goodbooks reader.
- **NeuMF similar-item:** compares learned MLP item embeddings to find books similar to selected favorites.
- **Rule-based inference:** applies the Practical 2 production rules and provides a rule-backed catalog ranking.
- **Hybrid ensemble:** averages min-max-normalized outputs from the available rule, TF-IDF, KNN, SVD, regression, and NeuMF paths; components without usable profile data are omitted and identified in the result explanation.
- **Classical evaluation:** shows the Task 6 benchmark values recorded in the lab app, clearly labeled as static reference values rather than a fresh run.
- **NeuMF optimization:** runs the Task 7 dynamic INT8 quantization comparison and reports measured size, CPU latency, and prediction change; this mode does not produce book recommendations.
- **Popularity regression:** is presented as catalog rating prediction with filters and feature contribution details, not as personalized reader recommendations.

In Discover, choosing a mode first shows its method, data inputs, and output type. Recommendation modes then show their own controls and results. Classical evaluation and NeuMF optimization instead show metrics/benchmarks because those lab functions are not recommenders.

Every recommendation has a plain-language explanation and responsible-use note, with technical evidence available on request. For the popularity rating regression, the app includes a global SHAP chart over a reproducible catalog sample and a per-book LIME chart for an individual rating estimate. These charts explain that regression model only; they do not claim to explain the separate collaborative, content, or neural ranking models. Exact TF-IDF shared-term contributions and Practical 2 rule traces remain available in technical details. The recommendation graph traces selected favorites to recommendations, content similarity, authorship, rule matches, linked-reader rating history, and anonymized existing Goodbooks readers who rated both a favorite and the recommendation highly.

Each displayed list is transparently re-ranked using candidate-pool model rank (85%), catalog popularity percentile dampening (15%), and a lead-author repetition penalty (0.08 per previous pick). The app reports that this is an exposure/diversity adjustment, not demographic fairness certification; the dataset does not provide suitable protected-group labels. Existing Goodbooks reader IDs can be attached to a local profile to run KNN, SVD, and NeuMF from the actual dataset.

The app supports three reader paths: **existing Goodbooks readers** from the left-side selector; **new cold-start readers** with no history; and **readers with books**, who choose titles in Discover and can save them as rated history. Existing-reader history appears directly in Discover and My shelf; reader-with-books profiles keep selected titles locally and can use them for content and collaborative models. New readers can start with rule-based or catalog-based discovery, then add titles when ready. Profile and shelf data persist locally in `book_lovers_profiles.sqlite3`.

The price comparison uses SerpApi. The local key is read from `.streamlit/secrets.toml`, which is excluded from Git; use this format if configuring another machine:

```toml
SERPAPI_KEY = "your-key"
```
