import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'books.csv')
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
MODELS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

os.makedirs(ARTIFACTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
df['authors'] = df['authors'].fillna('')
df['title'] = df['title'].fillna('')
df['original_title'] = df['original_title'].fillna('')

# Combine text for tfidf
df['combined_features'] = df['authors'] + ' ' + df['title'] + ' ' + df['original_title']

# TFIDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Save artifacts
# tfidf_vectorizer
with open(os.path.join(ARTIFACTS_PATH, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)
with open(os.path.join(MODELS_PATH, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)

# tfidf_matrix
with open(os.path.join(ARTIFACTS_PATH, 'tfidf_matrix.pkl'), 'wb') as f:
    pickle.dump(tfidf_matrix, f)
import scipy.sparse
scipy.sparse.save_npz(os.path.join(MODELS_PATH, 'tfidf_matrix.npz'), tfidf_matrix)

# books_indexed
df_indexed = df[['title', 'authors', 'average_rating', 'original_publication_year']].copy()
# Add a dummy category to prevent app.py from breaking if we don't change app.py too much,
# or we just save df_indexed.
with open(os.path.join(ARTIFACTS_PATH, 'books_indexed.pkl'), 'wb') as f:
    pickle.dump(df_indexed, f)
with open(os.path.join(MODELS_PATH, 'books_indexed.pkl'), 'wb') as f:
    pickle.dump(df_indexed, f)

print("Training completed and artifacts saved.")
