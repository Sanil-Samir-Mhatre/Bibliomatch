import pickle
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), 'artifacts')

def load_artifacts():
    with open(os.path.join(ARTIFACTS_PATH, 'tfidf_vectorizer.pkl'), 'rb') as f: v = pickle.load(f)
    with open(os.path.join(ARTIFACTS_PATH, 'tfidf_matrix.pkl'), 'rb') as f: m = pickle.load(f)
    with open(os.path.join(ARTIFACTS_PATH, 'books_indexed.pkl'), 'rb') as f: d = pickle.load(f)
    return v, m, d

def recommend_from_books(book_titles, top_n=10):
    v, m, d = load_artifacts()
    matches = d[d['title'].isin(book_titles)]
    input_pos = [d.index.get_loc(i) for i in matches.index]
    if not input_pos: return []
    
    # Calculate mean vector of selected books
    q = np.asarray(m[input_pos].mean(axis=0))
    sim = cosine_similarity(q, m).flatten()
    
    # Exclude input books
    for p in input_pos: sim[p] = -1
    
    # Get top matches
    c_pos = sim.argsort()[-(top_n*3):][::-1]
    cands = d.iloc[c_pos].copy()
    cands['sim_score'] = sim[c_pos]
    
    return cands.sort_values(['average_rating', 'sim_score'], ascending=False).head(top_n).to_dict('records')
