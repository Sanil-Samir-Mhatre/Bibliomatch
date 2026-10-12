import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="BiblioMatch - NeuMF Recommender", layout="wide")

st.title("BiblioMatch - NeuMF Recommender (PyTorch)")
st.markdown("This app serves recommendations using the trained Neural Collaborative Filtering (NeuMF) PyTorch model.")

DATA_DIR = '.'

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, gmf_dim=16, mlp_dim=32):
        super(NeuMF, self).__init__()
        self.gmf_user_embedding = nn.Embedding(num_users, gmf_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, gmf_dim)
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_dim)
        self.mlp_layers = nn.Sequential(
            nn.Linear(mlp_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.prediction_layer = nn.Sequential(
            nn.Linear(gmf_dim + 16, 1),
            nn.Sigmoid()
        )

    def forward(self, user_indices, item_indices):
        gmf_u = self.gmf_user_embedding(user_indices)
        gmf_i = self.gmf_item_embedding(item_indices)
        gmf_vector = gmf_u * gmf_i
        
        mlp_u = self.mlp_user_embedding(user_indices)
        mlp_i = self.mlp_item_embedding(item_indices)
        mlp_vector = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)
        
        predict_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        prediction = self.prediction_layer(predict_vector)
        return prediction.squeeze(-1)

@st.cache_resource
def load_data_and_model():
    try:
        data_path_for_csv = r"C:\Users\Admin\OneDrive\Desktop\BiblioMatch\10k\content_recommender\data"
        books = pd.read_csv(os.path.join(data_path_for_csv, 'books.csv'))
        ratings = pd.read_csv(os.path.join(data_path_for_csv, 'ratings.csv'))
        
        with open(os.path.join(DATA_DIR, 'user_id_index.pkl'), 'rb') as f:
            user_id_index = pickle.load(f)
        with open(os.path.join(DATA_DIR, 'book_id_index.pkl'), 'rb') as f:
            book_id_index = pickle.load(f)
            
        num_users = len(user_id_index)
        num_items = len(book_id_index)
        
        model = NeuMF(num_users, num_items)
        model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'neumf_model.pt'), map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        
        item_embeddings = model.mlp_item_embedding.weight.detach().numpy()
        
        return books, ratings, user_id_index, book_id_index, model, item_embeddings
    except Exception as e:
        return None, None, None, None, None, e

books, ratings, user_id_index, book_id_index, model, item_embeddings = load_data_and_model()

if model is None:
    st.warning(f"⚠️ Could not load model or dataset files. Make sure `neumf_model.pt` exists.")
    st.stop()

# Filter ratings to exactly match what model was trained on
user_counts = ratings['user_id'].value_counts()
book_counts = ratings['book_id'].value_counts()
valid_users = user_counts[user_counts >= 5].index
valid_books = book_counts[book_counts >= 10].index
ratings_filtered = ratings[ratings['user_id'].isin(valid_users) & ratings['book_id'].isin(valid_books)].copy()

def recommend_for_user(user_id, top_n=10):
    if user_id not in user_id_index:
        return [], "insufficient history for NCF"
    
    user_idx = user_id_index[user_id]
    user_rated_books = ratings_filtered[ratings_filtered['user_id'] == user_id]['book_id'].tolist()
    unrated_books = books[~books['book_id'].isin(user_rated_books)]
    
    unrated_book_ids = unrated_books['book_id'].tolist()
    unrated_book_idxs = [book_id_index[bid] for bid in unrated_book_ids if bid in book_id_index]
    valid_unrated_book_ids = [bid for bid in unrated_book_ids if bid in book_id_index]
    
    if not unrated_book_idxs:
        return []
    
    u_tensor = torch.tensor([user_idx] * len(unrated_book_idxs), dtype=torch.long)
    i_tensor = torch.tensor(unrated_book_idxs, dtype=torch.long)
    
    with torch.no_grad():
        preds = model(u_tensor, i_tensor).numpy()
    
    top_indices = preds.argsort()[::-1][:top_n]
    
    recommendations = []
    for idx in top_indices:
        b_id = valid_unrated_book_ids[idx]
        b_info = books[books['book_id'] == b_id].iloc[0]
        score = float(preds[idx])
        
        recommendations.append({
            'book_id': b_id,
            'title': b_info['title'],
            'authors': b_info['authors'],
            'ncf_score': score,
            'explanation': "Our neural model detected a non-linear taste pattern matching this book",
            'source': "ncf"
        })
    return recommendations

def recommend_similar_books(book_title, top_n=10):
    matched_books = books[books['title'].str.contains(book_title, case=False, na=False)]
    if matched_books.empty:
        return []
    
    b_id = matched_books.iloc[0]['book_id']
    if b_id not in book_id_index:
        return []
    
    b_idx = book_id_index[b_id]
    query_emb = item_embeddings[b_idx].reshape(1, -1)
    sim_scores = cosine_similarity(query_emb, item_embeddings).flatten()
    sim_scores[b_idx] = -1
    
    top_indices = sim_scores.argsort()[::-1][:top_n]
    inverse_book_id_index = {v: k for k, v in book_id_index.items()}
    
    recommendations = []
    for idx in top_indices:
        if idx not in inverse_book_id_index:
            continue
        rec_b_id = inverse_book_id_index[idx]
        b_info = books[books['book_id'] == rec_b_id].iloc[0]
        
        recommendations.append({
            'book_id': rec_b_id,
            'title': b_info['title'],
            'authors': b_info['authors'],
            'ncf_score': float(sim_scores[idx]),
            'explanation': f"Found similar learned features to {book_title}",
            'source': "ncf"
        })
    return recommendations

tab1, tab2 = st.tabs(["Personalized Recommendations", "Similar Books"])

with tab1:
    st.header("Get Recommendations for a User")
    sample_users = list(user_id_index.keys())[:50]
    selected_user = st.selectbox("Select User ID", sample_users)
    
    if st.button("Recommend for User"):
        with st.spinner("Generating NCF predictions..."):
            recs = recommend_for_user(selected_user, top_n=10)
            
            if isinstance(recs, tuple):
                st.error(recs[1]) 
            elif not recs:
                st.warning("No unrated books to recommend.")
            else:
                st.success("Recommendations Generated!")
                st.dataframe(pd.DataFrame(recs))

with tab2:
    st.header("Find Similar Books (Item-Item via MLP Embeddings)")
    search_query = st.text_input("Enter book title search term (e.g. 'Harry Potter')")
    
    if st.button("Find Similar Books"):
        if search_query:
            with st.spinner("Calculating embedding similarities..."):
                sim_recs = recommend_similar_books(search_query, top_n=10)
                if not sim_recs:
                    st.warning("Book not found or not in vocabulary.")
                else:
                    st.success("Similar books found!")
                    st.dataframe(pd.DataFrame(sim_recs))
        else:
            st.error("Please enter a search term.")
