"""Reader-facing BiblioMatch app using the lab's actual ML and DL artifacts."""
from pathlib import Path
from html import escape
import pickle
import sqlite3
import io
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "10k" / "content_recommender" / "data"
ARTIFACTS = ROOT / "10k" / "content_recommender" / "artifacts"
NCF = ROOT / "Deep Learning Model 10k"
PROFILE_DB = ROOT / "book_lovers_profiles.sqlite3"

st.set_page_config(page_title="BiblioMatch | Reader's corner", page_icon="📚", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
.stApp{background:linear-gradient(145deg,#fbf7ef,#f4efe5 58%,#f8f5ee);color:#282b25}
h1,h2,h3{font-family:'Playfair Display',serif!important;color:#26382e}
.hero{padding:2rem 2.4rem;border-radius:24px;background:linear-gradient(120deg,#243c32,#55704f);color:white;margin:.4rem 0 1.4rem}
.hero .brand-title{font-family:'Playfair Display',serif;font-size:clamp(2.6rem,5vw,4rem);font-weight:700;letter-spacing:.025em;line-height:1.05;color:#fff}.hero .brand-subtitle{font-size:.92rem;font-weight:600;letter-spacing:.18em;color:#dec39b;margin:.3rem 0 .7rem}.hero h1{color:white!important;font-size:2.15rem;margin:.2rem 0}.hero p{color:#e5eadf;font-size:1rem;margin:0}
.book-card{padding:1rem;border-radius:16px;background:#fffdf8;border:1px solid #e9e2d5;min-height:135px;box-shadow:0 5px 18px #3935200a}
.book-card h3{font-size:1.1rem;margin:.3rem 0}.muted{color:#71776d;font-size:.9rem}.eyebrow{letter-spacing:.1em;text-transform:uppercase;color:#a36d3e;font-size:.72rem;font-weight:700}
div.stButton>button{border-radius:12px;border:1px solid #d4c9b7;background:#fffdf8;color:#2c4134}
</style>""", unsafe_allow_html=True)

def db():
    conn = sqlite3.connect(PROFILE_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS profiles (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, avatar TEXT NOT NULL DEFAULT '📚', lab_user_id INTEGER)")
    profile_columns={row[1] for row in conn.execute("PRAGMA table_info(profiles)")}
    if "profile_type" not in profile_columns:
        conn.execute("ALTER TABLE profiles ADD COLUMN profile_type TEXT NOT NULL DEFAULT 'new'")
    conn.execute("CREATE TABLE IF NOT EXISTS saved_books (profile_id INTEGER, book_id INTEGER, status TEXT DEFAULT 'Want to read', rating INTEGER, PRIMARY KEY(profile_id,book_id))")
    conn.commit()
    return conn

def all_profiles():
    with db() as conn:
        return pd.read_sql_query("SELECT * FROM profiles ORDER BY id", conn)

def load_saved_books(profile_id):
    """Read shelf IDs from SQLite and join book metadata from the loaded catalog DataFrame."""
    with db() as conn:
        saved = pd.read_sql_query(
            "SELECT book_id,status,rating FROM saved_books WHERE profile_id=? ORDER BY rowid DESC",
            conn,
            params=(int(profile_id),),
        )
    if saved.empty:
        return saved
    saved["book_id"] = pd.to_numeric(saved.book_id, errors="coerce").astype("Int64")
    metadata = books[["book_id","title","authors","image_url","average_rating","ratings_count"]].copy()
    metadata["book_id"] = pd.to_numeric(metadata.book_id, errors="coerce").astype("Int64")
    return saved.merge(metadata, on="book_id", how="left", validate="many_to_one")

def create_profile(name, avatar, lab_user_id=None, profile_type="new"):
    with db() as conn:
        conn.execute("INSERT INTO profiles(name,avatar,lab_user_id,profile_type) VALUES(?,?,?,?)", (name.strip(), avatar, lab_user_id, profile_type))

@st.cache_data(show_spinner="Opening the BiblioMatch library…")
def load_data():
    books = pd.read_csv(DATA / "books.csv")
    ratings = pd.read_csv(DATA / "ratings.csv")
    with open(ARTIFACTS / "tfidf_matrix.pkl", "rb") as f: tfidf = pickle.load(f)
    with open(ARTIFACTS / "books_indexed.pkl", "rb") as f: indexed = pickle.load(f)
    with open(ARTIFACTS / "tfidf_vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
    # The lab artifact intentionally omits book_id. Its rows are saved in
    # books.csv order, which is also the TF-IDF matrix order; restore IDs only
    # after checking all three row sequences still align.
    if len(indexed)!=len(books) or tfidf.shape[0]!=len(books):
        raise ValueError("Books, TF-IDF rows, and the saved index have different lengths; rebuild the content artifacts from the current books.csv.")
    for column in ["title","authors"]:
        source=books[column].fillna("").astype(str).reset_index(drop=True)
        artifact=indexed[column].fillna("").astype(str).reset_index(drop=True)
        if not source.equals(artifact):
            raise ValueError(f"The saved TF-IDF index is not aligned with books.csv on {column}; rebuild the content artifacts before using book explanations.")
    indexed=indexed.copy()
    indexed["book_id"]=books["book_id"].to_numpy()
    return books, ratings, indexed, tfidf, vectorizer

books, ratings, indexed, tfidf, vectorizer = load_data()
books["book_id"] = pd.to_numeric(books.book_id, errors="coerce").astype("Int64")

# Profile menu: profile icon in the sidebar, quick switching, and create flow.
profiles = all_profiles()
with st.sidebar:
    st.markdown("## 👤 Reader profile")
    if profiles.empty:
        st.caption("Create a reader profile to save books and preferences.")
        active_id = None
    else:
        options = profiles.id.tolist()
        default_id = st.session_state.get("active_profile_id", options[0])
        if default_id not in options: default_id = options[0]
        active_id = st.selectbox("Switch profile", options, index=options.index(default_id), format_func=lambda x: f"{profiles.loc[profiles.id.eq(x),'avatar'].iloc[0]}  {profiles.loc[profiles.id.eq(x),'name'].iloc[0]} · {('Goodbooks' if pd.notna(profiles.loc[profiles.id.eq(x),'lab_user_id'].iloc[0]) else 'Reader with books' if profiles.loc[profiles.id.eq(x),'profile_type'].iloc[0]=='reader' else 'New reader')}", label_visibility="collapsed")
        if st.session_state.get("active_profile_id") not in (None, active_id):
            st.session_state.pop("last_recs", None)
        st.session_state.active_profile_id = active_id
        if st.button("＋ Create another profile", use_container_width=True): st.session_state.show_profile_form = True
    if profiles.empty or st.session_state.get("show_profile_form"):
        with st.form("create_profile", clear_on_submit=True):
            st.markdown("**Create a reader profile**")
            name = st.text_input("Name", max_chars=40)
            avatar = st.selectbox("Choose an avatar", ["📚","🌿","🦊","🐈","🌙","☕","✨","🪴"])
            profile_type = st.radio("Choose the starting point",["new","reader"],format_func=lambda kind:"New reader · no history yet" if kind=="new" else "Reader with books to add",horizontal=True)
            if st.form_submit_button("Create profile", type="primary"):
                if not name.strip(): st.warning("Add a name for this profile.")
                else:
                    try:
                        create_profile(name, avatar, profile_type=profile_type)
                        st.session_state.show_profile_form = False
                        st.rerun()
                    except sqlite3.IntegrityError: st.error("That profile name already exists.")
    if active_id is not None:
        active = profiles[profiles.id.eq(active_id)].iloc[0]
        active_kind="Existing Goodbooks reader" if pd.notna(active.lab_user_id) else ("Reader with books" if active.profile_type=="reader" else "New reader · cold start")
        st.caption(f"Currently reading as **{active.avatar} {active['name']}** · {active_kind}")

    st.markdown("### Existing Goodbooks readers")
    st.caption("Switch directly to a reader already present in the lab ratings data.")
    sidebar_user_counts=ratings.head(100000).groupby("user_id").size().sort_values(ascending=False)
    sidebar_user_options=sidebar_user_counts.index.tolist()
    prior_sidebar_user=st.session_state.get("sidebar_lab_user_id")
    if prior_sidebar_user not in sidebar_user_options:
        linked_current=profiles.loc[profiles.id.eq(active_id),"lab_user_id"] if active_id is not None else pd.Series(dtype=float)
        prior_sidebar_user=int(linked_current.iloc[0]) if not linked_current.empty and pd.notna(linked_current.iloc[0]) and int(linked_current.iloc[0]) in sidebar_user_options else sidebar_user_options[0]
    sidebar_lab_user=st.selectbox(
        "Select an existing reader",
        sidebar_user_options,
        index=sidebar_user_options.index(prior_sidebar_user),
        format_func=lambda user_id:f"Reader {user_id} · {int(sidebar_user_counts.loc[user_id]):,} ratings",
        key="sidebar_lab_user_id",
    )
    if st.button("Switch to this existing reader",use_container_width=True,key="switch_existing_reader"):
        with db() as conn:
            linked_row=conn.execute("SELECT id FROM profiles WHERE lab_user_id=? ORDER BY id LIMIT 1",(int(sidebar_lab_user),)).fetchone()
            if linked_row:
                chosen_profile_id=linked_row[0]
                conn.execute("UPDATE profiles SET profile_type='goodbooks' WHERE id=?",(chosen_profile_id,))
            else:
                profile_name=f"Goodbooks Reader {int(sidebar_lab_user)}"
                existing_name=conn.execute("SELECT id FROM profiles WHERE name=?",(profile_name,)).fetchone()
                if existing_name:
                    chosen_profile_id=existing_name[0]
                    conn.execute("UPDATE profiles SET lab_user_id=?,profile_type='goodbooks' WHERE id=?",(int(sidebar_lab_user),chosen_profile_id))
                else:
                    cursor=conn.execute("INSERT INTO profiles(name,avatar,lab_user_id,profile_type) VALUES(?,?,?,?)",(profile_name,"📖",int(sidebar_lab_user),"goodbooks"))
                    chosen_profile_id=cursor.lastrowid
        st.session_state.active_profile_id=chosen_profile_id
        st.session_state.pop("last_recs",None)
        st.rerun()

st.markdown("<div class='hero'><div class='brand-title'>BIBLIOMATCH</div><div class='brand-subtitle'>YOUR READING COMPANION</div><h1>Find your next favourite.</h1><p>Explore the full range of BiblioMatch ML and deep learning recommenders.</p></div>", unsafe_allow_html=True)
discover, models_tab, compare, shelf = st.tabs(["✨ Discover", "🧠 Recommendation models", "🛍️ Compare prices", "♡ My shelf"])

def get_saved_ids(profile_id):
    if profile_id is None: return set()
    with db() as conn:
        return {r[0] for r in conn.execute("SELECT book_id FROM saved_books WHERE profile_id=?", (int(profile_id),))}

def save_book(profile_id, bid, status="Want to read", rating=None):
    if profile_id is None:
        st.warning("Create or select a reader profile first."); return
    with db() as conn:
        conn.execute("INSERT INTO saved_books(profile_id,book_id,status,rating) VALUES(?,?,?,?) ON CONFLICT(profile_id,book_id) DO UPDATE SET status=excluded.status,rating=excluded.rating", (int(profile_id), int(bid), status, rating))

def render_book(row, key, detail_context=None):
    bid = int(row.book_id)
    cols = st.columns([.8, 4, 1.2])
    with cols[0]:
        url = row.get("image_url")
        if isinstance(url, str) and url.startswith("http"): st.image(url, width=88)
    with cols[1]:
        title = escape(str(row.get("title", "Untitled"))); author = escape(str(row.get("authors", "Unknown author")))
        rating = pd.to_numeric(row.get("average_rating", 0), errors="coerce")
        model_tag=f"{detail_context[0]} · recommendation" if detail_context else "Your saved book"
        st.markdown(f"<div class='book-card'><div class='eyebrow'>{escape(model_tag)}</div><h3>{title}</h3><div class='muted'>{author}</div><div class='muted'>⭐ {rating:.2f} · {int(row.get('ratings_count',0) or 0):,} ratings</div></div>", unsafe_allow_html=True)
    with cols[2]:
        saved = get_saved_ids(active_id)
        if bid in saved: st.caption("♥ On your shelf")
        elif st.button("♡ Save", key=f"save_{key}_{bid}", use_container_width=True): save_book(active_id, bid); st.rerun()
    if detail_context:
        model_name, seeds, result_set = detail_context
        explain_result(row, model_name, seeds, result_set)

def content_recs(seeds, limit=10):
    found = indexed[indexed.title.isin(seeds)]
    if found.empty: return pd.DataFrame()
    positions = [indexed.index.get_loc(idx) for idx in found.index]
    sims = cosine_similarity(np.asarray(tfidf[positions].mean(axis=0)), tfidf).ravel()
    sims[positions] = -1
    order = sims.argsort()[::-1][:limit]
    result = indexed.iloc[order].copy(); result["model_score"] = sims[order]
    return result.sort_values(["model_score","average_rating"], ascending=False)

def ensure_ratings_matrix():
    @st.cache_resource
    def build(frame):
        # User-book sparse matrix used by the lab's collaborative filtering route.
        u = pd.Categorical(frame.user_id); b = pd.Categorical(frame.book_id)
        matrix = coo_matrix((frame.rating.to_numpy(dtype=np.float32),(u.codes,b.codes))).tocsr()
        return matrix, u.categories, b.categories
    return build(ratings)

@st.cache_resource
def svd_factors():
    # Same 100k interaction sample and rank cap as Task 6 in 10k_app.py.
    frame = ratings.head(100000).copy()
    u = pd.Categorical(frame.user_id); b = pd.Categorical(frame.book_id)
    matrix = coo_matrix((frame.rating.to_numpy(dtype=np.float32),(u.codes,b.codes))).asfptype().tocsr()
    k = min(50, min(matrix.shape)-1)
    if k < 1: raise ValueError("Not enough ratings to fit SVD.")
    U, sigma, Vt = svds(matrix, k=k)
    return (U * sigma) @ Vt, u.categories, b.categories

@st.cache_resource
def regression_model():
    cols=["ratings_count","work_text_reviews_count","original_publication_year"]
    frame=books.dropna(subset=cols+["average_rating"]).copy()
    model=LinearRegression().fit(frame[cols],frame.average_rating)
    frame["predicted_rating"]=model.predict(frame[cols])
    return frame, model

REGRESSION_FEATURES=["ratings_count","work_text_reviews_count","original_publication_year"]
REGRESSION_LABELS={"ratings_count":"Number of ratings","work_text_reviews_count":"Written reviews","original_publication_year":"Publication year"}

@st.cache_data(show_spinner=False)
def regression_shap_summary():
    """Global SHAP values for the real catalog rating regression model."""
    import shap
    frame,model=regression_model()
    sample=frame[REGRESSION_FEATURES].sample(min(500,len(frame)),random_state=42)
    explainer=shap.LinearExplainer(model,frame[REGRESSION_FEATURES].sample(min(100,len(frame)),random_state=7))
    values=explainer.shap_values(sample)
    return pd.DataFrame({"Feature":[REGRESSION_LABELS[x] for x in REGRESSION_FEATURES],"Mean absolute SHAP value":np.abs(values).mean(axis=0)})

@st.cache_data(show_spinner=False)
def regression_lime(book_id):
    """Local LIME feature contributions for one book's rating estimate."""
    from lime.lime_tabular import LimeTabularExplainer
    frame,model=regression_model()
    match=frame.loc[frame.book_id.eq(int(book_id))]
    if match.empty: return pd.DataFrame()
    training=frame[REGRESSION_FEATURES].to_numpy(dtype=float)
    explainer=LimeTabularExplainer(training,feature_names=[REGRESSION_LABELS[x] for x in REGRESSION_FEATURES],mode="regression",discretize_continuous=True,random_state=42)
    explanation=explainer.explain_instance(match.iloc[0][REGRESSION_FEATURES].to_numpy(dtype=float),model.predict,num_features=len(REGRESSION_FEATURES),num_samples=300)
    return pd.DataFrame(explanation.as_list(),columns=["Feature","LIME contribution"])

def show_lime_chart(book_id):
    try:
        lime_values=regression_lime(int(book_id))
        if lime_values.empty:
            st.info("This book is not in the rating model’s training catalog, so a LIME chart is unavailable.")
        else:
            lime_values=lime_values.sort_values("LIME contribution")
            fig=px.bar(lime_values,x="LIME contribution",y="Feature",orientation="h",color="LIME contribution",color_continuous_scale=["#bd6b58","#e9e3d6","#2f6b4f"],title="LIME · this book’s estimated rating")
            fig.update_layout(showlegend=False,coloraxis_showscale=False,xaxis_title="Effect on estimated rating (model scale)",yaxis_title="")
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Positive bars raise and negative bars lower the popularity regression’s rating estimate. This is an individual book explanation; for other recommendation methods it gives rating context, not the reason that method ranked the book.")
    except ImportError:
        st.error("Install the optional explainability packages with `pip install shap lime` to view SHAP and LIME charts.")
    except Exception as exc:
        st.error(f"Could not create the LIME explanation: {exc}")

@st.cache_resource
def neural_model():
    import torch
    import torch.nn as nn
    class NeuMF(nn.Module):
        def __init__(self, num_users, num_items, gmf_dim=16, mlp_dim=32):
            super().__init__()
            self.gmf_user_embedding=nn.Embedding(num_users,gmf_dim); self.gmf_item_embedding=nn.Embedding(num_items,gmf_dim)
            self.mlp_user_embedding=nn.Embedding(num_users,mlp_dim); self.mlp_item_embedding=nn.Embedding(num_items,mlp_dim)
            self.mlp_layers=nn.Sequential(nn.Linear(mlp_dim*2,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
            self.prediction_layer=nn.Sequential(nn.Linear(gmf_dim+16,1),nn.Sigmoid())
        def forward(self, users, items):
            gmf=self.gmf_user_embedding(users)*self.gmf_item_embedding(items)
            mlp=self.mlp_layers(torch.cat([self.mlp_user_embedding(users),self.mlp_item_embedding(items)],dim=-1))
            return self.prediction_layer(torch.cat([gmf,mlp],dim=-1)).squeeze(-1)
    with open(NCF/"user_id_index.pkl","rb") as f: ui=pickle.load(f)
    with open(NCF/"book_id_index.pkl","rb") as f: bi=pickle.load(f)
    model=NeuMF(len(ui),len(bi))
    try: state=torch.load(NCF/"neumf_model.pt",map_location="cpu",weights_only=True)
    except TypeError: state=torch.load(NCF/"neumf_model.pt",map_location="cpu")
    model.load_state_dict(state); model.eval()
    return model,ui,bi

def user_history(profile_id):
    if profile_id is None: return pd.DataFrame(columns=["book_id","rating"])
    with db() as conn:
        return pd.read_sql_query("SELECT book_id,rating FROM saved_books WHERE profile_id=? AND rating IS NOT NULL",conn,params=(int(profile_id),))

def personalized_history(profile_id):
    own=user_history(profile_id)
    if not own.empty or profile_id is None:return own
    linked=profiles.loc[profiles.id.eq(profile_id),"lab_user_id"]
    if linked.empty or pd.isna(linked.iloc[0]):return own
    # Existing Goodbooks readers are first-class profiles for collaborative models.
    return ratings.loc[ratings.user_id.eq(int(linked.iloc[0])),["book_id","rating"]].copy()

def linked_lab_user_id(profile_id):
    if profile_id is None:return None
    linked=profiles.loc[profiles.id.eq(profile_id),"lab_user_id"]
    return int(linked.iloc[0]) if not linked.empty and pd.notna(linked.iloc[0]) else None

def goodbooks_read_history(profile_id):
    user_id=linked_lab_user_id(profile_id)
    if user_id is None:return pd.DataFrame()
    history=ratings.loc[ratings.user_id.eq(user_id),["book_id","rating"]].copy()
    history["book_id"]=pd.to_numeric(history.book_id,errors="coerce").astype("Int64")
    history=history.merge(books[["book_id","title","authors","image_url","average_rating","ratings_count"]],on="book_id",how="left",validate="many_to_one")
    return history.sort_values(["rating","title"],ascending=[False,True])

def knn_recommend(history, limit=10):
    if history.empty: return pd.DataFrame()
    matrix, users, item_ids = ensure_ratings_matrix()
    rated=set(history.book_id.astype(int)); item_pos={int(v):i for i,v in enumerate(item_ids)}
    seed_positions=[item_pos[x] for x in rated if x in item_pos]
    if not seed_positions: return pd.DataFrame()
    # KNN over users: find readers who rated this profile's favorite books highly.
    profile_vector=csr_matrix((1,matrix.shape[1]),dtype=np.float32)
    for bid, score in history[["book_id","rating"]].itertuples(index=False):
        if int(bid) in item_pos: profile_vector[0,item_pos[int(bid)]]=float(score)
    neighbors=NearestNeighbors(metric="cosine",algorithm="brute",n_neighbors=min(30,matrix.shape[0])).fit(matrix)
    distances, idx=neighbors.kneighbors(profile_vector)
    weights=(1-distances[0]).clip(0)
    pred=np.asarray(matrix[idx[0]].T @ weights).ravel() / max(weights.sum(),1e-8)
    rated_positions=[item_pos[x] for x in rated if x in item_pos]
    pred[rated_positions]=-1
    candidate_ids=[int(item_ids[i]) for i in pred.argsort()[::-1][:limit*5]]
    out=books[books.book_id.isin(candidate_ids)].copy(); out["model_score"]=out.book_id.map({int(item_ids[i]):float(pred[i]) for i in range(len(item_ids))})
    return out.sort_values("model_score",ascending=False).head(limit)

def svd_recommend(user_id, limit=10):
    preds, users, items=svd_factors()
    if user_id not in users: return pd.DataFrame()
    row=preds[list(users).index(user_id)].copy()
    seen=set(ratings.loc[ratings.user_id.eq(user_id),"book_id"].astype(int))
    item_to_score={int(b):float(row[i]) for i,b in enumerate(items) if int(b) not in seen}
    result=books[books.book_id.isin(item_to_score)].copy(); result["model_score"]=result.book_id.map(item_to_score)
    return result.sort_values("model_score",ascending=False).head(limit)

def neumf_recommend(user_id, limit=10):
    import torch
    model, users, item_map=neural_model()
    if user_id not in users: return pd.DataFrame()
    seen=set(ratings.loc[ratings.user_id.eq(user_id),"book_id"].astype(int))
    candidates=books[~books.book_id.isin(seen) & books.book_id.isin(item_map)].copy()
    if candidates.empty:return candidates
    user_idx=users[user_id]; ids=candidates.book_id.astype(int).tolist()
    item_indices=[item_map[x] for x in ids]
    with torch.no_grad(): scores=model(torch.tensor([user_idx]*len(ids)),torch.tensor(item_indices)).numpy()
    candidates["model_score"]=scores
    return candidates.sort_values("model_score",ascending=False).head(limit)

def neumf_similar(seeds, limit=10):
    from sklearn.metrics.pairwise import cosine_similarity as cosine
    if not seeds: return pd.DataFrame()
    model, _, item_map=neural_model()
    seed_rows=indexed[indexed.title.isin(seeds)]
    seed_ids=[int(x) for x in seed_rows.book_id if int(x) in item_map]
    if not seed_ids:return pd.DataFrame()
    seed_indices=[item_map[x] for x in seed_ids]
    embeddings=model.mlp_item_embedding.weight.detach().cpu().numpy()
    sims=cosine(embeddings[seed_indices].mean(axis=0,keepdims=True),embeddings).ravel()
    reverse={v:k for k,v in item_map.items()}
    for index in seed_indices:sims[index]=-1
    order=sims.argsort()[::-1][:limit*3]
    pairs={int(reverse[i]):float(sims[i]) for i in order if i in reverse}
    result=books[books.book_id.isin(pairs)].copy(); result["model_score"]=result.book_id.map(pairs)
    return result.sort_values("model_score",ascending=False).head(limit)

def rule_recommend(profile, limit=10):
    history_count=len(personalized_history(profile))
    frame=books.copy()
    rule_scores=[]; rule_labels=[]
    for _,row in frame.iterrows():
        matches=[(name,score) for name,score,_ in rule_trace(row,history_count) if not name.startswith("Rule 5")]
        if matches:
            label,score=max(matches,key=lambda x:x[1]); rule_scores.append(float(score)); rule_labels.append(label)
        else:
            rating=pd.to_numeric(row.get("average_rating"),errors="coerce")
            score=float(rating)/5 if pd.notna(rating) else 0.0
            rule_scores.append(score); rule_labels.append("General preference · rating fallback")
    frame["model_score"]=rule_scores; frame["rule_label"]=rule_labels
    return frame.sort_values(["model_score","average_rating","ratings_count"],ascending=False).head(limit)

@st.cache_resource
def quantized_neural_model():
    import torch
    model,users,items=neural_model()
    try:
        quantized=torch.ao.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8)
    except AttributeError:
        quantized=torch.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8)
    quantized.eval()
    return quantized,users,items

def benchmark_neumf(profile_id, repetitions=5):
    import torch
    user_id=linked_lab_user_id(profile_id)
    model,users,item_map=neural_model()
    if user_id is None or user_id not in users: return None
    optimized,_,_=quantized_neural_model()
    item_ids=list(item_map.keys())[:500]
    u=torch.tensor([users[user_id]]*len(item_ids),dtype=torch.long)
    i=torch.tensor([item_map[book_id] for book_id in item_ids],dtype=torch.long)
    def predict(network):
        with torch.no_grad():return network(u,i).detach().cpu().numpy()
    original=predict(model); compressed=predict(optimized)
    def size_mb(network):
        buffer=io.BytesIO(); torch.save(network.state_dict(),buffer); return buffer.getbuffer().nbytes/(1024*1024)
    def mean_latency(network):
        predict(network); times=[]
        for _ in range(repetitions):
            start=time.perf_counter(); predict(network); times.append(time.perf_counter()-start)
        return float(np.mean(times)*1000)
    original_size=size_mb(model); optimized_size=size_mb(optimized)
    original_ms=mean_latency(model); optimized_ms=mean_latency(optimized)
    return {"items_scored":len(item_ids),"original_size_mb":original_size,"quantized_size_mb":optimized_size,"size_reduction_pct":100*(original_size-optimized_size)/max(original_size,1e-9),"original_latency_ms":original_ms,"quantized_latency_ms":optimized_ms,"speed_change_pct":100*(original_ms-optimized_ms)/max(original_ms,1e-9),"mean_abs_prediction_delta":float(np.mean(np.abs(original-compressed)))}

def _recommend_candidates(model_name, seeds, profile, limit=8):
    history=personalized_history(profile)
    if model_name=="Rule-based inference": return rule_recommend(profile,limit)
    if model_name=="Content TF-IDF": return content_recs(seeds,limit)
    if model_name=="KNN collaborative": return knn_recommend(history,limit)
    if model_name=="SVD matrix factorization":
        linked=profiles.loc[profiles.id.eq(profile),"lab_user_id"]
        return svd_recommend(int(linked.iloc[0]),limit) if profile and not linked.empty and pd.notna(linked.iloc[0]) else pd.DataFrame()
    if model_name=="Popularity regression":
        frame,_=regression_model(); frame=frame.copy(); frame["model_score"]=frame["predicted_rating"]
        return frame.sort_values("model_score",ascending=False).head(limit)
    if model_name=="NeuMF similar-item": return neumf_similar(seeds,limit)
    if model_name=="NeuMF deep learning":
        linked=profiles.loc[profiles.id.eq(profile),"lab_user_id"]
        return neumf_recommend(int(linked.iloc[0]),limit) if profile and not linked.empty and pd.notna(linked.iloc[0]) else pd.DataFrame()
    return hybrid_recommend(seeds,profile,limit)

def responsibility_rerank(frame, limit):
    """Apply an inspectable long-tail and author-diversity adjustment to the raw model ranking."""
    if frame.empty: return frame
    out=frame.copy()
    raw=pd.to_numeric(out.get("model_score",pd.Series(0,index=out.index)),errors="coerce").fillna(0)
    out["model_rank_percentile"]=raw.rank(pct=True,method="average")
    catalog_counts=pd.to_numeric(books.ratings_count,errors="coerce").fillna(0)
    catalog_percentiles=catalog_counts.rank(pct=True)
    out["popularity_percentile"]=out.book_id.map(dict(zip(books.book_id.astype(int),catalog_percentiles))).fillna(0.5)
    out["responsible_score"]=0.85*out.model_rank_percentile-0.15*out.popularity_percentile
    # Greedily discourage a single author from occupying the whole short list.
    out["_author_group"]=out.authors.fillna("Unknown author").astype(str).str.split(",").str[0].str.strip()
    remaining=out.sort_values("responsible_score",ascending=False).copy(); chosen=[]; chosen_scores=[]; author_counts={}
    while not remaining.empty and len(chosen)<limit:
        remaining["_diversity_score"]=remaining.apply(lambda r:float(r.responsible_score)-0.08*author_counts.get(r["_author_group"],0),axis=1)
        pick=remaining.sort_values("_diversity_score",ascending=False).iloc[0]
        chosen.append(pick.name); chosen_scores.append(float(pick["_diversity_score"])); author=pick["_author_group"]; author_counts[author]=author_counts.get(author,0)+1
        remaining=remaining.drop(index=pick.name)
    out=out.loc[chosen].copy()
    out["responsible_score"]=chosen_scores
    out["author_repeat_in_list"]=out["_author_group"].map(author_counts).gt(1)
    out["list_position"]=range(1,len(out)+1)
    return out

def recommend(model_name, seeds, profile, limit=8):
    candidates=_recommend_candidates(model_name,seeds,profile,max(limit*5,limit))
    return responsibility_rerank(candidates,limit)

def rule_trace(row, history_count=0):
    rating=float(pd.to_numeric(row.get("average_rating",0),errors="coerce") or 0)
    count_value=pd.to_numeric(row.get("ratings_count",0),errors="coerce"); count=int(count_value) if pd.notna(count_value) else 0
    year=pd.to_numeric(row.get("original_publication_year"),errors="coerce")
    reviews_value=pd.to_numeric(row.get("work_text_reviews_count",0),errors="coerce"); reviews=int(reviews_value) if pd.notna(reviews_value) else 0
    fired=[]
    if pd.notna(year) and rating>=4.25 and year<=2000: fired.append(("Rule 1 · Masterpiece Classic",.95,f"rating {rating:.2f} ≥ 4.25 and year {int(year)} ≤ 2000"))
    elif pd.notna(year) and count>=500000 and year>2000: fired.append(("Rule 2 · Modern Blockbuster",.90,f"{count:,} ratings ≥ 500,000 and year {int(year)} > 2000"))
    elif rating>=4.15 and count<60000: fired.append(("Rule 3 · Hidden Gem",.85,f"rating {rating:.2f} ≥ 4.15 and {count:,} ratings < 60,000"))
    elif count>0 and reviews/count>=.04: fired.append(("Rule 4 · Thought Provoking",.88,f"review-to-rating ratio {reviews/count:.3f} ≥ 0.04"))
    if history_count==0: fired.append(("Rule 5 · Cold-start fallback",.80,"no profile rating history; use this catalog-quality signal as a fallback"))
    return fired

def shared_terms(seed_titles, target_row):
    target=indexed[indexed.book_id.eq(int(target_row.book_id))]
    if target.empty:return pd.DataFrame(), None
    tpos=indexed.index.get_loc(target.index[0]); best=None
    for title in seed_titles:
        source=indexed[indexed.title.eq(title)]
        if source.empty:continue
        spos=indexed.index.get_loc(source.index[0]); contributions=tfidf[spos].multiply(tfidf[tpos]).toarray().ravel()
        score=float(contributions.sum())
        if best is None or score>best[0]:best=(score,title,contributions)
    if best is None:return pd.DataFrame(),None
    score,title,contrib=best
    top=np.argsort(contrib)[::-1][:6]; terms=vectorizer.get_feature_names_out()
    table=pd.DataFrame({"Shared text feature":[terms[i] for i in top if contrib[i]>0],"Cosine contribution":[float(contrib[i]) for i in top if contrib[i]>0]})
    return table,(title,score)

def explain_result(row, model_name, seeds, result_set):
    title=str(row.get("title","this book"))
    with st.expander("Why this pick? · explanation & responsible use",expanded=False):
        seed_text = ", ".join(f"**{escape(str(seed))}**" for seed in seeds[:2])
        simple_reasons = {
            "Hybrid ensemble": f"Several available recommendation methods placed **{escape(title)}** near the top of the list. {('Your selected books, including ' + seed_text + ', helped shape the match.' ) if seeds else 'Because no reading history was selected, this is a discovery pick based on the available catalog signals.'}",
            "Rule-based inference": f"**{escape(title)}** matched a simple catalog rule: {escape(str(row.get('rule_label','a general discovery rule')))}.",
            "Popularity regression": f"**{escape(title)}** ranks well for the catalog details you selected. This is a general rating estimate, not a prediction tailored to your reading history.",
            "Content TF-IDF": f"**{escape(title)}** shares book or author information with your selected reading. {('It is most closely linked to ' + seed_text + '.' ) if seeds else 'Add a favorite to see what it has in common with this book.'}",
            "NeuMF similar-item": f"The book profile model places **{escape(title)}** near one of your selected books in its learned book map. This is a pattern match, rather than a guarantee about story or genre.",
            "KNN collaborative": f"Readers with rating patterns similar to this profile helped bring **{escape(title)}** into the list.",
            "SVD matrix factorization": f"Past reader–book rating patterns in the lab data helped rank **{escape(title)}** for this reader profile.",
            "NeuMF deep learning": f"The trained model found a strong pattern between this reader profile and **{escape(title)}** in past ratings.",
        }
        st.write(simple_reasons.get(model_name, f"**{escape(title)}** ranked highly using the information available to **{escape(model_name)}**."))
        st.markdown("**Individual LIME chart** · factors behind this book’s estimated average rating")
        show_lime_chart(int(row.book_id))

        pop_value=pd.to_numeric(row.get("popularity_percentile",0.5),errors="coerce")
        pop=float(pop_value) if pd.notna(pop_value) else 0.5
        count_value=pd.to_numeric(row.get("ratings_count",0),errors="coerce")
        count=int(count_value) if pd.notna(count_value) else 0
        repeat=bool(row.get("author_repeat_in_list",False))
        exposure="widely rated" if pop>=.9 else ("fairly popular" if pop>=.7 else "less widely rated")
        evidence="limited reader feedback" if count<25 else ("some reader feedback" if count<500 else "more reader feedback")
        st.markdown(f"**A note on this suggestion:** It is {exposure} ({count:,} ratings), so it has {evidence}. {'The same author appears elsewhere in this list.' if repeat else 'This list does not repeat this author.'} Recommendations can reflect gaps in the books and ratings available to the model; they are suggestions, not a measure of a book’s quality or a promise you will enjoy it.")
        st.caption("The catalog does not provide reliable demographic group data, so this app cannot assess demographic fairness. It does not infer sensitive traits.")

        with st.expander("Technical details · scores and model evidence",expanded=False):
            score=pd.to_numeric(row.get("model_score",np.nan),errors="coerce")
            adjusted=pd.to_numeric(row.get("responsible_score",np.nan),errors="coerce")
            rank_pct=float(pd.to_numeric(row.get("model_rank_percentile",np.nan),errors="coerce"))
            st.markdown(f"**Recommendation method:** {escape(model_name)}. **Raw model score:** {score:.4f} (model scale; not a probability). **Model rank percentile:** {rank_pct:.0%}. **Final rerank score:** {adjusted:.3f}.")
            if model_name=="Hybrid ensemble":
                all_components=["Rule-based inference","Content TF-IDF","KNN collaborative","SVD matrix factorization","Popularity regression","NeuMF deep learning"]
                components=[c for c in all_components if c in row.index and pd.notna(row[c])]
                if components:
                    st.write("Hybrid inputs are normalized within this candidate pool; the values below show each model’s contribution before averaging.")
                    st.dataframe(pd.DataFrame({"Model":components,"Normalized score":[round(float(row[c]),3) for c in components]}),hide_index=True,use_container_width=True)
                    omitted=[c for c in all_components if c not in components]
                    if omitted: st.caption("Not included for this result: "+", ".join(omitted)+" (their model path did not return a candidate score for the current profile/history).")
                    if "Content TF-IDF" in components and seeds:
                        terms,pair=shared_terms(seeds,row)
                        if pair:
                            st.write(f"Content component’s nearest seed: **{escape(pair[0])}** · cosine similarity **{pair[1]:.3f}**.")
                            if not terms.empty: st.dataframe(terms,hide_index=True,use_container_width=True)
            elif model_name=="Rule-based inference":
                st.write(f"**Task 2 inference:** {row.get('rule_label','General preference fallback')}. The score is the matching rule confidence metadata, or the rating fallback shown in the rule label.")
            elif model_name=="Popularity regression":
                st.write("The chart above explains this book’s estimated average rating from its catalog details.")
            elif model_name=="Content TF-IDF" and seeds:
                terms,pair=shared_terms(seeds,row)
                if pair:
                    st.write(f"Closest selected title: **{escape(pair[0])}** · content cosine similarity **{pair[1]:.3f}**.")
                    if not terms.empty: st.dataframe(terms,hide_index=True,use_container_width=True)
                else: st.write("This title was retrieved in the model’s similarity space; no selected seed had an aligned catalog row to show term overlap.")
            elif model_name=="NeuMF similar-item" and seeds:
                neural,_,item_map=neural_model(); target_id=int(row.book_id); seed_rows=indexed[indexed.title.isin(seeds)]
                matched=[int(bid) for bid in seed_rows.book_id if int(bid) in item_map and target_id in item_map]
                if matched:
                    embeddings=neural.mlp_item_embedding.weight.detach().cpu().numpy(); target_vector=embeddings[item_map[target_id]].reshape(1,-1)
                    similarities=[float(cosine_similarity(embeddings[item_map[bid]].reshape(1,-1),target_vector)[0,0]) for bid in matched]
                    best=int(np.argmax(similarities)); best_book=int(seed_rows[seed_rows.book_id.isin(matched)].iloc[best].book_id)
                    best_title=str(indexed.loc[indexed.book_id.eq(best_book),"title"].iloc[0])
                    st.write(f"Nearest selected title in trained NeuMF item space: **{escape(best_title)}** · embedding cosine similarity **{similarities[best]:.3f}**.")
                    st.caption("Embedding similarity is a latent model signal; it does not identify human-readable semantic causes.")
            elif model_name=="KNN collaborative":
                history=personalized_history(active_id)
                st.write(f"KNN compares **{len(history)} rated books** from this profile (your shelf or its linked existing reader) with neighboring Goodbooks reader histories. Its score is a rating prediction, not certainty.")
            elif model_name in ["SVD matrix factorization","NeuMF deep learning"]:
                linked=profiles.loc[profiles.id.eq(active_id),"lab_user_id"] if active_id else pd.Series(dtype=float)
                user_id=int(linked.iloc[0]) if not linked.empty and pd.notna(linked.iloc[0]) else None
                st.write(f"The pretrained {('SVD' if model_name.startswith('SVD') else 'NeuMF')} model ranked this book for linked Goodbooks reader **{user_id}**. The score reflects learned rating/interactions from the lab dataset; it is not a calibrated probability.")
            else:
                st.write("This pick ranks highly under the selected model’s saved book metadata or learned representation.")
            history_count=len(personalized_history(active_id))
            traces=rule_trace(row,history_count)
            if traces:
                st.markdown("**Rule-based evidence from Practical 2**")
                st.dataframe(pd.DataFrame([{"Rule":name,"Matched facts":facts,"Lab rule metadata":f"{conf:.0%} (uncalibrated)"} for name,conf,facts in traces]),hide_index=True,use_container_width=True)
                st.caption("Rule confidence is the lab’s rule metadata and should not be interpreted as a calibrated chance that you will like the book.")
            pop=float(pd.to_numeric(row.get("popularity_percentile",0.5),errors="coerce") or 0)
            count_value=pd.to_numeric(row.get("ratings_count",0),errors="coerce"); count=int(count_value) if pd.notna(count_value) else 0
            repeat=bool(row.get("author_repeat_in_list",False))
            risk="High exposure" if pop>=.9 else ("Moderate exposure" if pop>=.7 else "Long-tail / lower exposure")
            st.markdown("**Responsible-AI checks for this result**")
            st.write(f"- **Popularity exposure:** {risk} (catalog percentile {pop:.0%}; {count:,} ratings). The final list score is `0.85 × candidate-pool model percentile − 0.15 × catalog popularity percentile`, with an additional `0.08` penalty for each earlier pick by the same lead author. The raw model score remains visible above.")
            st.write(f"- **Evidence volume:** {'Limited' if count<25 else 'Some' if count<500 else 'Broader'} (reader-rating count is a proxy for evidence volume, not quality).")
            st.write(f"- **List diversity:** {'This author appears more than once in the displayed list.' if repeat else 'No repeated-author flag for this displayed result.'}")
            st.write("- **Fairness limitation:** Goodbooks-10k has no reliable protected-group labels here, so demographic fairness cannot be measured. The app does not infer sensitive traits.")
    
def recommendation_graph(row, model_name, seeds):
    """Build a trace from observed book similarities, ratings, rule facts, and metadata only."""
    graph=nx.Graph(); target_id=int(row.book_id); target_label=f"Recommended · {row.title}"
    graph.add_node("target",label=target_label,kind="recommendation")
    target_row=books[books.book_id.eq(target_id)].iloc[0]
    if active_id is not None:
        active_name=profiles.loc[profiles.id.eq(active_id),"name"].iloc[0]
        graph.add_node("reader_profile",label=f"Your profile · {active_name}",kind="reader")
    author=str(target_row.get("authors","Unknown author")).split(",")[0].strip()
    if author and author.lower() not in ["nan","unknown"]:
        author_key="author"; graph.add_node(author_key,label=f"Author · {author}",kind="author"); graph.add_edge("target",author_key,relation="written by")
    seed_node_by_id={}
    for i,title in enumerate(seeds):
        found=indexed[indexed.title.eq(title)]
        if found.empty:continue
        source_id=int(found.iloc[0].book_id); source_key=f"seed_{i}"
        seed_node_by_id[source_id]=source_key
        graph.add_node(source_key,label=f"Favorite · {title}",kind="favorite")
        if active_id is not None: graph.add_edge("reader_profile",source_key,relation="selected as a favorite")
        similarity=None
        target_index=indexed[indexed.book_id.eq(target_id)]
        if not target_index.empty:
            si=indexed.index.get_loc(found.index[0]); ti=indexed.index.get_loc(target_index.index[0])
            metric="TF-IDF cosine"
            if model_name=="NeuMF similar-item":
                neural,_,item_map=neural_model(); source_id=int(found.iloc[0].book_id)
                if source_id in item_map and target_id in item_map:
                    embeddings=neural.mlp_item_embedding.weight.detach().cpu().numpy()
                    similarity=float(cosine_similarity(embeddings[item_map[source_id]].reshape(1,-1),embeddings[item_map[target_id]].reshape(1,-1))[0,0]); metric="NeuMF item-embedding cosine"
            else: similarity=float(cosine_similarity(tfidf[si],tfidf[ti])[0,0])
        if similarity is not None:
            is_signal=(model_name in ["Content TF-IDF","NeuMF similar-item"] or (model_name=="Hybrid ensemble" and "Content TF-IDF" in row.index and pd.notna(row["Content TF-IDF"])))
            relation=f"{'ranking signal' if is_signal else 'context only'} · {metric} {similarity:.3f}"
        else: relation="selected favorite → recommendation"
        graph.add_edge(source_key,"target",relation=relation)
    if not seed_node_by_id:
        own_history=personalized_history(active_id).sort_values("rating",ascending=False).head(3)
        for index,record in enumerate(own_history.itertuples(index=False)):
            book_row=books.loc[books.book_id.eq(int(record.book_id)),"title"]
            if book_row.empty:continue
            key=f"rated_seed_{index}"; seed_node_by_id[int(record.book_id)]=key
            graph.add_node(key,label=f"Rated · {book_row.iloc[0]}",kind="favorite")
            if active_id is not None: graph.add_edge("reader_profile",key,relation=f"rated {float(record.rating):.0f}/5")
    # Show anonymized existing readers who rated both a favorite and the result highly.
    if seed_node_by_id:
        seed_ids=set(seed_node_by_id)
        target_readers=set(ratings.loc[ratings.book_id.eq(target_id)&ratings.rating.ge(4),"user_id"].tolist())
        if seed_ids and target_readers:
            liked=ratings.loc[ratings.book_id.isin(seed_ids)&ratings.rating.ge(4),["user_id","book_id"]]
            peer_ids=[u for u in liked.user_id.drop_duplicates().tolist() if u in target_readers][:3]
            for n,user_id in enumerate(peer_ids,1):
                peer_key=f"peer_{n}"; graph.add_node(peer_key,label=f"Goodbooks reader {n} · anonymized",kind="peer")
                graph.add_edge(peer_key,"target",relation="rated this book 4–5★")
                for _,hit in liked[liked.user_id.eq(user_id)].head(2).iterrows():
                    book=books.loc[books.book_id.eq(int(hit.book_id)),"title"]
                    if book.empty:continue
                    book_id=int(hit.book_id); key=seed_node_by_id.get(book_id,f"peer_book_{n}_{book_id}")
                    if key not in graph: graph.add_node(key,label=f"Reader {n} liked · {book.iloc[0]}",kind="favorite")
                    graph.add_edge(peer_key,key,relation="rated 4–5★")
    linked=profiles.loc[profiles.id.eq(active_id),"lab_user_id"] if active_id else pd.Series(dtype=float)
    user_id=int(linked.iloc[0]) if not linked.empty and pd.notna(linked.iloc[0]) else None
    if user_id is not None:
        node="linked_reader"; graph.add_node(node,label="Linked lab reader",kind="reader")
        past=ratings.loc[ratings.user_id.eq(user_id)].nlargest(4,"rating")
        for n,(_,record) in enumerate(past.iterrows()):
            book=books.loc[books.book_id.eq(int(record.book_id)),"title"]
            if book.empty:continue
            key=f"linked_book_{n}"; graph.add_node(key,label=f"Rated · {book.iloc[0]}",kind="favorite")
            graph.add_edge(node,key,relation=f"observed rating {record.rating:.0f}/5")
        graph.add_edge(node,"target",relation="model-predicted interaction" if model_name=="NeuMF deep learning" else "SVD predicted rating")
    for name,conf,facts in rule_trace(target_row,len(personalized_history(active_id))):
        key=f"rule_{name}"; graph.add_node(key,label=f"{name} · {conf:.0%}",kind="rule"); graph.add_edge(key,"target",relation=facts)
    if graph.number_of_edges()==0:
        graph.add_node("catalog",label="Goodbooks-10k catalog",kind="context"); graph.add_edge("catalog","target",relation="catalog metadata ranking")
    pos=nx.spring_layout(graph,seed=17,k=.9)
    ex=[]; ey=[]; edge_x=[]; edge_y=[]; edge_hover=[]
    for a,b,data in graph.edges(data=True):
        x0,y0=pos[a]; x1,y1=pos[b]; ex.extend([x0,x1,None]); ey.extend([y0,y1,None]); edge_x.append((x0+x1)/2); edge_y.append((y0+y1)/2); edge_hover.append(data.get("relation","connected"))
    edge=go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=1.5,color="#aaa895"),hoverinfo="skip")
    edge_details=go.Scatter(x=edge_x,y=edge_y,mode="markers",marker=dict(size=10,color="#a49c86",opacity=.75),hovertext=edge_hover,hoverinfo="text")
    nxv=[]; nyv=[]; labels=[]; hover=[]; colors=[]
    palette={"recommendation":"#2f6b4f","favorite":"#6689bd","peer":"#d18b49","reader":"#9e6bb1","author":"#66a487","rule":"#d18a45","context":"#999"}
    for node,data in graph.nodes(data=True):
        x,y=pos[node]; nxv.append(x); nyv.append(y); labels.append(data["label"][:28]); hover.append(data["label"]); colors.append(palette.get(data["kind"],"#888"))
    nodes=go.Scatter(x=nxv,y=nyv,mode="markers+text",text=labels,textposition="top center",hovertext=hover,hoverinfo="text",marker=dict(size=19,color=colors,line=dict(width=2,color="white")))
    fig=go.Figure([edge,edge_details,nodes]); fig.update_layout(height=460,showlegend=False,margin=dict(l=10,r=10,t=18,b=10),xaxis=dict(visible=False),yaxis=dict(visible=False),paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Edges show selected favorites, measured text or learned-embedding similarity, observed 4–5★ co-ratings, authorship, rule matches, or model predictions. Hover a connection for its label. Similarity and co-rating edges are marked as context when they were not a ranking input. Existing readers are anonymized.")

def hybrid_recommend(seeds, profile, limit=10):
    # Blend measured component scores on a shared candidate pool; absent user history cleanly falls back to content/popularity.
    components=[]
    for name in ["Rule-based inference","Content TF-IDF","KNN collaborative","SVD matrix factorization","Popularity regression","NeuMF deep learning"]:
        try:
            frame=recommend(name,seeds,profile,limit=80)
            if not frame.empty:
                frame=frame[["book_id","title","authors","image_url","average_rating","ratings_count","model_score"]].copy()
                s=pd.to_numeric(frame.model_score,errors="coerce")
                lo,hi=s.min(),s.max(); frame[name]=((s-lo)/(hi-lo)) if hi>lo else 0.5
                components.append(frame.set_index("book_id")[name])
        except Exception: pass
    if not components:return pd.DataFrame()
    scores=pd.concat(components,axis=1)
    base=books.set_index("book_id").join(scores,how="inner")
    parts=[x for x in scores.columns]
    base["model_score"]=base[parts].mean(axis=1,skipna=True)
    return base.reset_index().sort_values("model_score",ascending=False).head(limit)

with discover:
    titles=indexed.title.dropna().drop_duplicates().tolist()
    model_options=["Hybrid ensemble","Rule-based inference","Content TF-IDF","KNN collaborative","SVD matrix factorization","Popularity regression","NeuMF deep learning","NeuMF similar-item","Classical model evaluation","NeuMF optimization benchmark"]
    selected_model=st.selectbox("Choose a model or lab function",model_options)
    descriptions={
        "Hybrid ensemble":"Task 9 combines the available rule, content, collaborative, regression, and NeuMF scores. Each component is min-max normalized on its candidate pool, then averaged; components without usable profile data are omitted and listed in each result’s explanation.",
        "Rule-based inference":"Task 2 forward-chains the lab’s IF–THEN rules over rating, publication year, review ratio, and reader-history facts. This is a transparent rule-driven catalog ranking, not a learned personalized model.",
        "Content TF-IDF":"Task 6 compares book title, original title, and author text using saved TF-IDF vectors and cosine similarity. It recommends books like selected or previously rated titles.",
        "KNN collaborative":"Task 6 finds nearby readers in the existing user–book rating matrix, then estimates candidate ratings from those readers. It needs ratings on this profile’s shelf or an existing linked reader.",
        "SVD matrix factorization":"Task 6 factorizes the lab’s first 100,000 rating interactions and ranks unseen books for a linked existing reader using predicted ratings.",
        "Popularity regression":"Task 6 fits LinearRegression to rating count, text-review count, and publication year to estimate average rating. This is a catalog analysis, not a personalized recommender.",
        "NeuMF deep learning":"Task 7 loads the saved PyTorch GMF + MLP checkpoint and scores unseen user–book pairs for a linked existing reader.",
        "NeuMF similar-item":"Task 7 compares the trained model’s learned item embeddings to show books near selected or previously rated titles. This is item similarity, not user scoring.",
        "Classical model evaluation":"Task 6 benchmark view compares the metrics recorded in the original lab app. These are reference values from that app, not a fresh evaluation run.",
        "NeuMF optimization benchmark":"Task 7 dynamically INT8-quantizes NeuMF Linear layers and measures model size, latency, and prediction change. This is a model optimization experiment, not a book recommender."
    }
    st.subheader(f"{selected_model}")
    st.info(f"**Functionality:** {descriptions[selected_model]}")
    if selected_model=="Hybrid ensemble":
        st.markdown("**Ensemble score:** `mean(normalized rule, TF-IDF, KNN, SVD, regression, NeuMF scores that are available)` → transparent popularity/author rerank → ranked titles. Missing models are left out rather than filled with simulated scores.")

    no_recommendation_modes=["Classical model evaluation","NeuMF optimization benchmark"]
    if selected_model=="Classical model evaluation":
        st.markdown("#### Task 6 · recorded offline benchmark")
        benchmark=pd.DataFrame({
            "Model":["Content TF-IDF","Collaborative SVD","Popularity Regression","Random baseline"],
            "RMSE":[.942,.865,.912,1.450],"MAE":[.738,.672,.710,1.180],
            "Precision@10":[.285,.342,.198,.045],"Recall@10":[.241,.315,.165,.038],
            "NDCG@10":[.385,.462,.285,.082],"Catalog coverage":["92.4%","78.6%","100.0%","100.0%"]
        })
        st.dataframe(benchmark,hide_index=True,use_container_width=True)
        st.warning("These values are copied from the benchmark table in 10k_app.py; that table is hard-coded and is not recalculated here. Treat it as the lab’s reported reference, not a verified live measurement.")
        error_col,ranking_col=st.columns(2)
        with error_col:
            fig_error=go.Figure()
            for metric in ["RMSE","MAE"]: fig_error.add_trace(go.Bar(name=metric,x=benchmark.Model,y=benchmark[metric]))
            fig_error.update_layout(barmode="group",title="Rating prediction error (lower is better)",height=400)
            st.plotly_chart(fig_error,use_container_width=True)
        with ranking_col:
            fig_rank=go.Figure()
            for metric in ["Precision@10","Recall@10","NDCG@10"]: fig_rank.add_trace(go.Bar(name=metric,x=benchmark.Model,y=benchmark[metric]))
            fig_rank.update_layout(barmode="group",title="Top-10 ranking quality (higher is better)",height=400)
            st.plotly_chart(fig_rank,use_container_width=True)

    elif selected_model=="NeuMF optimization benchmark":
        linked_reader=linked_lab_user_id(active_id)
        st.markdown("#### Task 7 · dynamic INT8 quantization")
        if linked_reader is None:
            st.info("Switch to an existing Goodbooks reader from the left sidebar to benchmark predictions on that reader’s model inputs.")
        elif st.button("Run NeuMF optimization benchmark",type="primary",key="run_quantization"):
            try:
                with st.spinner("Comparing original and quantized NeuMF…"):
                    metrics=benchmark_neumf(active_id)
                if metrics is None: st.error("This linked reader ID is absent from the trained NeuMF user index.")
                else:
                    st.caption(f"Linked reader {linked_reader} · {metrics['items_scored']} item pairs · 5 warm CPU measurements per model.")
                    m1,m2,m3=st.columns(3)
                    m1.metric("Original model size",f"{metrics['original_size_mb']:.2f} MB")
                    m2.metric("Quantized size",f"{metrics['quantized_size_mb']:.2f} MB",f"{metrics['size_reduction_pct']:.1f}% change")
                    m3.metric("Mean absolute score change",f"{metrics['mean_abs_prediction_delta']:.5f}")
                    st.dataframe(pd.DataFrame([{"Model":"NeuMF original","Mean CPU latency (ms)":metrics["original_latency_ms"]},{"Model":"NeuMF dynamic INT8","Mean CPU latency (ms)":metrics["quantized_latency_ms"]}]),hide_index=True,use_container_width=True)
                    st.caption(f"Observed latency change: {metrics['speed_change_pct']:.1f}%. Timing depends on the machine and load; a negative change means quantization was slower in this run.")
            except Exception as exc: st.error(f"NeuMF optimization benchmark failed: {exc}")

    elif selected_model=="Popularity regression":
        st.markdown("#### Task 6 · non-personalized rating prediction")
        st.write("Choose a year and ratings-count band. The model predicts average ratings from three catalog features; it does not use your reader profile.")
        years=pd.to_numeric(books.original_publication_year,errors="coerce").dropna()
        min_year=max(1900,int(years.min())) if not years.empty else 1900; max_year=min(2026,int(years.max())) if not years.empty else 2026
        max_count=max(1000,int(pd.to_numeric(books.ratings_count,errors="coerce").max()))
        c1,c2=st.columns(2)
        with c1: year_range=st.slider("Publication year",min_year,max_year,(max(min_year,1990),min(max_year,2010)),key="reg_year")
        with c2: count_range=st.slider("Ratings count",0,max_count,(0,min(max_count,500000)),step=max(1,max_count//500),key="reg_count")
        if st.button("Run popularity regression analysis",type="primary",key="run_regression_analysis"):
            frame,_=regression_model()
            filtered=frame[frame.original_publication_year.between(*year_range)&frame.ratings_count.between(*count_range)].sort_values("predicted_rating",ascending=False).head(20)
            st.session_state.regression_view=filtered
        analysis=st.session_state.get("regression_view",pd.DataFrame())
        if not analysis.empty:
            st.caption("Model: scikit-learn LinearRegression · output: predicted average-rating estimate")
            st.dataframe(analysis[["title","authors","original_publication_year","average_rating","predicted_rating","ratings_count"]].rename(columns={"title":"Book","authors":"Author","original_publication_year":"Year","average_rating":"Observed average","predicted_rating":"Predicted average","ratings_count":"Ratings"}),hide_index=True,use_container_width=True)
            focus=st.selectbox("Inspect model features for a book",analysis.book_id.tolist(),format_func=lambda bid:str(analysis.loc[analysis.book_id.eq(bid),"title"].iloc[0]),key="regression_focus")

    else:
        linked_reader=linked_lab_user_id(active_id)
        seed_models=["Content TF-IDF","NeuMF similar-item"]
        needs_user=["SVD matrix factorization","NeuMF deep learning"]
        if linked_reader is not None:
            history=goodbooks_read_history(active_id)
            st.markdown(f"**Existing reader {linked_reader} provides the profile context.** Their {len(history):,} rating records are used directly; their top-rated titles seed item similarity.")
            st.dataframe(history[["title","authors","rating"]].head(6).rename(columns={"title":"Rated book","authors":"Author","rating":"Reader rating"}),hide_index=True,use_container_width=True)
            seeds=history.head(3).title.dropna().tolist()
        else:
            local_history=load_saved_books(active_id) if active_id is not None else pd.DataFrame()
            saved_history=local_history[local_history.rating.notna()] if not local_history.empty else pd.DataFrame()
            saved_type=profiles.loc[profiles.id.eq(active_id),"profile_type"].iloc[0] if active_id is not None else "new"
            default_reader_type="reader" if saved_type=="reader" or not saved_history.empty else "new"
            reader_type=st.radio("Choose your reader path",["new","reader"],index=0 if default_reader_type=="new" else 1,format_func=lambda kind:"New reader · no books read yet" if kind=="new" else "Reader with books · choose titles you’ve read",horizontal=True,key=f"reader_path_{active_id}")
            if active_id is not None and reader_type!=saved_type:
                with db() as conn: conn.execute("UPDATE profiles SET profile_type=? WHERE id=?",(reader_type,int(active_id)))
                profiles.loc[profiles.id.eq(active_id),"profile_type"]=reader_type
                st.session_state.pop("last_recs",None)
            if reader_type=="new":
                st.info("Cold-start path: there is no reading history yet. Rule-based discovery, popularity analysis, and the available catalog models can still be explored; personalized user models will activate after you add books or choose an existing reader.")
                seeds=[]
            else:
                st.markdown("#### Choose books you have read")
                existing_titles=saved_history.title.dropna().tolist() if not saved_history.empty else []
                manual_key=f"read_titles_{active_id}"
                seeds=st.multiselect("Your reading history / favorites",titles,default=existing_titles,placeholder="Search the catalog for books you have read",key=manual_key)
                if seeds:
                    save_rating=st.slider("Rating to save for selected titles",1,5,4,key=f"seed_rating_{active_id}")
                    if st.button("Save these titles to my reader profile",key=f"save_read_titles_{active_id}"):
                        for read_title in seeds:
                            matching=books.loc[books.title.eq(read_title),"book_id"]
                            if not matching.empty: save_book(active_id,int(matching.iloc[0]),"Finished",save_rating)
                        st.success("Selected books are saved as reading history for this profile.")
                        st.rerun()
                elif selected_model in seed_models or selected_model=="Hybrid ensemble":
                    st.caption("Pick books to activate content similarity. You can still run catalog-only hybrid discovery without selected titles.")
            if selected_model=="KNN collaborative":
                history=personalized_history(active_id)
                st.caption(f"KNN will use {len(history):,} rated books saved on this profile. Save selected titles above to build its reading history.")
        d1,d2=st.columns([2,1])
        with d1: st.markdown(f"**Model in this mode:** `{selected_model}`")
        with d2: limit=st.slider("Results",4,16,8,key=f"limit_{selected_model}")
        blocked_for_user=selected_model in needs_user and linked_reader is None
        blocked_for_seed=selected_model in seed_models and linked_reader is None and not seeds
        if blocked_for_user:
            st.warning("This trained user-based model needs an existing Goodbooks reader. Choose one in the left sidebar.")
        if blocked_for_seed:
            st.info("Switch the reader path to **Reader with books** and choose titles to provide item-similarity context.")
        elif not blocked_for_user and st.button(f"Run {selected_model}",type="primary",key=f"run_{selected_model}"):
            try:
                with st.spinner(f"Running {selected_model}…"):
                    result=recommend(selected_model,seeds,active_id,limit)
                st.session_state.last_recs=result; st.session_state.last_model=selected_model; st.session_state.last_seed_titles=list(seeds); st.session_state.last_profile_id=active_id
            except Exception as exc: st.error(f"{selected_model} failed: {exc}")
        result=st.session_state.get("last_recs",pd.DataFrame())
        if not result.empty and st.session_state.get("last_model")==selected_model and st.session_state.get("last_profile_id")==active_id and st.session_state.get("last_seed_titles",[])==list(seeds):
            result_seeds=st.session_state.get("last_seed_titles",list(seeds))
            st.markdown(f"#### Results · {selected_model}")
            if selected_model=="Popularity regression": st.caption("Catalog ranking by predicted average rating; this is not personalized.")
            else:
                exposure=result.popularity_percentile
                a1,a2,a3=st.columns(3)
                a1.metric("Results",len(result)); a2.metric("Distinct authors",result.authors.nunique()); a3.metric("High-popularity picks",int(exposure.ge(.9).sum()))
                st.caption("Per-result explanation and responsible-AI checks appear below each title. This exposure/diversity adjustment is not a demographic fairness certification.")
            context=(selected_model,result_seeds,result)
            for idx,row in result.iterrows(): render_book(row,f"discover_{selected_model}_{idx}",context)
            st.markdown("#### Recommendation connectivity graph · Task 1 / Task 8")
            chosen_id=st.selectbox("Inspect a recommended book",result.book_id.astype(int).tolist(),format_func=lambda bid:str(result.loc[result.book_id.eq(bid),"title"].iloc[0]),key=f"graph_{selected_model}")
            recommendation_graph(result[result.book_id.eq(chosen_id)].iloc[0],selected_model,result_seeds)
            st.markdown("#### Global SHAP · what influences catalog rating estimates?")
            st.caption("This chart summarizes the popularity rating model across up to 500 catalog books. It shows which catalog features tend to influence estimated ratings; it is global for that rating model, not an explanation of the selected recommender’s ranking.")
            try:
                shap_values=regression_shap_summary().sort_values("Mean absolute SHAP value")
                fig=px.bar(shap_values,x="Mean absolute SHAP value",y="Feature",orientation="h",color="Mean absolute SHAP value",color_continuous_scale=["#d8d0c0","#2f6b4f"],title="Which catalog features influence rating estimates?")
                fig.update_layout(showlegend=False,coloraxis_showscale=False,xaxis_title="Average absolute effect on predicted rating",yaxis_title="")
                st.plotly_chart(fig,use_container_width=True)
            except ImportError:
                st.info("Install the optional explainability packages with `pip install shap lime` to view SHAP and LIME charts.")
            except Exception as exc:
                st.error(f"Could not create the global SHAP chart: {exc}")

with models_tab:
    st.subheader("The models behind your recommendations")
    st.write("Each option runs the corresponding model or inference path from the lab project; the hybrid combines available model scores for your profile.")
    model_rows=[
        ("Rule-based inference","Practical 2 forward-chaining IF–THEN rules","Catalog facts + reader-history fallback","Rule badges and rule-driven catalog ranking"),
        ("Content TF-IDF","TF-IDF vectors + cosine similarity","Favorites or linked reader’s highest-rated books","Similar books; works immediately"),
        ("KNN collaborative","NearestNeighbors with cosine distance over the ratings matrix","Rated shelf books or linked existing reader","User-neighborhood rating predictions"),
        ("SVD matrix factorization","Truncated SVD over the lab's 100k rating sample","Linked Goodbooks lab user","Predicted user–book ratings"),
        ("Popularity regression","LinearRegression using rating count, review count, publication year","Book metadata","Predicted average rating ranking"),
        ("NeuMF deep learning","Saved PyTorch GMF + MLP model checkpoint","Linked Goodbooks lab user","Neural interaction scores"),
        ("NeuMF similar-item","Cosine similarity over learned NeuMF MLP item embeddings","Favorites or linked reader’s highest-rated books","Similar books in learned embedding space"),
        ("Hybrid ensemble","Mean-normalized score fusion from available models","Favorites + ratings + optional linked user","Combined ranking with missing models skipped"),
        ("Classical model evaluation","Recorded Task 6 reference metrics","No reader input","RMSE, MAE, ranking metrics, catalog coverage"),
        ("NeuMF optimization benchmark","Dynamic INT8 quantization of Linear layers","Linked existing Goodbooks reader","Measured size, CPU latency, and score difference; no book list"),
    ]
    st.dataframe(pd.DataFrame(model_rows,columns=["Model","Method","Personalization input","Output"]),hide_index=True,use_container_width=True)
    st.markdown("#### Choose an existing Goodbooks reader")
    st.caption("Choose from existing anonymized readers represented in the lab's SVD training sample. Their histories activate KNN, SVD, and NeuMF; your own profile shelf can personalize KNN too.")
    if active_id is not None:
        svd_user_counts=ratings.head(100000).groupby("user_id").size().sort_values(ascending=False)
        user_options=svd_user_counts.index.tolist()
        linked_value=profiles.loc[profiles.id.eq(active_id),"lab_user_id"].iloc[0]
        user_index=user_options.index(int(linked_value)) if pd.notna(linked_value) and int(linked_value) in user_options else 0
        with st.form("link_user"):
            lab_user=st.selectbox("Existing reader",user_options,index=user_index,format_func=lambda user_id:f"Reader {user_id} · {int(svd_user_counts.loc[user_id]):,} training ratings")
            if st.form_submit_button("Link to this profile"):
                with db() as conn: conn.execute("UPDATE profiles SET lab_user_id=?,profile_type='goodbooks' WHERE id=?",(int(lab_user),int(active_id)))
                st.success("Linked. KNN, SVD, and NeuMF can now use this existing lab reader."); st.rerun()
    st.markdown("#### Rate books to personalize")
    st.caption("Save a book, then use My shelf to set a 1–5 star rating. Those ratings drive profile-based KNN recommendations.")

with compare:
    st.subheader("Compare current book prices")
    st.write("Search Google Shopping listings from supported Indian stores.")
    st.caption("Functionality source: SerpApi Google Shopping API · India results")
    title=st.text_input("Book title",placeholder="e.g. The Midnight Library")
    author=st.text_input("Author (optional)",placeholder="Matt Haig")
    if st.button("Search stores",type="primary"):
        api_key=st.secrets.get("SERPAPI_KEY","")
        if not api_key: st.error("Add SERPAPI_KEY to .streamlit/secrets.toml to enable price search.")
        elif not title.strip(): st.warning("Enter a book title first.")
        else:
            with st.spinner("Checking current listings…"):
                try:
                    response=requests.get("https://serpapi.com/search.json",params={"engine":"google_shopping","q":f"{title} {author} book".strip(),"api_key":api_key,"gl":"in","hl":"en","google_domain":"google.co.in"},timeout=25)
                    response.raise_for_status(); payload=response.json()
                    if payload.get("error"): st.error(payload["error"])
                    else:
                        found=[r for r in payload.get("shopping_results",[]) if r.get("price") or r.get("extracted_price")]
                        found.sort(key=lambda r:r.get("extracted_price",float("inf")))
                        if found: st.dataframe(pd.DataFrame([{"Store":r.get("source","Store"),"Book / edition":r.get("title",""),"Price":r.get("price","—"),"Rating":r.get("rating","—"),"Delivery":r.get("delivery","—"),"Offer":r.get("product_link",r.get("link",""))} for r in found]),column_config={"Offer":st.column_config.LinkColumn("View offer")},hide_index=True,use_container_width=True)
                        else: st.info("No priced listings came back. Try a shorter title or add its author.")
                except requests.RequestException as exc: st.error(f"Shopping search unavailable: {exc}")
    st.caption("Prices and availability vary; listings can include different editions and formats.")

with shelf:
    st.subheader(f"{active.avatar + ' ' if active_id else ''}My reading shelf")
    if active_id is None: st.info("Create or select a profile from the person icon in the sidebar.")
    else:
        imported_history=goodbooks_read_history(active_id)
        if not imported_history.empty:
            st.markdown("#### Read books from this Goodbooks reader")
            st.caption(f"Imported from existing reader {linked_lab_user_id(active_id)} · {len(imported_history):,} rated books. This dataset history is separate from books you save locally.")
            st.dataframe(imported_history[["title","authors","rating","average_rating","ratings_count"]].rename(columns={"title":"Book","authors":"Author","rating":"Their rating","average_rating":"Goodbooks average","ratings_count":"Reader ratings"}),hide_index=True,use_container_width=True)
        else:
            st.info("No linked Goodbooks reading history. Save books from Discover to build this profile’s shelf.")
        st.markdown("#### Books saved to this profile")
        saved=load_saved_books(active_id)
        if saved.empty: st.caption("No locally saved books yet.")
        else:
            for _,row in saved.iterrows():
                c1,c2,c3=st.columns([3,1,1])
                with c1: render_book(row,f"shelf_{int(row.book_id)}")
                with c2:
                    status=st.selectbox("Reading status",["Want to read","Reading","Finished"],index=["Want to read","Reading","Finished"].index(row.status),key=f"status_{row.book_id}")
                    rating=st.select_slider("Your rating",options=[1,2,3,4,5],value=int(row.rating) if pd.notna(row.rating) else 3,key=f"rating_{row.book_id}")
                    if st.button("Save rating",key=f"rate_{row.book_id}"):
                        save_book(active_id,row.book_id,status,rating); st.rerun()
                with c3:
                    if st.button("Remove",key=f"remove_{row.book_id}"):
                        with db() as conn: conn.execute("DELETE FROM saved_books WHERE profile_id=? AND book_id=?",(int(active_id),int(row.book_id)))
                        st.rerun()

st.divider()
st.caption("BiblioMatch Reader · Profiles and shelves are stored locally in book_lovers_profiles.sqlite3. Model recommendations use the Goodbooks-10k lab data and saved artifacts.")
