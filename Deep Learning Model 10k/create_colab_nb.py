import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

add_md("# SECTION 1 - Setup & Data Load\nMount Drive, import libraries, load data and ID mappings.")
add_code("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive (uncomment in Colab)
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_DIR = '/content/drive/MyDrive/BiblioMatch' # Adjust based on your Drive structure
DATA_DIR = '.' 

# Load datasets
ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
books = pd.read_csv(os.path.join(DATA_DIR, 'books.csv'))

# Load existing mappings
with open(os.path.join(DATA_DIR, 'user_id_index.pkl'), 'rb') as f:
    user_id_index = pickle.load(f)
with open(os.path.join(DATA_DIR, 'book_id_index.pkl'), 'rb') as f:
    book_id_index = pickle.load(f)""")

add_md("# SECTION 2 - Data Preparation & EDA\nFilter dataset, apply ID mappings, normalize, and visualize.")
add_code("""# EDA Before Preprocessing
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='rating', data=ratings)
plt.title('Distribution of Ratings (Before Preprocessing)')

plt.subplot(1, 2, 2)
user_counts = ratings.groupby('user_id').size()
sns.histplot(user_counts, bins=50, kde=False)
plt.title('Number of Ratings per User (Before Preprocessing)')
plt.xlim(0, 200)
plt.tight_layout()
plt.show()

# Apply filtering (users >= 5, books >= 10)
user_counts = ratings['user_id'].value_counts()
book_counts = ratings['book_id'].value_counts()

valid_users = user_counts[user_counts >= 5].index
valid_books = book_counts[book_counts >= 10].index

ratings_filtered = ratings[ratings['user_id'].isin(valid_users) & ratings['book_id'].isin(valid_books)].copy()

# Map IDs
ratings_filtered['user_idx'] = ratings_filtered['user_id'].map(user_id_index)
ratings_filtered['book_idx'] = ratings_filtered['book_id'].map(book_id_index)
ratings_filtered = ratings_filtered.dropna(subset=['user_idx', 'book_idx'])
ratings_filtered['user_idx'] = ratings_filtered['user_idx'].astype(int)
ratings_filtered['book_idx'] = ratings_filtered['book_idx'].astype(int)

# Normalize ratings
ratings_filtered['rating_norm'] = ratings_filtered['rating'] / 5.0

# EDA After Preprocessing / Feature Engineering Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='rating', data=ratings_filtered)
plt.title('Distribution of Ratings (After Preprocessing)')

plt.subplot(1, 2, 2)
user_counts_filtered = ratings_filtered.groupby('user_id').size()
sns.histplot(user_counts_filtered, bins=50, kde=False)
plt.title('Number of Ratings per User (After Preprocessing)')
plt.xlim(0, 200)
plt.tight_layout()
plt.show()

# Quality of dataset
print(f"Original ratings count: {len(ratings)}")
print(f"Filtered ratings count: {len(ratings_filtered)}")
print(f"Data retention: {len(ratings_filtered)/len(ratings)*100:.2f}%")

# Train/Val/Test Split (70/15/15)
train_val, test = train_test_split(ratings_filtered, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.1765, random_state=42) # 0.15 / 0.85 = 0.1765

print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")

user_input_train = train['user_idx'].values
item_input_train = train['book_idx'].values
labels_train = train['rating_norm'].values

user_input_val = val['user_idx'].values
item_input_val = val['book_idx'].values
labels_val = val['rating_norm'].values

user_input_test = test['user_idx'].values
item_input_test = test['book_idx'].values
labels_test = test['rating_norm'].values""")

add_md("# SECTION 3 - NeuMF Architecture\nDefine GMF and MLP branches, fuse them, and visualize the network diagram.")
add_code("""num_users = len(user_id_index)
num_items = len(book_id_index)

# GMF Branch - Generalized Matrix Factorization (mimics SVD)
gmf_user_input = Input(shape=(1,), name='gmf_user_input')
gmf_item_input = Input(shape=(1,), name='gmf_item_input')

gmf_user_embedding = Embedding(num_users, 16, name='gmf_user_embedding')(gmf_user_input)
gmf_item_embedding = Embedding(num_items, 16, name='gmf_item_embedding')(gmf_item_input)

gmf_user_flatten = Flatten()(gmf_user_embedding)
gmf_item_flatten = Flatten()(gmf_item_embedding)

gmf_vector = Multiply(name='gmf_multiply')([gmf_user_flatten, gmf_item_flatten])

# MLP Branch - Modeling non-linear interactions
mlp_user_input = Input(shape=(1,), name='mlp_user_input')
mlp_item_input = Input(shape=(1,), name='mlp_item_input')

mlp_user_embedding = Embedding(num_users, 32, name='mlp_user_embedding')(mlp_user_input)
mlp_item_embedding = Embedding(num_items, 32, name='mlp_item_embedding')(mlp_item_input)

mlp_user_flatten = Flatten()(mlp_user_embedding)
mlp_item_flatten = Flatten()(mlp_item_embedding)

mlp_vector = Concatenate(name='mlp_concat')([mlp_user_flatten, mlp_item_flatten])

mlp_layer_1 = Dense(128, activation='relu', name='mlp_layer_1')(mlp_vector)
mlp_layer_1 = Dropout(0.3)(mlp_layer_1)
mlp_layer_2 = Dense(64, activation='relu', name='mlp_layer_2')(mlp_layer_1)
mlp_layer_2 = Dropout(0.2)(mlp_layer_2)
mlp_layer_3 = Dense(32, activation='relu', name='mlp_layer_3')(mlp_layer_2)
mlp_layer_4 = Dense(16, activation='relu', name='mlp_layer_4')(mlp_layer_3)

# Fusion
predict_vector = Concatenate(name='fusion_concat')([gmf_vector, mlp_layer_4])
prediction = Dense(1, activation='sigmoid', name='prediction')(predict_vector)

model = Model(inputs=[gmf_user_input, gmf_item_input, mlp_user_input, mlp_item_input], outputs=prediction)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

model.summary()

# Plot Network Diagram / Information Flow
try:
    plot_model(model, to_file='neumf_architecture.png', show_shapes=True, show_layer_names=True)
    import matplotlib.image as mpimg
    img = mpimg.imread('neumf_architecture.png')
    plt.figure(figsize=(15, 15))
    plt.imshow(img)
    plt.axis('off')
    plt.title("NeuMF Architecture: Information Flow from Input to Fusion")
    plt.show()
except Exception as e:
    print("Could not plot model architecture (requires graphviz installed in Colab).", e)""")

add_md("# SECTION 4 - Training\nTrain the model and plot learning curves.")
add_code("""early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    [user_input_train, item_input_train, user_input_train, item_input_train],
    labels_train,
    batch_size=256,
    epochs=100,
    validation_data=([user_input_val, item_input_val, user_input_val, item_input_val], labels_val),
    callbacks=[early_stopping],
    verbose=1
)

# Plot training vs validation loss and MAE curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (MSE)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE')
plt.legend()
plt.tight_layout()
plt.show()""")

add_md("# SECTION 5 - Recommendation Functions\nInterface identical to the SVD model.")
add_code("""def recommend_for_user(user_id, top_n=10):
    if user_id not in user_id_index:
        return [], "insufficient history for NCF"
    
    user_idx = user_id_index[user_id]
    
    # Get unrated books for this user
    user_rated_books = ratings_filtered[ratings_filtered['user_id'] == user_id]['book_id'].tolist()
    unrated_books = books[~books['book_id'].isin(user_rated_books)]
    
    unrated_book_ids = unrated_books['book_id'].tolist()
    unrated_book_idxs = [book_id_index[bid] for bid in unrated_book_ids if bid in book_id_index]
    valid_unrated_book_ids = [bid for bid in unrated_book_ids if bid in book_id_index]
    
    if not unrated_book_idxs:
        return []
    
    user_idxs_array = np.array([user_idx] * len(unrated_book_idxs))
    book_idxs_array = np.array(unrated_book_idxs)
    
    preds = model.predict([user_idxs_array, book_idxs_array, user_idxs_array, book_idxs_array], verbose=0)
    preds = preds.flatten()
    
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

from sklearn.metrics.pairwise import cosine_similarity

def recommend_similar_books(book_title, top_n=10):
    # Use MLP item embeddings for richer representation
    mlp_item_emb_layer = model.get_layer('mlp_item_embedding')
    item_embeddings = mlp_item_emb_layer.get_weights()[0]
    
    # Find book
    matched_books = books[books['title'].str.contains(book_title, case=False, na=False)]
    if matched_books.empty:
        return []
    
    b_id = matched_books.iloc[0]['book_id']
    if b_id not in book_id_index:
        return []
    
    b_idx = book_id_index[b_id]
    query_emb = item_embeddings[b_idx].reshape(1, -1)
    
    sim_scores = cosine_similarity(query_emb, item_embeddings).flatten()
    
    # Exclude the book itself
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
            'ncf_score': float(sim_scores[idx]), # Re-using ncf_score for similarity score, scaled 0-1
            'explanation': f"Found similar learned features to {book_title}",
            'source': "ncf"
        })
        
    return recommendations""")

add_md("# SECTION 6 - Evaluation & Confusion Matrix\nEvaluate using RMSE, MAE, Precision/Recall, and present confusion matrices.")
add_code("""# Predictions
predictions_norm = model.predict([user_input_test, item_input_test, user_input_test, item_input_test], verbose=0)
predictions = predictions_norm.flatten() * 5.0 # Rescale to 1-5

# RMSE and MAE
rmse = np.sqrt(mean_squared_error(test['rating'], predictions))
mae = mean_absolute_error(test['rating'], predictions)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Precision@10 and Recall@10
def calculate_precision_recall_at_k(test_df, predictions, k=10, threshold=4.0):
    test_df_copy = test_df.copy()
    test_df_copy['prediction'] = predictions
    user_groups = test_df_copy.groupby('user_id')
    
    precisions = []
    recalls = []
    
    for user_id, group in user_groups:
        top_k = group.nlargest(k, 'prediction')
        
        num_relevant_top_k = sum(top_k['rating'] >= threshold)
        num_relevant_total = sum(group['rating'] >= threshold)
        
        precision_at_k = num_relevant_top_k / k
        recall_at_k = num_relevant_top_k / num_relevant_total if num_relevant_total > 0 else 0
        
        precisions.append(precision_at_k)
        recalls.append(recall_at_k)
        
    return np.mean(precisions), np.mean(recalls)

p10, r10 = calculate_precision_recall_at_k(test, predictions, k=10)
print(f"Precision@10: {p10:.4f}")
print(f"Recall@10: {r10:.4f}")

# Confusion Matrix for Rating >= 4 (Relevant vs Irrelevant)
y_true_binary = (test['rating'] >= 4).astype(int)
y_pred_binary = (predictions >= 4).astype(int)

cm = confusion_matrix(y_true_binary, y_pred_binary)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted (1 = Relevant)')
plt.ylabel('Actual (1 = Relevant)')
plt.title('Confusion Matrix (Relevant >= 4)')
plt.show()

print("\\nClassification Report:")
print(classification_report(y_true_binary, y_pred_binary))""")

add_md("# SECTION 7 - Save Artifacts & Verify\nExport to Drive and run sample recommendations.")
add_code("""# Save model
model.save(os.path.join(DATA_DIR, 'neumf_model.h5'))

# Save embeddings
gmf_user_emb = model.get_layer('gmf_user_embedding').get_weights()[0]
gmf_item_emb = model.get_layer('gmf_item_embedding').get_weights()[0]
mlp_user_emb = model.get_layer('mlp_user_embedding').get_weights()[0]
mlp_item_emb = model.get_layer('mlp_item_embedding').get_weights()[0]

with open(os.path.join(DATA_DIR, 'gmf_user_embeddings.pkl'), 'wb') as f:
    pickle.dump(gmf_user_emb, f)
with open(os.path.join(DATA_DIR, 'gmf_item_embeddings.pkl'), 'wb') as f:
    pickle.dump(gmf_item_emb, f)
with open(os.path.join(DATA_DIR, 'mlp_user_embeddings.pkl'), 'wb') as f:
    pickle.dump(mlp_user_emb, f)
with open(os.path.join(DATA_DIR, 'mlp_item_embeddings.pkl'), 'wb') as f:
    pickle.dump(mlp_item_emb, f)
    
with open(os.path.join(DATA_DIR, 'training_history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)
    
print("Artifacts successfully saved to Drive.")

# Verify by loading
from tensorflow.keras.models import load_model
loaded_model = load_model(os.path.join(DATA_DIR, 'neumf_model.h5'))
print("Model loaded successfully for verification.")

# Print example recommendations
user_id_example = list(user_id_index.keys())[0]
recs = recommend_for_user(user_id_example, top_n=3)

print(f"\\nRecommendations for user {user_id_example}:")
for r in recs:
    print(r)""")

output_path = r"C:\Users\Admin\OneDrive\Desktop\BiblioMatch\Deep Learning Model 10k\NeuMF_Recommender.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook successfully created at {output_path}")
