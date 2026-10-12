import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
DATA_PATH = r"C:\Users\Admin\OneDrive\Desktop\BiblioMatch\10k\content_recommender\data"
SAVE_DIR = r"C:\Users\Admin\OneDrive\Desktop\BiblioMatch\Deep Learning Model 10k"

ratings = pd.read_csv(os.path.join(DATA_PATH, 'ratings.csv'))
books = pd.read_csv(os.path.join(DATA_PATH, 'books.csv'))

# Create or load mappings since they weren't found locally
print("Processing ID mappings...")
user_ids = ratings['user_id'].unique()
book_ids = ratings['book_id'].unique()

user_id_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
book_id_index = {book_id: idx for idx, book_id in enumerate(book_ids)}

# Save mappings for the app to use
with open(os.path.join(SAVE_DIR, 'user_id_index.pkl'), 'wb') as f:
    pickle.dump(user_id_index, f)
with open(os.path.join(SAVE_DIR, 'book_id_index.pkl'), 'wb') as f:
    pickle.dump(book_id_index, f)

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

print(f"Original ratings count: {len(ratings)}")
print(f"Filtered ratings count: {len(ratings_filtered)}")

# Train/Val/Test Split (70/15/15)
train_val, test = train_test_split(ratings_filtered, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.1765, random_state=42)

user_input_train = train['user_idx'].values
item_input_train = train['book_idx'].values
labels_train = train['rating_norm'].values

user_input_val = val['user_idx'].values
item_input_val = val['book_idx'].values
labels_val = val['rating_norm'].values

user_input_test = test['user_idx'].values
item_input_test = test['book_idx'].values
labels_test = test['rating_norm'].values

num_users = len(user_id_index)
num_items = len(book_id_index)

print("Building NeuMF model...")
# GMF Branch
gmf_user_input = Input(shape=(1,), name='gmf_user_input')
gmf_item_input = Input(shape=(1,), name='gmf_item_input')
gmf_user_embedding = Embedding(num_users, 16, name='gmf_user_embedding')(gmf_user_input)
gmf_item_embedding = Embedding(num_items, 16, name='gmf_item_embedding')(gmf_item_input)
gmf_user_flatten = Flatten()(gmf_user_embedding)
gmf_item_flatten = Flatten()(gmf_item_embedding)
gmf_vector = Multiply(name='gmf_multiply')([gmf_user_flatten, gmf_item_flatten])

# MLP Branch
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

print("Starting training...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    [user_input_train, item_input_train, user_input_train, item_input_train],
    labels_train,
    batch_size=256,
    epochs=10, # Limiting locally to 10 for speed unless you want to wait for 100
    validation_data=([user_input_val, item_input_val, user_input_val, item_input_val], labels_val),
    callbacks=[early_stopping],
    verbose=1
)

print("Evaluating...")
predictions_norm = model.predict([user_input_test, item_input_test, user_input_test, item_input_test], verbose=0)
predictions = predictions_norm.flatten() * 5.0

rmse = np.sqrt(mean_squared_error(test['rating'], predictions))
mae = mean_absolute_error(test['rating'], predictions)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Save
print("Saving artifacts...")
model.save(os.path.join(SAVE_DIR, 'neumf_model.h5'))

gmf_user_emb = model.get_layer('gmf_user_embedding').get_weights()[0]
gmf_item_emb = model.get_layer('gmf_item_embedding').get_weights()[0]
mlp_user_emb = model.get_layer('mlp_user_embedding').get_weights()[0]
mlp_item_emb = model.get_layer('mlp_item_embedding').get_weights()[0]

with open(os.path.join(SAVE_DIR, 'gmf_user_embeddings.pkl'), 'wb') as f:
    pickle.dump(gmf_user_emb, f)
with open(os.path.join(SAVE_DIR, 'gmf_item_embeddings.pkl'), 'wb') as f:
    pickle.dump(gmf_item_emb, f)
with open(os.path.join(SAVE_DIR, 'mlp_user_embeddings.pkl'), 'wb') as f:
    pickle.dump(mlp_user_emb, f)
with open(os.path.join(SAVE_DIR, 'mlp_item_embeddings.pkl'), 'wb') as f:
    pickle.dump(mlp_item_emb, f)
    
print("Training complete! Model and embeddings saved.")
