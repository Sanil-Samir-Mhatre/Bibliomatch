import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Loading data...")
DATA_PATH = r"C:\Users\Admin\OneDrive\Desktop\BiblioMatch\10k\content_recommender\data"
SAVE_DIR = r"C:\Users\Admin\OneDrive\Desktop\BiblioMatch\Deep Learning Model 10k"

ratings = pd.read_csv(os.path.join(DATA_PATH, 'ratings.csv'))
books = pd.read_csv(os.path.join(DATA_PATH, 'books.csv'))

print("Processing ID mappings...")
user_ids = ratings['user_id'].unique()
book_ids = ratings['book_id'].unique()

user_id_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
book_id_index = {book_id: idx for idx, book_id in enumerate(book_ids)}

with open(os.path.join(SAVE_DIR, 'user_id_index.pkl'), 'wb') as f:
    pickle.dump(user_id_index, f)
with open(os.path.join(SAVE_DIR, 'book_id_index.pkl'), 'wb') as f:
    pickle.dump(book_id_index, f)

user_counts = ratings['user_id'].value_counts()
book_counts = ratings['book_id'].value_counts()

valid_users = user_counts[user_counts >= 5].index
valid_books = book_counts[book_counts >= 10].index

ratings_filtered = ratings[ratings['user_id'].isin(valid_users) & ratings['book_id'].isin(valid_books)].copy()

ratings_filtered['user_idx'] = ratings_filtered['user_id'].map(user_id_index)
ratings_filtered['book_idx'] = ratings_filtered['book_id'].map(book_id_index)
ratings_filtered = ratings_filtered.dropna(subset=['user_idx', 'book_idx'])
ratings_filtered['user_idx'] = ratings_filtered['user_idx'].astype(int)
ratings_filtered['book_idx'] = ratings_filtered['book_idx'].astype(int)
ratings_filtered['rating_norm'] = ratings_filtered['rating'] / 5.0

print(f"Original ratings count: {len(ratings)}")
print(f"Filtered ratings count: {len(ratings_filtered)}")

train_val, test = train_test_split(ratings_filtered, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.1765, random_state=42)

class RatingDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
    def __len__(self):
        return len(self.users)
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

train_dataset = RatingDataset(train['user_idx'].values, train['book_idx'].values, train['rating_norm'].values)
val_dataset = RatingDataset(val['user_idx'].values, val['book_idx'].values, val['rating_norm'].values)
test_dataset = RatingDataset(test['user_idx'].values, test['book_idx'].values, test['rating_norm'].values)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

num_users = len(user_id_index)
num_items = len(book_id_index)

print("Building PyTorch NeuMF model...")
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuMF(num_users, num_items).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100 
patience = 5
best_val_loss = float('inf')
patience_counter = 0

print(f"Starting training on {device}...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for u, i, r in train_loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        optimizer.zero_grad()
        preds = model(u, i)
        loss = criterion(preds, r)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(u)
    
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for u, i, r in val_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            preds = model(u, i)
            loss = criterion(preds, r)
            val_loss += loss.item() * len(u)
    val_loss /= len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'neumf_model.pt'))
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print("Evaluating...")
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'neumf_model.pt'), weights_only=True))
model.eval()
all_preds = []
all_trues = []
with torch.no_grad():
    for u, i, r in test_loader:
        u, i = u.to(device), i.to(device)
        preds = model(u, i)
        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(r.numpy())

all_preds = np.array(all_preds) * 5.0
all_trues = np.array(all_trues) * 5.0

rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
mae = mean_absolute_error(all_trues, all_preds)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

print("Saving artifacts...")
gmf_user_emb = model.gmf_user_embedding.weight.detach().cpu().numpy()
gmf_item_emb = model.gmf_item_embedding.weight.detach().cpu().numpy()
mlp_user_emb = model.mlp_user_embedding.weight.detach().cpu().numpy()
mlp_item_emb = model.mlp_item_embedding.weight.detach().cpu().numpy()

with open(os.path.join(SAVE_DIR, 'gmf_user_embeddings.pkl'), 'wb') as f:
    pickle.dump(gmf_user_emb, f)
with open(os.path.join(SAVE_DIR, 'gmf_item_embeddings.pkl'), 'wb') as f:
    pickle.dump(gmf_item_emb, f)
with open(os.path.join(SAVE_DIR, 'mlp_user_embeddings.pkl'), 'wb') as f:
    pickle.dump(mlp_user_emb, f)
with open(os.path.join(SAVE_DIR, 'mlp_item_embeddings.pkl'), 'wb') as f:
    pickle.dump(mlp_item_emb, f)
    
print("Training complete! PyTorch Model and embeddings saved.")
