import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (
    CBFDataset,
    build_train_test_split,
    load_raw_data,
    preprocess_features,
)
from eval_ import evaluate_retrieval
from model import TwoTower


def main():
    # 1. Device selection[cite: 1]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Data processing[cite: 1]
    users, movie, ratings = load_raw_data(data_dir="data/ml-100k")
    num_users = ratings["user_id"].nunique()
    num_items = ratings["movie_id"].nunique()

    user_features, movie_features = preprocess_features(users, movie)
    train_df, test_df = build_train_test_split(ratings, num_items)

    train_set = CBFDataset(train_df, user_features, movie_features)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

    # 3. Model instantiation[cite: 1]
    model = TwoTower(
        user_input_dim=user_features.shape[1],
        movie_input_dim=movie_features.shape[1],
        embedding_dim=32,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Training loop[cite: 1]
    epochs = 25
    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_user, batch_item, batch_labels in train_loader:
            batch_user = batch_user.to(device)
            batch_item = batch_item.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_user, batch_item)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {avg_loss:.4f}")

    # 5. Retrieval & evaluation[cite: 1]
    print("\nEvaluating Retrieval Performance...")
    k_values = [10, 50, 100, 200, 500]
    metrics = evaluate_retrieval(
        model=model,
        user_features=user_features,
        movie_features=movie_features,
        ratings=ratings,
        test_df=test_df,
        num_users=num_users,
        num_items=num_items,
        device=device,
        k_values=k_values,
    )

    for K, res in metrics.items():
        print(
            f"K={K:>4} | Recall: {res['recall']:.4f} | "
            f"Precision: {res['precision']:.4f} | NDCG: {res['ndcg']:.4f}"
        )


if __name__ == "__main__":
    main()