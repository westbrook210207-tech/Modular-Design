import faiss
import numpy as np
import pandas as pd
import torch
from model import TwoTower


def generate_embeddings(
    model: TwoTower,
    user_features: np.ndarray,
    movie_features: np.ndarray,
    device: torch.device,
):
    """Computes user and item embeddings from trained model towers[cite: 1]."""
    model.eval()
    with torch.no_grad():
        movie_tensor = torch.tensor(movie_features, dtype=torch.float32).to(device)
        user_tensor = torch.tensor(user_features, dtype=torch.float32).to(device)

        movie_embeddings = model.movie_mlp(movie_tensor).cpu().numpy().astype("float32")
        user_embeddings = model.user_mlp(user_tensor).cpu().numpy().astype("float32")

    return user_embeddings, movie_embeddings


def evaluate_retrieval(
    model: TwoTower,
    user_features: np.ndarray,
    movie_features: np.ndarray,
    ratings: pd.DataFrame,
    test_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    device: torch.device,
    k_values: list = [10, 50, 100, 200, 500],
):
    """Builds FAISS index, filters seen items, and computes Recall@K, Precision@K, NDCG@K[cite: 1]."""
    user_emb, movie_emb = generate_embeddings(model, user_features, movie_features, device)

    # Build FAISS inner product index[cite: 1]
    embedding_dim = movie_emb.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(movie_emb)

    user_emb = np.ascontiguousarray(user_emb, dtype=np.float32)
    faiss.omp_set_num_threads(1)

    _, movie_ids = index.search(user_emb, num_items)

    test_movie_by_user = test_df.set_index("user_id")["movie_id"].to_dict()
    seen_movies_by_user = ratings.groupby("user_id")["movie_id"].apply(set).to_dict()

    # Exclude test movies from the 'seen' filter list to allow retrieval[cite: 1]
    for u_id, test_m_id in test_movie_by_user.items():
        seen_movies_by_user[u_id].discard(test_m_id)

    max_K = max(k_values)
    all_top_k = []
    for u_id in range(num_users):
        ranked_movies = movie_ids[u_id]
        seen_movies = seen_movies_by_user.get(u_id, set())
        filtered_movies = [m for m in ranked_movies if m not in seen_movies]
        all_top_k.append(filtered_movies[:max_K])

    results = {}
    for K in k_values:
        hits = 0
        precision_sum = 0
        ndcg_sum = 0
        num_eval_users = 0

        for u_id in range(num_users):
            test_m_id = test_movie_by_user.get(u_id)
            if test_m_id is None:
                continue

            num_eval_users += 1
            top_k = all_top_k[u_id][:K]

            if test_m_id in top_k:
                hits += 1
                rank = top_k.index(test_m_id)
                precision_sum += 1 / K
                ndcg_sum += 1 / np.log2(rank + 2)

        results[K] = {
            "recall": hits / num_eval_users,
            "precision": precision_sum / num_eval_users,
            "ndcg": ndcg_sum / num_eval_users,
        }

    return results