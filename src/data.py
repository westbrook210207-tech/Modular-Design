import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset


class CBFDataset(Dataset):
    """PyTorch Dataset for Two-Tower Recommendation Training[cite: 1]."""

    def __init__(self, df: pd.DataFrame, user_features: np.ndarray, movie_features: np.ndarray):
        self.user_ids = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.movie_ids = torch.tensor(df["movie_id"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

        self.user_features = torch.tensor(user_features, dtype=torch.float32)
        self.movie_features = torch.tensor(movie_features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        movie_id = self.movie_ids[idx]
        return (
            self.user_features[user_id],
            self.movie_features[movie_id],
            self.labels[idx],
        )


def load_raw_data(data_dir: str = "data/ml-100k"):
    """Loads raw MovieLens 100k datasets and normalizes IDs to 0-index[cite: 1]."""
    user_cols = ["user_id", "age", "gender", "occupation", "zip_code"]
    users = pd.read_csv(f"{data_dir}/u.user", sep="|", names=user_cols, engine="python")

    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    movie_cols = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL"] + genre_cols

    movie = pd.read_csv(
        f"{data_dir}/u.item",
        sep="|",
        names=movie_cols,
        encoding="latin-1",
        engine="python",
    ).drop(columns=["video_release_date", "IMDb_URL"])

    rating_cols = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(f"{data_dir}/u.data", sep="\t", names=rating_cols, engine="python")

    # Shift 1-indexed IDs to 0-indexed[cite: 1]
    users["user_id"] -= 1
    movie["movie_id"] -= 1
    ratings["user_id"] -= 1
    ratings["movie_id"] -= 1

    return users, movie, ratings


def preprocess_features(users: pd.DataFrame, movie: pd.DataFrame):
    """Preprocesses user and item features into numerical feature matrices[cite: 1]."""
    # Item features: One-hot encoded genres[cite: 1]
    genre_cols = movie.columns[3:].tolist()
    movie_features = movie[genre_cols].values.astype(np.float32)

    # User features: Binary gender, standardized age, one-hot occupation[cite: 1]
    users["gender_idx"] = users["gender"].map({"M": 0, "F": 1})
    users = users.drop(columns=["gender", "zip_code"])

    age_scaler = StandardScaler()
    users["age"] = age_scaler.fit_transform(users[["age"]])

    occupation_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    occupation_encoded = occupation_encoder.fit_transform(users[["occupation"]])
    occupation_cols = occupation_encoder.get_feature_names_out(["occupation"])

    occupation_df = pd.DataFrame(occupation_encoded, columns=occupation_cols, index=users.index)
    users = pd.concat([users, occupation_df], axis=1).drop(columns=["occupation"])

    user_feature_cols = users.columns[1:].tolist()
    user_features = users[user_feature_cols].values.astype(np.float32)

    return user_features, movie_features


def build_train_test_split(ratings: pd.DataFrame, num_items: int):
    """Performs Leave-One-Out split on positive samples and negative sampling[cite: 1]."""
    pos_df = ratings[ratings["rating"] >= 4].copy()
    pos_df["label"] = 1
    pos_df = pos_df.drop(columns="rating")

    # Leave-One-Out test split by timestamp per user[cite: 1]
    pos_df = pos_df.sort_values(["user_id", "timestamp"])
    test_df = pos_df.groupby("user_id").tail(1)
    train_df = pos_df.drop(test_df.index)

    train_df = train_df.drop(columns="timestamp")
    test_df = test_df.drop(columns="timestamp")

    user_positive_items = pos_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    # Negative sampling (1:1 ratio)[cite: 1]
    train_rows = []
    for _, row in train_df.iterrows():
        u_id = int(row["user_id"])
        pos_item = int(row["movie_id"])

        train_rows.append([u_id, pos_item, 1])

        user_positives = user_positive_items[u_id]
        negative_candidates = list(set(range(num_items)) - user_positives)
        neg_item = random.choice(negative_candidates)

        train_rows.append([u_id, neg_item, 0])

    sampled_train_df = pd.DataFrame(train_rows, columns=["user_id", "movie_id", "label"])
    return sampled_train_df, test_df