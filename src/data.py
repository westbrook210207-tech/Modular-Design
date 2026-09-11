import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset


class CBFDataset(Dataset):
    """PyTorch Dataset for Two-Tower Recommendation Training."""

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
    """Loads raw MovieLens 100k datasets and normalizes IDs to 0-index."""
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

    # Shift 1-indexed IDs to 0-indexed
    users["user_id"] -= 1
    movie["movie_id"] -= 1
    ratings["user_id"] -= 1
    ratings["movie_id"] -= 1

    return users, movie, ratings


def preprocess_features(users: pd.DataFrame, movie: pd.DataFrame):
    """Preprocesses user and item features into numerical feature matrices."""
    # --- Item Features ---
    genre_cols = movie.columns[3:].tolist()
    genre_features = movie[genre_cols].values.astype(np.float32)

    # Extract & scale release year
    release_year = pd.to_numeric(movie["release_date"].str.split("-").str[-1], errors="coerce")
    release_year = release_year.fillna(release_year.median())
    scaler_year = StandardScaler()
    year_scaled = scaler_year.fit_transform(release_year.values.reshape(-1, 1)).astype(np.float32)

    # Genre-year interaction features
    genre_year_interaction = genre_features * year_scaled
    movie_features = np.hstack([genre_features, year_scaled, genre_year_interaction]).astype(np.float32)

    # --- User Features ---
    users_df = users.copy()
    users_df["gender_idx"] = users_df["gender"].map({"M": 0, "F": 1})

    # Age standardization
    age_scaler = StandardScaler()
    users_df["age"] = age_scaler.fit_transform(users_df[["age"]])

    # Occupation One-Hot Encoding
    occupation_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    occupation_encoded = occupation_encoder.fit_transform(users_df[["occupation"]])
    occupation_cols = occupation_encoder.get_feature_names_out(["occupation"])
    occupation_df = pd.DataFrame(occupation_encoded, columns=occupation_cols, index=users_df.index)

    # Zip Code Region Feature
    zip_region = users_df["zip_code"].astype(str).str[0]
    zip_region = zip_region.where(zip_region.str.isdigit(), "intl")
    zip_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    zip_encoded = zip_encoder.fit_transform(zip_region.values.reshape(-1, 1))
    zip_cols = zip_encoder.get_feature_names_out(["zip_region"])
    zip_df = pd.DataFrame(zip_encoded, columns=zip_cols, index=users_df.index)

    # Combine processed user features
    users_df = users_df.drop(columns=["gender", "occupation", "zip_code"])
    users_df = pd.concat([users_df, occupation_df, zip_df], axis=1)

    user_feature_cols = users_df.columns[1:].tolist()
    user_features = users_df[user_feature_cols].values.astype(np.float32)

    return user_features, movie_features


def build_train_test_split(ratings: pd.DataFrame, num_items: int):
    """Performs Leave-One-Out split on positive samples and negative sampling with hard negatives."""
    pos_df = ratings[ratings["rating"] >= 4].copy()
    pos_df["label"] = 1
    pos_df = pos_df.drop(columns="rating")

    # Leave-One-Out test split by timestamp per user
    pos_df = pos_df.sort_values(["user_id", "timestamp"])
    test_df = pos_df.groupby("user_id").tail(1)
    train_df = pos_df.drop(test_df.index)

    train_df = train_df.drop(columns="timestamp")
    test_df = test_df.drop(columns="timestamp")

    user_positive_items = pos_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    user_disliked_items = ratings[ratings["rating"] <= 3].groupby("user_id")["movie_id"].apply(set).to_dict()

    # Negative sampling (1 positive : 2 negative ratio, including hard negatives)
    train_rows = []
    for _, row in train_df.iterrows():
        u_id = int(row["user_id"])
        pos_item = int(row["movie_id"])

        # Positive sample
        train_rows.append([u_id, pos_item, 1])

        # Negative candidates
        user_positives = user_positive_items.get(u_id, set())
        negative_candidates = list(set(range(num_items)) - user_positives)

        disliked_set = user_disliked_items.get(u_id, set())
        if disliked_set:
            neg_item_hard = random.choice(list(disliked_set))
        else:
            neg_item_hard = random.choice(negative_candidates)

        neg_item_random = random.choice(negative_candidates)

        train_rows.append([u_id, neg_item_hard, 0])
        train_rows.append([u_id, neg_item_random, 0])

    sampled_train_df = pd.DataFrame(train_rows, columns=["user_id", "movie_id", "label"])
    return sampled_train_df, test_df