import torch
import torch.nn as nn


class TwoTower(nn.Module):
    """Two-Tower Architecture for User and Item Embedding Matching."""

    def __init__(self, user_input_dim: int, movie_input_dim: int, embedding_dim: int = 32):
        super().__init__()

        self.user_mlp = nn.Sequential(
            nn.Linear(user_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

        self.movie_mlp = nn.Sequential(
            nn.Linear(movie_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, user_features: torch.Tensor, movie_features: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_mlp(user_features)
        movie_embedding = self.movie_mlp(movie_features)

        # Dot product scoring
        return (user_embedding * movie_embedding).sum(dim=1)