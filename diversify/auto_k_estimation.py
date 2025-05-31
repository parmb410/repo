import os
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# File: diversify/auto_k_estimation.py
# This module provides automated k estimation functionality for DIVERSIFY.


def estimate_optimal_k(features, k_min=2, k_max=10):
    """
    Estimate the optimal number of clusters (k) based on silhouette score.

    Args:
        features (np.ndarray): Feature matrix of shape (n_samples, n_features).
        k_min (int): Minimum number of clusters to try.
        k_max (int): Maximum number of clusters to try.

    Returns:
        best_k (int): Optimal number of clusters.
        best_labels (np.ndarray): Cluster labels using the best k.
    """
    best_k = k_min
    best_score = -1.0
    best_labels = None

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    return best_k, best_labels


class AutomatedKDataset:
    """
    Wrapper for automated k estimation on environment features for DIVERSIFY.

    Usage:
        src_envs: list of source environment features (list of numpy arrays)
        Each element in src_envs corresponds to one environment's feature matrix.
    """
    def __init__(self, src_envs, k_min=2, k_max=10):
        self.src_envs = src_envs
        self.k_min = k_min
        self.k_max = k_max
        self.k = None
        self.labels = None

    def run_estimation(self):
        """
        Estimate k across concatenated features and assign domain labels.
        """
        # Concatenate features from all environments
        all_features = np.concatenate(self.src_envs, axis=0)
        # Estimate optimal k and get labels
        self.k, all_labels = estimate_optimal_k(all_features, self.k_min, self.k_max)

        # Split labels back to each environment
        splits = np.cumsum([env.shape[0] for env in self.src_envs])[:-1]
        env_labels = np.split(all_labels, splits)
        self.labels = env_labels

        return self.k, self.labels


# Integration into diversify's training pipeline:
# In diversify/trainer.py or wherever clusters are defined, replace manual k with automated estimation.
# Example pseudocode:

# from diversify.auto_k_estimation import AutomatedKDataset
#
# def prepare_domains(data_loader):
#     src_envs = []
#     for (x, y, d) in data_loader:
#         # extract features (e.g., embeddings) for each environment
#         features = model.extract_features(x)
#         src_envs.append(features.cpu().numpy())
#
#     auto_k = AutomatedKDataset(src_envs, k_min=2, k_max=10)
#     optimal_k, domain_labels = auto_k.run_estimation()
#
#     # Use domain_labels for each sample to set up sub-domains in DIVERSIFY
#     ...
