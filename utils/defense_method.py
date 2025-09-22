import numpy as np
import random


def Add_noise(args, gdata_train):

    noise_std = args.noise_std if hasattr(args, 'noise_std') else 0.01  # default intensity = 0.01

    print(" Noise intensity ：", noise_std)

    noisy_features = []
    for i, feature in enumerate(gdata_train.features):
        if isinstance(feature, np.ndarray):
            noise = np.random.normal(0, noise_std, feature.shape)
            noisy_feature = feature + noise
            noisy_features.append(noisy_feature)
        else:
            raise ValueError(f"Feature at index {i} is not a NumPy ndarray.")

    gdata_train.features = noisy_features

    return gdata_train


def Prune_low_similarity(args, gdata):
    for gidx in range(len(gdata)):
        adj = gdata.adj_list[gidx]
        nidx_list = list(range(len(gdata.adj_list[gidx])))
        features = gdata.features[gidx]
        for i, nidx_i in enumerate(nidx_list):
            for j, nidx_j in enumerate(nidx_list):
                if i >= j:
                    continue
                similarity = compute_similarity(features[nidx_i], features[nidx_j])

                if similarity < args.sim_prune_threshold:
                    if random.random() < 0.2:
                        adj[nidx_i][nidx_j] = 0
                        adj[nidx_j][nidx_i] = 0
    return gdata


def compute_similarity(feature1, feature2):
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    return similarity

