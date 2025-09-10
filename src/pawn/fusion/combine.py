#!/usr/bin/env python3

import numpy as np

def fuse_scores(supervised_proba, unsupervised_scores, weight_supervised=0.7):
    weight_unsupervised = 1.0 - weight_supervised
    fused = weight_supervised * supervised_proba + weight_unsupervised * unsupervised_scores
    return fused

def normalize_anomaly_scores(scores):
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    return scores_norm
