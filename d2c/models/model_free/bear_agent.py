"""
Implementation of BEAR Q learning.
Paper:
    Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction
    (http://arxiv.org/abs/1906.00949)
    Note that the authors removed ensembles in their new version,
    and used a minimum/average over 2 Q-functions,
    without an ensemble-based conservative estimate based on sample variance.
"""