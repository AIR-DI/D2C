""" Implementation of BEAR Q learning.

This module implements the BEAR Q learning algorithm, which is a stabilizing off-policy
Q-learning method based on bootstrapping error reduction.

Paper: Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction
(http://arxiv.org/abs/1906.00949) Note that the authors removed ensembles in their new version,
and used a minimum/average over 2 Q-functions, without an ensemble-based conservative estimate based on sample variance.
"""