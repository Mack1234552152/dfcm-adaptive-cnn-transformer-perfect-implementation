"""
Models package for DFCM Adaptive CNN Transformer
Perfect reproduction of the research paper
"""

from .dfcm import DFCM
from .adaptive_cnn import AdaptiveCNN
from .adaptive_transformer import AdaptiveTransformer
from .fusion_model import DFCM_A_CNN_Transformer

__all__ = [
    'DFCM',
    'AdaptiveCNN', 
    'AdaptiveTransformer',
    'DFCM_A_CNN_Transformer'
]

__version__ = "1.0.0-perfect-reproduction"
__author__ = "Perfect Reproduction Implementation"
__description__ = "Perfect reproduction of Adaptive CNN-Transformer Fusion for Discriminative Fuzzy C-Means Clustering"