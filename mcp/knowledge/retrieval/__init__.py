"""
知识检索模块 - 负责从各种来源检索相关知识
"""

from .base import KnowledgeRetriever, RetrievalConfig, RetrievalResult, RetrievalStrategy

# 按需导入具体实现类
from .retrievers import (
    SimilarityRetriever,
    HybridRetriever,
    KeywordRetriever,
    EnsembleRetriever,
    ReRankingRetriever
)

__all__ = [
    'KnowledgeRetriever',
    'RetrievalConfig',
    'RetrievalResult',
    'RetrievalStrategy',
    'SimilarityRetriever',
    'HybridRetriever',
    'KeywordRetriever',
    'EnsembleRetriever',
    'ReRankingRetriever'
]
