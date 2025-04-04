"""
向量存储模块 - 负责存储和检索文本嵌入向量
"""

from .base import VectorStore, VectorStoreConfig, VectorRecord, SearchResult

# 按需导入具体实现类
try:
    from .memory_store import InMemoryVectorStore
except ImportError:
    pass

try:
    from .sqlite_store import SQLiteVectorStore
except ImportError:
    pass

__all__ = [
    'VectorStore',
    'VectorStoreConfig',
    'VectorRecord',
    'SearchResult',
    'InMemoryVectorStore',
    'SQLiteVectorStore'
]
