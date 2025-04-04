"""
嵌入管理模块 - 负责管理文本到向量的转换过程
"""

from .base import EmbeddingManager, EmbeddingConfig, TextEmbedding

# 按需导入具体实现类
try:
    from .local_embedding import LocalEmbedding
except ImportError:
    # sentence-transformers可能未安装
    pass

try:
    from .api_embedding import OpenAIEmbedding
except ImportError:
    # openai可能未安装
    pass

from .cache import EmbeddingCache, DiskEmbeddingCache, SQLiteEmbeddingCache

__all__ = [
    'EmbeddingManager',
    'EmbeddingConfig',
    'TextEmbedding',
    'LocalEmbedding',
    'OpenAIEmbedding',
    'EmbeddingCache',
    'DiskEmbeddingCache',
    'SQLiteEmbeddingCache'
]
