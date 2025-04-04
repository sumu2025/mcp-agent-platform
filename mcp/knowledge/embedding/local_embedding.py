"""
本地嵌入管理器 - 使用sentence-transformers在本地生成嵌入
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import torch
import logging

from .base import EmbeddingManager, EmbeddingConfig, TextEmbedding
from .cache import EmbeddingCache, DiskEmbeddingCache

# 设置日志
logger = logging.getLogger(__name__)


class LocalEmbedding(EmbeddingManager):
    """
    本地嵌入管理器，使用sentence-transformers在本地生成嵌入
    """
    
    def __init__(self, 
                 config: Optional[EmbeddingConfig] = None,
                 cache: Optional[EmbeddingCache] = None):
        """
        初始化本地嵌入管理器
        
        Args:
            config: 嵌入配置，如果为None则使用默认配置
            cache: 嵌入缓存，如果为None且配置启用缓存则创建默认缓存
        """
        super().__init__(config)
        
        self.model = None
        self.device = None
        
        # 设置缓存
        if cache:
            self.cache = cache
        elif config and config.use_cache:
            cache_dir = config.cache_dir or os.path.join(os.path.expanduser("~"), ".mcp", "embedding_cache")
            self.cache = DiskEmbeddingCache(cache_dir=cache_dir)
        else:
            self.cache = None
    
    def initialize(self) -> None:
        """初始化嵌入管理器（加载模型）"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "请安装sentence-transformers库以使用本地嵌入: "
                "pip install sentence-transformers"
            )
        
        # 确定设备
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("使用GPU进行嵌入计算")
        else:
            self.device = torch.device("cpu")
            if self.config.use_gpu:
                logger.warning("GPU不可用，使用CPU进行嵌入计算")
            else:
                logger.info("使用CPU进行嵌入计算")
        
        # 加载模型
        logger.info(f"加载嵌入模型: {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name, device=str(self.device))
        
        # 初始化缓存
        if self.cache:
            self.cache.initialize()
            
        self._initialized = True
        logger.info(f"本地嵌入管理器初始化完成，模型: {self.config.model_name}, 嵌入维度: {self.get_embedding_dim()}")
    
    def embed_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> TextEmbedding:
        """
        嵌入单个文本
        
        Args:
            text: 要嵌入的文本
            metadata: 文本相关元数据
            
        Returns:
            文本嵌入
        """
        self.ensure_initialized()
        
        # 检查缓存
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                logger.debug(f"使用缓存的嵌入: {text[:20]}...")
                return TextEmbedding(text=text, embedding=cached, metadata=metadata)
        
        # 生成嵌入
        embedding = self._generate_embedding(text)
        
        # 缓存嵌入
        if self.cache:
            self.cache.set(text, embedding)
        
        return TextEmbedding(text=text, embedding=embedding, metadata=metadata)
    
    def embed_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[TextEmbedding]:
        """
        批量嵌入多个文本
        
        Args:
            texts: 要嵌入的文本列表
            metadatas: 文本相关元数据列表
            
        Returns:
            文本嵌入列表
        """
        self.ensure_initialized()
        
        if not texts:
            return []
        
        if metadatas and len(texts) != len(metadatas):
            raise ValueError("texts和metadatas长度必须相同")
        
        metadatas = metadatas or [None] * len(texts)
        
        # 检查缓存
        if self.cache:
            cached_embeddings = {}
            for text in texts:
                cached = self.cache.get(text)
                if cached is not None:
                    cached_embeddings[text] = cached
            
            # 如果所有嵌入都在缓存中，直接返回
            if len(cached_embeddings) == len(texts):
                logger.debug("所有嵌入都在缓存中")
                return [
                    TextEmbedding(text=text, embedding=cached_embeddings[text], metadata=metadata)
                    for text, metadata in zip(texts, metadatas)
                ]
            
            # 否则，只嵌入未缓存的文本
            texts_to_embed = [text for text in texts if text not in cached_embeddings]
            logger.debug(f"需要生成嵌入: {len(texts_to_embed)}/{len(texts)}个文本")
        else:
            texts_to_embed = texts
            cached_embeddings = {}
        
        # 批量生成嵌入
        embeddings = self._generate_embeddings(texts_to_embed)
        
        # 缓存新生成的嵌入
        if self.cache:
            for text, embedding in zip(texts_to_embed, embeddings):
                self.cache.set(text, embedding)
        
        # 组合缓存的和新生成的嵌入
        result = []
        for text, metadata in zip(texts, metadatas):
            if text in cached_embeddings:
                embedding = cached_embeddings[text]
            else:
                idx = texts_to_embed.index(text)
                embedding = embeddings[idx]
            
            result.append(TextEmbedding(text=text, embedding=embedding, metadata=metadata))
        
        return result
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        嵌入查询文本
        
        Args:
            query: 查询文本
            
        Returns:
            查询嵌入向量
        """
        self.ensure_initialized()
        
        # 查询嵌入通常不缓存，因为它们是动态的
        return self._generate_embedding(query)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数（0-1）
        """
        self.ensure_initialized()
        
        embedding1 = self.embed_text(text1).embedding
        embedding2 = self.embed_text(text2).embedding
        
        # 计算余弦相似度
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def get_embedding_dim(self) -> int:
        """
        获取嵌入维度
        
        Returns:
            嵌入维度
        """
        self.ensure_initialized()
        return self.model.get_sentence_embedding_dimension()
    
    def get_manager_name(self) -> str:
        """
        获取管理器名称
        
        Returns:
            管理器名称
        """
        return "local_embedding"
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        生成单个文本的嵌入
        
        Args:
            text: 要嵌入的文本
            
        Returns:
            嵌入向量
        """
        if self.config.max_length and len(text) > self.config.max_length:
            logger.warning(f"文本长度({len(text)})超过最大长度({self.config.max_length})，将被截断")
            text = text[:self.config.max_length]
        
        # 生成嵌入
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embedding
    
    def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        批量生成多个文本的嵌入
        
        Args:
            texts: 要嵌入的文本列表
            
        Returns:
            嵌入向量列表
        """
        if self.config.max_length:
            # 截断过长的文本
            texts = [
                text[:self.config.max_length] if len(text) > self.config.max_length else text
                for text in texts
            ]
        
        # 批量生成嵌入
        batch_size = self.config.batch_size
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            logger.debug(f"处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}, 大小: {len(batch_texts)}")
            
            batch_embeddings = self.model.encode(
                batch_texts,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
                batch_size=batch_size
            )
            
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
