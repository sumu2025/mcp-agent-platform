"""
API嵌入管理器 - 使用外部API服务生成嵌入
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import EmbeddingManager, EmbeddingConfig, TextEmbedding
from .cache import EmbeddingCache, DiskEmbeddingCache

# 设置日志
logger = logging.getLogger(__name__)


class OpenAIEmbedding(EmbeddingManager):
    """
    OpenAI嵌入管理器，使用OpenAI API生成嵌入
    """
    
    OPENAI_EMBEDDING_MODELS = {
        # 模型名称: (维度, 最大tokens)
        "text-embedding-3-small": (1536, 8191),
        "text-embedding-3-large": (3072, 8191),
        "text-embedding-ada-002": (1536, 8191)
    }
    
    def __init__(self, 
                 config: Optional[EmbeddingConfig] = None,
                 cache: Optional[EmbeddingCache] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: int = 60):
        """
        初始化OpenAI嵌入管理器
        
        Args:
            config: 嵌入配置，如果为None则使用默认配置
            cache: 嵌入缓存，如果为None且配置启用缓存则创建默认缓存
            api_key: OpenAI API密钥，如果为None则从环境变量OPENAI_API_KEY获取
            base_url: API基础URL，如果为None则使用默认URL
            timeout: API请求超时时间（秒）
        """
        if not config:
            config = EmbeddingConfig(
                model_name="text-embedding-3-small",
                embedding_dim=1536,
                batch_size=16,
                use_cache=True
            )
        
        super().__init__(config)
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("未提供OpenAI API密钥，请设置环境变量OPENAI_API_KEY或在初始化时提供api_key参数")
            
        self.base_url = base_url or "https://api.openai.com/v1"
        self.timeout = timeout
        
        # 设置缓存
        if cache:
            self.cache = cache
        elif config and config.use_cache:
            cache_dir = config.cache_dir or os.path.join(os.path.expanduser("~"), ".mcp", "embedding_cache")
            self.cache = DiskEmbeddingCache(cache_dir=cache_dir)
        else:
            self.cache = None
    
    def initialize(self) -> None:
        """初始化嵌入管理器"""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "请安装openai库以使用OpenAI嵌入: "
                "pip install openai"
            )
        
        if not self.api_key:
            raise ValueError("OpenAI API密钥未设置")
        
        # 验证模型名称
        if self.config.model_name not in self.OPENAI_EMBEDDING_MODELS:
            logger.warning(
                f"未知的OpenAI嵌入模型: {self.config.model_name}, "
                f"已知模型: {', '.join(self.OPENAI_EMBEDDING_MODELS.keys())}"
            )
        
        # 初始化嵌入维度
        if self.config.model_name in self.OPENAI_EMBEDDING_MODELS:
            self.config.embedding_dim = self.OPENAI_EMBEDDING_MODELS[self.config.model_name][0]
        
        # 初始化缓存
        if self.cache:
            self.cache.initialize()
            
        self._initialized = True
        logger.info(f"OpenAI嵌入管理器初始化完成，模型: {self.config.model_name}, 嵌入维度: {self.get_embedding_dim()}")
    
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
        return self.config.embedding_dim
    
    def get_manager_name(self) -> str:
        """
        获取管理器名称
        
        Returns:
            管理器名称
        """
        return "openai_embedding"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        生成单个文本的嵌入
        
        Args:
            text: 要嵌入的文本
            
        Returns:
            嵌入向量
        """
        import openai
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        try:
            response = client.embeddings.create(
                model=self.config.model_name,
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            
            # 根据配置规范化嵌入
            if self.config.normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
                
            return embedding
            
        except Exception as e:
            logger.error(f"OpenAI嵌入生成错误: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        批量生成多个文本的嵌入
        
        Args:
            texts: 要嵌入的文本列表
            
        Returns:
            嵌入向量列表
        """
        import openai
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # 分批处理
        batch_size = self.config.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            logger.debug(f"处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}, 大小: {len(batch_texts)}")
            
            try:
                response = client.embeddings.create(
                    model=self.config.model_name,
                    input=batch_texts
                )
                
                # 确保嵌入按原始顺序返回
                embeddings = [np.array(item.embedding) for item in response.data]
                
                # 根据配置规范化嵌入
                if self.config.normalize_embeddings:
                    embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]
                
                all_embeddings.extend(embeddings)
                
                # 避免API速率限制
                if i + batch_size < len(texts):
                    time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"OpenAI批量嵌入生成错误: {str(e)}")
                raise
        
        return all_embeddings
