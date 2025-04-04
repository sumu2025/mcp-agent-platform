"""
嵌入管理器基类 - 定义嵌入管理的通用接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import hashlib
import json


@dataclass
class EmbeddingConfig:
    """嵌入配置类，定义嵌入过程的各种参数"""
    
    # 嵌入维度
    embedding_dim: int = 384
    
    # 批处理大小
    batch_size: int = 16
    
    # 是否缓存嵌入
    use_cache: bool = True
    
    # 缓存目录
    cache_dir: Optional[str] = None
    
    # 最大文本长度，超过将被截断
    max_length: Optional[int] = None
    
    # 模型名称或路径
    model_name: str = "paraphrase-MiniLM-L6-v2"
    
    # 是否使用GPU加速
    use_gpu: bool = True
    
    # 是否规范化向量
    normalize_embeddings: bool = True
    
    # 自定义参数
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化后处理，确保自定义参数是字典"""
        if self.custom_params is None:
            self.custom_params = {}


class TextEmbedding:
    """文本嵌入类，表示一个文本及其向量表示"""
    
    def __init__(
        self,
        text: str,
        embedding: np.ndarray,
        text_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化文本嵌入
        
        Args:
            text: 文本内容
            embedding: 向量表示
            text_id: 文本ID，如果为None则自动生成
            metadata: 相关元数据
        """
        self.text = text
        self.embedding = embedding
        self.text_id = text_id or self._generate_id(text)
        self.metadata = metadata or {}
        
    def _generate_id(self, text: str) -> str:
        """根据文本内容生成唯一ID"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def cosine_similarity(self, other: 'TextEmbedding') -> float:
        """计算与另一个嵌入的余弦相似度"""
        return np.dot(self.embedding, other.embedding) / (
            np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """将嵌入转换为字典表示"""
        return {
            'text': self.text,
            'embedding': self.embedding.tolist(),
            'text_id': self.text_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextEmbedding':
        """从字典创建嵌入"""
        return cls(
            text=data['text'],
            embedding=np.array(data['embedding']),
            text_id=data.get('text_id'),
            metadata=data.get('metadata')
        )
    
    def __repr__(self) -> str:
        """嵌入的字符串表示"""
        preview = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f"TextEmbedding(id={self.text_id}, text='{preview}', dim={self.embedding.shape[0]})"


class EmbeddingManager(ABC):
    """
    嵌入管理器抽象基类，定义嵌入管理的通用接口
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        初始化嵌入管理器
        
        Args:
            config: 嵌入配置，如果为None则使用默认配置
        """
        self.config = config or EmbeddingConfig()
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """
        初始化嵌入管理器（加载模型等）
        """
        pass
    
    def ensure_initialized(self) -> None:
        """确保嵌入管理器已初始化"""
        if not self._initialized:
            self.initialize()
            self._initialized = True
    
    @abstractmethod
    def embed_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> TextEmbedding:
        """
        嵌入单个文本
        
        Args:
            text: 要嵌入的文本
            metadata: 文本相关元数据
            
        Returns:
            文本嵌入
        """
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[TextEmbedding]:
        """
        批量嵌入多个文本
        
        Args:
            texts: 要嵌入的文本列表
            metadatas: 文本相关元数据列表
            
        Returns:
            文本嵌入列表
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        嵌入查询文本
        
        Args:
            query: 查询文本
            
        Returns:
            查询嵌入向量
        """
        pass
    
    @abstractmethod
    def similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数（0-1）
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        获取嵌入维度
        
        Returns:
            嵌入维度
        """
        pass
    
    @abstractmethod
    def get_manager_name(self) -> str:
        """
        获取管理器名称
        
        Returns:
            管理器名称
        """
        pass
