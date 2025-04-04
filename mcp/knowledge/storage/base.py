"""
向量存储基类 - 定义向量存储的通用接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import uuid
import logging
import json
from datetime import datetime

# 设置日志
logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """向量存储配置类，定义存储的各种参数"""
    
    # 嵌入维度
    embedding_dim: int = 384
    
    # 索引类型 (flat, hnsw, ivf)
    index_type: str = "flat"
    
    # 距离度量 (cosine, l2, dot)
    distance_metric: str = "cosine"
    
    # 存储路径（如果适用）
    storage_path: Optional[str] = None
    
    # 是否使用元数据过滤
    use_metadata_filter: bool = True
    
    # 自定义索引参数
    index_params: Dict[str, Any] = field(default_factory=dict)
    
    # 是否使用归一化向量
    normalize_vectors: bool = True
    
    def __post_init__(self):
        """初始化后处理，确保索引参数是字典"""
        if self.index_params is None:
            self.index_params = {}


@dataclass
class VectorRecord:
    """向量记录类，表示一个存储的向量及其元数据"""
    
    # 唯一标识符
    id: str
    
    # 向量表示
    embedding: np.ndarray
    
    # 原始文本（可选）
    text: Optional[str] = None
    
    # 元数据字典
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 创建时间
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def __post_init__(self):
        """初始化后处理，确保元数据是字典"""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """将记录转换为字典表示"""
        return {
            'id': self.id,
            'embedding': self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            'text': self.text,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorRecord':
        """从字典创建记录"""
        # 确保embedding是numpy数组
        embedding = data['embedding']
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
            
        return cls(
            id=data['id'],
            embedding=embedding,
            text=data.get('text'),
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at', datetime.now().timestamp())
        )


@dataclass
class SearchResult:
    """搜索结果类，表示一个搜索匹配"""
    
    # 匹配的向量记录
    record: VectorRecord
    
    # 相似度得分
    score: float
    
    # 匹配索引
    index: Optional[int] = None
    
    # 额外信息
    info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理，确保info是字典"""
        if self.info is None:
            self.info = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """将搜索结果转换为字典表示"""
        return {
            'record': self.record.to_dict(),
            'score': self.score,
            'index': self.index,
            'info': self.info
        }


class VectorStore(ABC):
    """
    向量存储抽象基类，定义向量存储的通用接口
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        初始化向量存储
        
        Args:
            config: 存储配置，如果为None则使用默认配置
        """
        self.config = config or VectorStoreConfig()
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """
        初始化向量存储（创建索引等）
        """
        pass
    
    def ensure_initialized(self) -> None:
        """确保向量存储已初始化"""
        if not self._initialized:
            self.initialize()
            self._initialized = True
    
    @abstractmethod
    def add(self, record: VectorRecord) -> str:
        """
        添加单个向量记录
        
        Args:
            record: 向量记录
            
        Returns:
            记录ID
        """
        pass
    
    @abstractmethod
    def add_batch(self, records: List[VectorRecord]) -> List[str]:
        """
        批量添加向量记录
        
        Args:
            records: 向量记录列表
            
        Returns:
            记录ID列表
        """
        pass
    
    @abstractmethod
    def search(self, 
              query_vector: np.ndarray, 
              k: int = 10, 
              filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            k: 返回结果数量
            filter: 元数据过滤条件
            
        Returns:
            搜索结果列表
        """
        pass
    
    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """
        删除向量记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            是否成功删除
        """
        pass
    
    @abstractmethod
    def get(self, record_id: str) -> Optional[VectorRecord]:
        """
        获取向量记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            向量记录，如果不存在则为None
        """
        pass
    
    @abstractmethod
    def update(self, record: VectorRecord) -> bool:
        """
        更新向量记录
        
        Args:
            record: 更新后的向量记录
            
        Returns:
            是否成功更新
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        获取存储的向量数量
        
        Returns:
            向量数量
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        清空向量存储
        """
        pass
    
    @abstractmethod
    def save(self) -> None:
        """
        保存向量存储（如果支持持久化）
        """
        pass
    
    @abstractmethod
    def load(self) -> None:
        """
        加载向量存储（如果支持持久化）
        """
        pass
    
    @abstractmethod
    def get_store_info(self) -> Dict[str, Any]:
        """
        获取存储信息
        
        Returns:
            存储信息字典
        """
        pass
    
    def generate_record_id(self) -> str:
        """
        生成唯一记录ID
        
        Returns:
            记录ID
        """
        return str(uuid.uuid4())
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        归一化向量
        
        Args:
            vector: 原始向量
            
        Returns:
            归一化后的向量
        """
        if self.config.normalize_vectors:
            norm = np.linalg.norm(vector)
            if norm > 0:
                return vector / norm
        return vector
    
    def _check_filter(self, record: VectorRecord, filter: Dict[str, Any]) -> bool:
        """
        检查记录是否满足过滤条件
        
        Args:
            record: 向量记录
            filter: 过滤条件
            
        Returns:
            是否满足条件
        """
        if not filter:
            return True
            
        for key, value in filter.items():
            # 处理嵌套字段 (field.subfield)
            if '.' in key:
                parts = key.split('.')
                curr = record.metadata
                for part in parts[:-1]:
                    if part not in curr:
                        return False
                    curr = curr[part]
                
                if parts[-1] not in curr or curr[parts[-1]] != value:
                    return False
            
            # 处理直接字段
            elif key not in record.metadata or record.metadata[key] != value:
                return False
                
        return True
