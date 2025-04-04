"""
知识检索基类 - 定义知识检索的通用接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import logging
from datetime import datetime

# 导入依赖模块
from ..storage import VectorStore, SearchResult, VectorRecord
from ..embedding import EmbeddingManager, TextEmbedding

# 设置日志
logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """检索策略枚举"""
    
    SIMILARITY = auto()  # 基于向量相似度检索
    KEYWORD = auto()     # 基于关键词检索
    HYBRID = auto()      # 混合检索（结合向量和关键词）
    ENSEMBLE = auto()    # 集成检索（多种检索策略结合）
    RERANKING = auto()   # 重排序检索（先检索后重排）


@dataclass
class RetrievalConfig:
    """检索配置类，定义检索的各种参数"""
    
    # 检索策略
    strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY
    
    # 检索数量
    top_k: int = 5
    
    # 相似度阈值 (0-1)，低于阈值的结果将被过滤
    similarity_threshold: float = 0.0
    
    # 混合检索的权重 (向量相似度权重)
    hybrid_weight: float = 0.7
    
    # 是否删除重复内容
    deduplicate: bool = True
    
    # 去重相似度阈值，高于此值的文档被视为重复
    dedupe_threshold: float = 0.98
    
    # 元数据过滤器
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    
    # 内容过滤函数
    content_filter: Optional[Callable[[str], bool]] = None
    
    # 是否包含文档元数据
    include_metadata: bool = True
    
    # 是否返回原始检索结果
    return_raw_results: bool = False
    
    # 最大文本长度限制
    max_text_length: Optional[int] = None
    
    # 自定义参数
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理，确保参数有效"""
        if self.top_k < 1:
            logger.warning(f"top_k值 {self.top_k} 无效，已设置为默认值 5")
            self.top_k = 5
            
        if not 0 <= self.similarity_threshold <= 1:
            logger.warning(f"similarity_threshold值 {self.similarity_threshold} 无效，已设置为 0")
            self.similarity_threshold = 0
            
        if not 0 <= self.hybrid_weight <= 1:
            logger.warning(f"hybrid_weight值 {self.hybrid_weight} 无效，已设置为 0.7")
            self.hybrid_weight = 0.7
            
        if not 0 <= self.dedupe_threshold <= 1:
            logger.warning(f"dedupe_threshold值 {self.dedupe_threshold} 无效，已设置为 0.98")
            self.dedupe_threshold = 0.98
            
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class RetrievalResult:
    """检索结果类，表示一个检索结果"""
    
    # 结果文本
    text: str
    
    # 相似度得分 (0-1)
    score: float
    
    # 结果ID
    id: str
    
    # 元数据字典
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 原始搜索结果
    raw_result: Optional[Any] = None
    
    # 检索来源
    source: str = "unknown"
    
    # 检索时间戳
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def __post_init__(self):
        """初始化后处理，确保元数据是字典"""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """将检索结果转换为字典表示"""
        return {
            'text': self.text,
            'score': self.score,
            'id': self.id,
            'metadata': self.metadata,
            'source': self.source,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_search_result(cls, 
                          search_result: SearchResult, 
                          source: str = "vector_store") -> 'RetrievalResult':
        """从向量存储搜索结果创建检索结果"""
        return cls(
            text=search_result.record.text or "",
            score=search_result.score,
            id=search_result.record.id,
            metadata=search_result.record.metadata,
            raw_result=search_result,
            source=source
        )


class KnowledgeRetriever(ABC):
    """
    知识检索器抽象基类，定义知识检索的通用接口
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        初始化知识检索器
        
        Args:
            config: 检索配置，如果为None则使用默认配置
        """
        self.config = config or RetrievalConfig()
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """
        初始化知识检索器
        """
        pass
    
    def ensure_initialized(self) -> None:
        """确保检索器已初始化"""
        if not self._initialized:
            self.initialize()
            self._initialized = True
    
    @abstractmethod
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        根据查询检索知识
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果列表
        """
        pass
    
    @abstractmethod
    def retrieve_with_filter(self, 
                           query: str, 
                           filter: Dict[str, Any]) -> List[RetrievalResult]:
        """
        使用过滤器检索知识
        
        Args:
            query: 查询文本
            filter: 元数据过滤条件
            
        Returns:
            检索结果列表
        """
        pass
    
    @abstractmethod
    def get_relevant_documents(self, 
                              query: str, 
                              k: Optional[int] = None) -> List[str]:
        """
        获取与查询相关的文档文本
        
        Args:
            query: 查询文本
            k: 返回结果数量，如果为None则使用配置值
            
        Returns:
            相关文档文本列表
        """
        pass
    
    def deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        去除重复结果
        
        Args:
            results: 检索结果列表
            
        Returns:
            去重后的结果列表
        """
        if not self.config.deduplicate or len(results) <= 1:
            return results
            
        # 贪婪算法：按相似度排序，依次添加不重复的结果
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        deduplicated = [sorted_results[0]]
        
        for result in sorted_results[1:]:
            is_duplicate = False
            
            for added_result in deduplicated:
                # 计算文本相似度（使用简单的Jaccard相似度）
                similarity = self._calculate_text_similarity(
                    result.text, added_result.text
                )
                
                if similarity > self.config.dedupe_threshold:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                deduplicated.append(result)
                
        return deduplicated
    
    def filter_by_score(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        根据相似度分数过滤结果
        
        Args:
            results: 检索结果列表
            
        Returns:
            过滤后的结果列表
        """
        if self.config.similarity_threshold <= 0:
            return results
            
        return [
            result for result in results 
            if result.score >= self.config.similarity_threshold
        ]
    
    def filter_by_content(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        使用内容过滤函数过滤结果
        
        Args:
            results: 检索结果列表
            
        Returns:
            过滤后的结果列表
        """
        if not self.config.content_filter:
            return results
            
        return [
            result for result in results 
            if self.config.content_filter(result.text)
        ]
    
    def limit_text_length(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        限制结果文本长度
        
        Args:
            results: 检索结果列表
            
        Returns:
            处理后的结果列表
        """
        if not self.config.max_text_length:
            return results
            
        for result in results:
            if len(result.text) > self.config.max_text_length:
                result.text = result.text[:self.config.max_text_length] + "..."
                
        return results
    
    def process_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        处理检索结果
        
        Args:
            results: 原始检索结果列表
            
        Returns:
            处理后的结果列表
        """
        # 按相似度分数过滤
        filtered_results = self.filter_by_score(results)
        
        # 按内容过滤
        filtered_results = self.filter_by_content(filtered_results)
        
        # 去重
        if self.config.deduplicate:
            filtered_results = self.deduplicate_results(filtered_results)
        
        # 限制文本长度
        filtered_results = self.limit_text_length(filtered_results)
        
        # 限制结果数量
        if len(filtered_results) > self.config.top_k:
            filtered_results = filtered_results[:self.config.top_k]
        
        return filtered_results
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（Jaccard相似度）
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度 (0-1)
        """
        # 将文本分割为单词集合
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        # 计算Jaccard相似度
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0
            
        return intersection / union
    
    @abstractmethod
    def get_retriever_type(self) -> str:
        """
        获取检索器类型
        
        Returns:
            检索器类型
        """
        pass
        
    @abstractmethod
    def get_retriever_info(self) -> Dict[str, Any]:
        """
        获取检索器信息
        
        Returns:
            检索器信息字典
        """
        pass
