"""
上下文增强基类 - 定义上下文增强的通用接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from datetime import datetime

# 导入依赖模块
from ..retrieval import RetrievalResult

# 设置日志
logger = logging.getLogger(__name__)


class AugmentationMode(Enum):
    """上下文增强模式枚举"""
    
    BASIC = auto()       # 基本模式，简单拼接
    STRUCTURED = auto()  # 结构化模式，按部分组织
    TEMPLATE = auto()    # 模板模式，使用预定义模板


@dataclass
class AugmentationConfig:
    """上下文增强配置类，定义增强的各种参数"""
    
    # 增强模式
    mode: AugmentationMode = AugmentationMode.STRUCTURED
    
    # 最大上下文长度（单位：tokens）
    max_context_length: int = 4000
    
    # 系统指令部分的最大长度
    max_system_length: int = 1000
    
    # 检索结果部分的最大长度
    max_retrieval_length: int = 2000
    
    # 用户查询部分的最大长度
    max_query_length: int = 500
    
    # 是否包含元数据
    include_metadata: bool = True
    
    # 元数据包含字段
    metadata_fields: List[str] = field(default_factory=list)
    
    # 检索结果格式化模板
    retrieval_format: str = "{content}"
    
    # 系统指令模板
    system_template: str = "使用以下参考信息回答用户问题。如果参考信息无法回答问题，请基于你的知识谨慎回答。\n\n{retrieval_results}"
    
    # 是否启用Token管理
    use_token_management: bool = True
    
    # 优先级排序（在超出长度时保留哪些部分）
    priority_order: List[str] = field(default_factory=lambda: ["system", "query", "retrieval"])
    
    # 自定义参数
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理，确保参数有效"""
        if self.max_context_length < 100:
            logger.warning(f"max_context_length值 {self.max_context_length} 过小，已设置为默认值 4000")
            self.max_context_length = 4000
            
        if not self.metadata_fields:
            # 默认包含的元数据字段
            self.metadata_fields = ["category", "source", "title", "date", "author"]
            
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class AugmentedContext:
    """增强上下文类，表示一个增强后的上下文"""
    
    # 完整的提示文本
    prompt: str
    
    # 系统指令部分
    system: str
    
    # 检索结果部分
    retrieval: str
    
    # 用户查询部分
    query: str
    
    # 使用的检索结果
    retrieval_results: List[RetrievalResult]
    
    # Token计数
    token_count: Dict[str, int] = field(default_factory=dict)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 创建时间戳
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def __post_init__(self):
        """初始化后处理，确保元数据是字典"""
        if self.metadata is None:
            self.metadata = {}
        
        if not self.token_count:
            self.token_count = {
                "system": 0,
                "retrieval": 0,
                "query": 0,
                "total": 0
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """将增强上下文转换为字典表示"""
        return {
            'prompt': self.prompt,
            'system': self.system,
            'retrieval': self.retrieval,
            'query': self.query,
            'retrieval_results': [r.to_dict() for r in self.retrieval_results],
            'token_count': self.token_count,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


class ContextAugmenter(ABC):
    """
    上下文增强器抽象基类，定义上下文增强的通用接口
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        初始化上下文增强器
        
        Args:
            config: 增强配置，如果为None则使用默认配置
        """
        self.config = config or AugmentationConfig()
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """
        初始化上下文增强器
        """
        pass
    
    def ensure_initialized(self) -> None:
        """确保增强器已初始化"""
        if not self._initialized:
            self.initialize()
            self._initialized = True
    
    @abstractmethod
    def augment(self, 
               query: str, 
               retrieval_results: List[RetrievalResult]) -> AugmentedContext:
        """
        增强上下文
        
        Args:
            query: 用户查询
            retrieval_results: 检索结果列表
            
        Returns:
            增强后的上下文
        """
        pass
    
    @abstractmethod
    def format_retrieval_results(self, 
                                retrieval_results: List[RetrievalResult]) -> str:
        """
        格式化检索结果
        
        Args:
            retrieval_results: 检索结果列表
            
        Returns:
            格式化后的文本
        """
        pass
    
    @abstractmethod
    def build_system_prompt(self, 
                           retrieval_text: str,
                           query: str) -> str:
        """
        构建系统提示
        
        Args:
            retrieval_text: 格式化后的检索结果文本
            query: 用户查询
            
        Returns:
            系统提示文本
        """
        pass
    
    @abstractmethod
    def build_user_prompt(self, query: str) -> str:
        """
        构建用户提示
        
        Args:
            query: 用户查询
            
        Returns:
            用户提示文本
        """
        pass
    
    @abstractmethod
    def manage_context_length(self, 
                             system: str,
                             retrieval: str,
                             query: str) -> Tuple[str, str, str]:
        """
        管理上下文长度，确保不超过最大长度
        
        Args:
            system: 系统指令部分
            retrieval: 检索结果部分
            query: 用户查询部分
            
        Returns:
            调整后的(系统指令, 检索结果, 用户查询)元组
        """
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        获取文本的token数量
        
        Args:
            text: 文本
            
        Returns:
            token数量
        """
        pass
    
    @abstractmethod
    def get_augmenter_type(self) -> str:
        """
        获取增强器类型
        
        Returns:
            增强器类型
        """
        pass
    
    @abstractmethod
    def get_augmenter_info(self) -> Dict[str, Any]:
        """
        获取增强器信息
        
        Returns:
            增强器信息字典
        """
        pass
