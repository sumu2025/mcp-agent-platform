"""
RAG引擎 - 提供端到端的检索增强生成功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from datetime import datetime
import json
import time
import os

# 导入依赖模块
from ..embedding import EmbeddingManager
from ..storage import VectorStore
from ..retrieval import (
    KnowledgeRetriever, 
    RetrievalConfig,
    RetrievalResult,
    SimilarityRetriever,
    HybridRetriever
)
from ..augmentation import (
    ContextAugmenter,
    AugmentationConfig,
    AugmentedContext,
    get_augmenter
)

# 设置日志
logger = logging.getLogger(__name__)


class RAGStrategy(Enum):
    """RAG策略枚举"""
    
    BASIC = auto()      # 基本策略
    ADVANCED = auto()   # 高级策略（自动选择检索器和增强器）
    ITERATIVE = auto()  # 迭代策略（多轮检索）
    CUSTOM = auto()     # 自定义策略


@dataclass
class RAGConfig:
    """RAG配置类，定义RAG的各种参数"""
    
    # RAG策略
    strategy: RAGStrategy = RAGStrategy.ADVANCED
    
    # 检索配置
    retrieval_config: Optional[RetrievalConfig] = None
    
    # 增强配置
    augmentation_config: Optional[AugmentationConfig] = None
    
    # LLM配置
    llm_config: Dict[str, Any] = field(default_factory=dict)
    
    # 是否启用日志记录
    enable_logging: bool = True
    
    # 日志目录
    log_dir: Optional[str] = None
    
    # 是否启用缓存
    enable_cache: bool = True
    
    # 缓存大小（条目数）
    cache_size: int = 100
    
    # 自定义参数
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理，确保参数有效"""
        # 设置默认检索配置
        if not self.retrieval_config:
            self.retrieval_config = RetrievalConfig()
            
        # 设置默认增强配置
        if not self.augmentation_config:
            self.augmentation_config = AugmentationConfig()
            
        # 设置默认LLM配置
        if not self.llm_config:
            self.llm_config = {
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
        # 设置默认日志目录
        if self.enable_logging and not self.log_dir:
            self.log_dir = os.path.join(os.path.expanduser("~"), ".mcp", "logs", "rag")
            
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class RAGResult:
    """RAG结果类，表示一个RAG过程的结果"""
    
    # 生成的回答
    answer: str
    
    # 用户查询
    query: str
    
    # 使用的检索结果
    retrieval_results: List[RetrievalResult]
    
    # 增强上下文
    context: AugmentedContext
    
    # 处理时间（秒）
    processing_time: float
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 创建时间戳
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def __post_init__(self):
        """初始化后处理，确保元数据是字典"""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """将RAG结果转换为字典表示"""
        return {
            'answer': self.answer,
            'query': self.query,
            'retrieval_results': [r.to_dict() for r in self.retrieval_results],
            'context': self.context.to_dict(),
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
    
    def save_to_file(self, file_path: str) -> None:
        """
        将RAG结果保存到文件
        
        Args:
            file_path: 文件路径
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class RAGEngine:
    """
    RAG引擎，提供端到端的检索增强生成功能
    """
    
    def __init__(self, 
                vector_store: VectorStore,
                embedding_manager: EmbeddingManager,
                llm_client: Any,
                config: Optional[RAGConfig] = None):
        """
        初始化RAG引擎
        
        Args:
            vector_store: 向量存储
            embedding_manager: 嵌入管理器
            llm_client: LLM客户端
            config: RAG配置，如果为None则使用默认配置
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.config = config or RAGConfig()
        
        # 初始化组件
        self.retriever = None
        self.augmenter = None
        
        # 结果缓存
        self.cache = {}
        self.cache_queue = []
    
    def initialize(self) -> None:
        """初始化引擎"""
        # 初始化向量存储和嵌入管理器
        self.vector_store.ensure_initialized()
        self.embedding_manager.ensure_initialized()
        
        # 根据策略创建检索器
        self._create_retriever()
        
        # 创建上下文增强器
        self._create_augmenter()
        
        # 创建日志目录
        if self.config.enable_logging and self.config.log_dir:
            os.makedirs(self.config.log_dir, exist_ok=True)
            
        logger.info(f"RAG引擎初始化完成，策略: {self.config.strategy.name}")
    
    def _create_retriever(self) -> None:
        """根据策略创建检索器"""
        if self.config.strategy == RAGStrategy.BASIC:
            # 基本策略使用相似度检索器
            self.retriever = SimilarityRetriever(
                vector_store=self.vector_store,
                embedding_manager=self.embedding_manager,
                config=self.config.retrieval_config
            )
        elif self.config.strategy == RAGStrategy.ADVANCED:
            # 高级策略使用混合检索器
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedding_manager=self.embedding_manager,
                config=self.config.retrieval_config
            )
        elif self.config.strategy == RAGStrategy.ITERATIVE:
            # 迭代策略也使用混合检索器，但会在后面的过程中进行多轮检索
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedding_manager=self.embedding_manager,
                config=self.config.retrieval_config
            )
        elif self.config.strategy == RAGStrategy.CUSTOM:
            # 自定义策略，使用外部提供的检索器
            if "retriever" in self.config.custom_params:
                self.retriever = self.config.custom_params["retriever"]
            else:
                logger.warning("自定义策略但未提供检索器，使用混合检索器")
                self.retriever = HybridRetriever(
                    vector_store=self.vector_store,
                    embedding_manager=self.embedding_manager,
                    config=self.config.retrieval_config
                )
        else:
            logger.warning(f"未知的RAG策略: {self.config.strategy}，使用混合检索器")
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedding_manager=self.embedding_manager,
                config=self.config.retrieval_config
            )
        
        # 初始化检索器
        self.retriever.initialize()
    
    def _create_augmenter(self) -> None:
        """创建上下文增强器"""
        if self.config.strategy == RAGStrategy.CUSTOM and "augmenter" in self.config.custom_params:
            # 使用自定义增强器
            self.augmenter = self.config.custom_params["augmenter"]
        else:
            # 使用配置创建增强器
            self.augmenter = get_augmenter(self.config.augmentation_config)
            
        # 初始化增强器
        self.augmenter.initialize()
    
    def generate(self, query: str) -> RAGResult:
        """
        生成回答
        
        Args:
            query: 用户查询
            
        Returns:
            RAG结果
        """
        # 检查缓存
        if self.config.enable_cache:
            cache_key = self._get_cache_key(query)
            if cache_key in self.cache:
                logger.info(f"缓存命中: {query}")
                return self.cache[cache_key]
        
        # 开始计时
        start_time = time.time()
        
        # 根据策略执行RAG流程
        if self.config.strategy == RAGStrategy.ITERATIVE:
            result = self._iterative_rag(query)
        else:
            result = self._basic_rag(query)
            
        # 计算处理时间
        processing_time = time.time() - start_time
        result.processing_time = processing_time
        
        # 记录结果
        if self.config.enable_logging:
            self._log_result(result)
            
        # 缓存结果
        if self.config.enable_cache:
            self._cache_result(query, result)
            
        return result
    
    def _basic_rag(self, query: str) -> RAGResult:
        """
        执行基本RAG流程
        
        Args:
            query: 用户查询
            
        Returns:
            RAG结果
        """
        # 1. 检索相关知识
        retrieval_results = self.retriever.retrieve(query)
        
        # 2. 增强上下文
        context = self.augmenter.augment(query, retrieval_results)
        
        # 3. 生成回答
        answer = self._generate_answer(context.prompt)
        
        # 4. 创建结果
        result = RAGResult(
            answer=answer,
            query=query,
            retrieval_results=retrieval_results,
            context=context,
            processing_time=0.0
        )
        
        return result
    
    def _iterative_rag(self, query: str) -> RAGResult:
        """
        执行迭代RAG流程
        
        Args:
            query: 用户查询
            
        Returns:
            RAG结果
        """
        # 第一轮检索
        retrieval_results = self.retriever.retrieve(query)
        
        # 如果没有结果，直接使用基本RAG
        if not retrieval_results:
            return self._basic_rag(query)
            
        # 增强上下文
        context = self.augmenter.augment(query, retrieval_results)
        
        # 第一轮生成
        first_answer = self._generate_answer(context.prompt, with_thoughts=True)
        
        # 分析回答，提取疑问或需要补充的部分
        follow_up_queries = self._extract_follow_up_queries(first_answer, query)
        
        # 如果没有跟进查询，直接返回第一轮结果
        if not follow_up_queries:
            # 清理思考部分
            final_answer = self._clean_answer(first_answer)
            
            return RAGResult(
                answer=final_answer,
                query=query,
                retrieval_results=retrieval_results,
                context=context,
                processing_time=0.0
            )
            
        # 第二轮检索
        all_results = retrieval_results.copy()
        
        for follow_up in follow_up_queries:
            # 检索补充信息
            additional_results = self.retriever.retrieve(follow_up)
            all_results.extend(additional_results)
            
        # 去重
        unique_results = []
        seen_ids = set()
        
        for result in all_results:
            if result.id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.id)
                
        # 根据相关性排序
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # 限制结果数量
        top_results = unique_results[:self.config.retrieval_config.top_k]
        
        # 增强上下文
        final_context = self.augmenter.augment(query, top_results)
        
        # 最终生成
        final_answer = self._generate_answer(final_context.prompt)
        
        # 创建结果
        result = RAGResult(
            answer=final_answer,
            query=query,
            retrieval_results=top_results,
            context=final_context,
            processing_time=0.0
        )
        
        return result
    
    def _generate_answer(self, prompt: str, with_thoughts: bool = False) -> str:
        """
        生成回答
        
        Args:
            prompt: 提示
            with_thoughts: 是否包含思考过程
            
        Returns:
            生成的回答
        """
        # 这里需要根据实际使用的LLM客户端实现
        # 以下是一个通用示例
        
        try:
            # 设置参数
            params = self.config.llm_config.copy()
            
            if with_thoughts:
                # 如果需要包含思考过程，添加相应指令
                prompt = (
                    prompt + "\n\n首先思考这个问题需要哪些信息，哪些信息已有，"
                    "哪些信息缺失。思考用<thinking>标签标记，不会展示给用户。"
                )
            
            # 调用LLM客户端
            response = self.llm_client.generate(prompt, **params)
            
            # 提取回答
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                # 尝试转换为字符串
                return str(response)
                
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            return f"很抱歉，生成回答时出现错误: {str(e)}"
    
    def _extract_follow_up_queries(self, answer: str, original_query: str) -> List[str]:
        """
        从回答中提取跟进查询
        
        Args:
            answer: 生成的回答
            original_query: 原始查询
            
        Returns:
            跟进查询列表
        """
        # 提取思考部分
        thoughts = ""
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', answer, re.DOTALL)
        if thinking_match:
            thoughts = thinking_match.group(1).strip()
        
        if not thoughts:
            return []
            
        # 在思考中寻找表示信息缺失的关键词
        keywords = ["缺少", "需要", "没有", "缺乏", "不清楚", "不确定", "不知道"]
        
        follow_ups = []
        
        for keyword in keywords:
            if keyword in thoughts:
                # 创建一个针对缺失信息的查询
                sentences = [s for s in thoughts.split('。') if keyword in s]
                for sentence in sentences:
                    # 基于句子创建查询
                    query = self._create_follow_up_query(sentence, original_query)
                    if query:
                        follow_ups.append(query)
        
        # 去重和限制数量
        unique_follow_ups = list(set(follow_ups))
        return unique_follow_ups[:2]  # 最多2个跟进查询
    
    def _create_follow_up_query(self, sentence: str, original_query: str) -> Optional[str]:
        """
        基于句子创建跟进查询
        
        Args:
            sentence: 包含信息需求的句子
            original_query: 原始查询
            
        Returns:
            跟进查询，如果无法创建则为None
        """
        # 简单拼接
        clean_sentence = re.sub(r'[，。！？]', '', sentence)
        return f"{original_query} {clean_sentence}"
    
    def _clean_answer(self, answer: str) -> str:
        """
        清理答案，移除思考部分
        
        Args:
            answer: 原始答案
            
        Returns:
            清理后的答案
        """
        # 移除<thinking>标签
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', answer, flags=re.DOTALL)
        return cleaned.strip()
    
    def _get_cache_key(self, query: str) -> str:
        """
        生成缓存键
        
        Args:
            query: 用户查询
            
        Returns:
            缓存键
        """
        # 使用查询的哈希作为缓存键
        return hashlib.md5(query.encode()).hexdigest()
    
    def _cache_result(self, query: str, result: RAGResult) -> None:
        """
        缓存结果
        
        Args:
            query: 用户查询
            result: RAG结果
        """
        cache_key = self._get_cache_key(query)
        
        # 如果缓存已满，移除最旧的条目
        if len(self.cache) >= self.config.cache_size:
            if self.cache_queue:
                oldest_key = self.cache_queue.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
        
        # 添加到缓存
        self.cache[cache_key] = result
        self.cache_queue.append(cache_key)
    
    def _log_result(self, result: RAGResult) -> None:
        """
        记录RAG结果
        
        Args:
            result: RAG结果
        """
        if not self.config.log_dir:
            return
            
        try:
            # 创建日志目录
            os.makedirs(self.config.log_dir, exist_ok=True)
            
            # 生成日志文件名
            timestamp = datetime.fromtimestamp(result.timestamp).strftime("%Y%m%d_%H%M%S")
            file_name = f"rag_result_{timestamp}.json"
            file_path = os.path.join(self.config.log_dir, file_name)
            
            # 保存结果
            result.save_to_file(file_path)
            
        except Exception as e:
            logger.error(f"记录RAG结果失败: {str(e)}")
