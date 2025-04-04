"""
知识检索器实现 - 提供各种知识检索策略的具体实现
"""

import re
import os
import time
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from collections import Counter, defaultdict
import json
import math
from datetime import datetime

# 导入依赖模块
from .base import (
    KnowledgeRetriever, 
    RetrievalConfig, 
    RetrievalResult, 
    RetrievalStrategy
)
from ..storage import VectorStore, SearchResult, VectorRecord
from ..embedding import EmbeddingManager, TextEmbedding

# 设置日志
logger = logging.getLogger(__name__)


class SimilarityRetriever(KnowledgeRetriever):
    """
    相似度检索器 - 基于向量相似度检索知识
    """
    
    def __init__(self, 
                vector_store: VectorStore,
                embedding_manager: EmbeddingManager,
                config: Optional[RetrievalConfig] = None):
        """
        初始化相似度检索器
        
        Args:
            vector_store: 向量存储
            embedding_manager: 嵌入管理器
            config: 检索配置，如果为None则使用默认配置
        """
        super().__init__(config)
        
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def initialize(self) -> None:
        """初始化检索器"""
        self.vector_store.ensure_initialized()
        self.embedding_manager.ensure_initialized()
        
        self._initialized = True
        logger.info(f"相似度检索器初始化完成，策略: {self.config.strategy.name}")
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        根据查询检索知识
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果列表
        """
        self.ensure_initialized()
        
        # 生成查询向量
        query_vector = self.embedding_manager.embed_query(query)
        
        # 指定过滤条件
        filter_dict = self.config.metadata_filters if self.config.metadata_filters else None
        
        # 搜索相似向量
        search_results = self.vector_store.search(
            query_vector=query_vector,
            k=self.config.top_k * 2,  # 获取更多结果，以便后续处理
            filter=filter_dict
        )
        
        # 将搜索结果转换为检索结果
        results = [
            RetrievalResult.from_search_result(result, source="similarity")
            for result in search_results
        ]
        
        # 处理结果
        processed_results = self.process_results(results)
        
        return processed_results
    
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
        self.ensure_initialized()
        
        # 合并过滤条件
        combined_filter = {}
        if self.config.metadata_filters:
            combined_filter.update(self.config.metadata_filters)
        combined_filter.update(filter)
        
        # 生成查询向量
        query_vector = self.embedding_manager.embed_query(query)
        
        # 搜索相似向量
        search_results = self.vector_store.search(
            query_vector=query_vector,
            k=self.config.top_k * 2,  # 获取更多结果，以便后续处理
            filter=combined_filter
        )
        
        # 将搜索结果转换为检索结果
        results = [
            RetrievalResult.from_search_result(result, source="similarity_filtered")
            for result in search_results
        ]
        
        # 处理结果
        processed_results = self.process_results(results)
        
        return processed_results
    
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
        k = k or self.config.top_k
        
        # 获取检索结果
        results = self.retrieve(query)
        
        # 提取文档文本
        return [result.text for result in results[:k]]
    
    def get_retriever_type(self) -> str:
        """
        获取检索器类型
        
        Returns:
            检索器类型
        """
        return "similarity_retriever"
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """
        获取检索器信息
        
        Returns:
            检索器信息字典
        """
        return {
            "type": self.get_retriever_type(),
            "strategy": self.config.strategy.name,
            "vector_store": self.vector_store.get_store_info(),
            "embedding_manager": self.embedding_manager.get_manager_name(),
            "top_k": self.config.top_k,
            "similarity_threshold": self.config.similarity_threshold,
            "deduplicate": self.config.deduplicate,
            "metadata_filters": self.config.metadata_filters
        }


class KeywordRetriever(KnowledgeRetriever):
    """
    关键词检索器 - 基于关键词匹配检索知识
    """
    
    def __init__(self, 
                vector_store: VectorStore,
                config: Optional[RetrievalConfig] = None):
        """
        初始化关键词检索器
        
        Args:
            vector_store: 向量存储
            config: 检索配置，如果为None则使用默认配置
        """
        if not config:
            config = RetrievalConfig(strategy=RetrievalStrategy.KEYWORD)
        else:
            config.strategy = RetrievalStrategy.KEYWORD
            
        super().__init__(config)
        
        self.vector_store = vector_store
        self.documents = {}  # id -> document content
        self.index = defaultdict(set)  # word -> set of document ids
    
    def initialize(self) -> None:
        """初始化检索器"""
        self.vector_store.ensure_initialized()
        
        # 构建关键词索引
        self._build_index()
        
        self._initialized = True
        logger.info(f"关键词检索器初始化完成，索引包含 {len(self.documents)} 个文档")
    
    def _build_index(self) -> None:
        """构建关键词索引"""
        # 清空索引
        self.documents = {}
        self.index = defaultdict(set)
        
        # 检查向量存储中的记录数
        count = self.vector_store.count()
        
        if count == 0:
            logger.warning("向量存储为空，无法构建关键词索引")
            return
            
        # 获取所有记录
        # 注意：这不是一个高效的实现，实际使用时应该使用批量处理
        # 但目前向量存储接口没有提供获取所有记录的方法
        # 这里仅用于演示
        
        logger.info(f"从向量存储加载 {count} 个文档")
        
        # 模拟获取所有记录
        # 在实际实现中，应该有更高效的方式，比如向量存储提供批量获取的接口
        for i in range(count):
            try:
                # 这不是真正的实现，只是演示
                record_id = f"record_{i}"
                record = self.vector_store.get(record_id)
                
                if record and record.text:
                    # 存储文档
                    self.documents[record_id] = record.text
                    
                    # 为文档构建索引
                    words = self._tokenize(record.text)
                    for word in words:
                        self.index[word].add(record_id)
                        
            except Exception as e:
                logger.warning(f"获取记录失败: {str(e)}")
        
        logger.info(f"关键词索引已构建，包含 {len(self.documents)} 个文档, {len(self.index)} 个关键词")
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        根据查询检索知识
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果列表
        """
        self.ensure_initialized()
        
        # 分词
        query_words = self._tokenize(query)
        
        # 找到包含查询词的文档
        doc_scores = Counter()
        
        for word in query_words:
            if word in self.index:
                for doc_id in self.index[word]:
                    doc_scores[doc_id] += 1
        
        # 计算TF-IDF得分
        results = []
        for doc_id, raw_score in doc_scores.most_common(self.config.top_k * 2):
            if doc_id in self.documents:
                # 归一化得分 (0-1)
                score = raw_score / max(len(query_words), 1)
                
                # 创建检索结果
                result = RetrievalResult(
                    text=self.documents[doc_id],
                    score=score,
                    id=doc_id,
                    metadata={},  # 没有元数据
                    source="keyword"
                )
                
                results.append(result)
        
        # 处理结果
        processed_results = self.process_results(results)
        
        return processed_results
    
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
        # 关键词检索器不支持元数据过滤，直接使用普通检索
        logger.warning("关键词检索器不支持元数据过滤，执行普通检索")
        return self.retrieve(query)
    
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
        k = k or self.config.top_k
        
        # 获取检索结果
        results = self.retrieve(query)
        
        # 提取文档文本
        return [result.text for result in results[:k]]
    
    def get_retriever_type(self) -> str:
        """
        获取检索器类型
        
        Returns:
            检索器类型
        """
        return "keyword_retriever"
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """
        获取检索器信息
        
        Returns:
            检索器信息字典
        """
        return {
            "type": self.get_retriever_type(),
            "strategy": self.config.strategy.name,
            "document_count": len(self.documents),
            "keyword_count": len(self.index),
            "top_k": self.config.top_k
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """
        将文本分词
        
        Args:
            text: 文本
            
        Returns:
            单词列表
        """
        # 简单的分词实现，仅用于演示
        # 在实际实现中，应该使用更强大的分词工具
        
        # 转为小写
        text = text.lower()
        
        # 移除标点符号
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 分词
        words = text.split()
        
        # 移除停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of'}
        words = [w for w in words if w not in stop_words and len(w) > 1]
        
        return words


class HybridRetriever(KnowledgeRetriever):
    """
    混合检索器 - 结合向量相似度和关键词检索
    """
    
    def __init__(self, 
                vector_store: VectorStore,
                embedding_manager: EmbeddingManager,
                config: Optional[RetrievalConfig] = None):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储
            embedding_manager: 嵌入管理器
            config: 检索配置，如果为None则使用默认配置
        """
        if not config:
            config = RetrievalConfig(strategy=RetrievalStrategy.HYBRID)
        else:
            config.strategy = RetrievalStrategy.HYBRID
            
        super().__init__(config)
        
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        
        # 创建子检索器
        self.similarity_retriever = SimilarityRetriever(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            config=config
        )
        
        self.keyword_retriever = KeywordRetriever(
            vector_store=vector_store,
            config=config
        )
    
    def initialize(self) -> None:
        """初始化检索器"""
        self.similarity_retriever.initialize()
        self.keyword_retriever.initialize()
        
        self._initialized = True
        logger.info(
            f"混合检索器初始化完成，向量权重: {self.config.hybrid_weight}, "
            f"关键词权重: {1 - self.config.hybrid_weight}"
        )
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        根据查询检索知识
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果列表
        """
        self.ensure_initialized()
        
        # 获取向量相似度检索结果
        similarity_results = self.similarity_retriever.retrieve(query)
        
        # 获取关键词检索结果
        keyword_results = self.keyword_retriever.retrieve(query)
        
        # 合并结果
        combined_results = self._combine_results(
            similarity_results,
            keyword_results,
            self.config.hybrid_weight
        )
        
        # 处理结果
        processed_results = self.process_results(combined_results)
        
        return processed_results
    
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
        self.ensure_initialized()
        
        # 获取向量相似度检索结果（带过滤）
        similarity_results = self.similarity_retriever.retrieve_with_filter(query, filter)
        
        # 获取关键词检索结果（不支持过滤）
        keyword_results = self.keyword_retriever.retrieve(query)
        
        # 合并结果
        combined_results = self._combine_results(
            similarity_results,
            keyword_results,
            self.config.hybrid_weight
        )
        
        # 处理结果
        processed_results = self.process_results(combined_results)
        
        return processed_results
    
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
        k = k or self.config.top_k
        
        # 获取检索结果
        results = self.retrieve(query)
        
        # 提取文档文本
        return [result.text for result in results[:k]]
    
    def get_retriever_type(self) -> str:
        """
        获取检索器类型
        
        Returns:
            检索器类型
        """
        return "hybrid_retriever"
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """
        获取检索器信息
        
        Returns:
            检索器信息字典
        """
        return {
            "type": self.get_retriever_type(),
            "strategy": self.config.strategy.name,
            "vector_weight": self.config.hybrid_weight,
            "keyword_weight": 1 - self.config.hybrid_weight,
            "similarity_retriever": self.similarity_retriever.get_retriever_info(),
            "keyword_retriever": self.keyword_retriever.get_retriever_info(),
            "top_k": self.config.top_k
        }
    
    def _combine_results(self, 
                        similarity_results: List[RetrievalResult],
                        keyword_results: List[RetrievalResult],
                        weight: float) -> List[RetrievalResult]:
        """
        合并检索结果
        
        Args:
            similarity_results: 相似度检索结果
            keyword_results: 关键词检索结果
            weight: 相似度检索的权重 (0-1)
            
        Returns:
            合并后的结果列表
        """
        # 创建ID到结果的映射
        id_to_result = {}
        
        # 处理相似度结果
        for result in similarity_results:
            id_to_result[result.id] = result
            result.score *= weight
        
        # 处理关键词结果
        keyword_weight = 1 - weight
        for result in keyword_results:
            if result.id in id_to_result:
                # 如果已经存在，更新分数
                id_to_result[result.id].score += result.score * keyword_weight
            else:
                # 如果不存在，添加新结果
                result.score *= keyword_weight
                id_to_result[result.id] = result
        
        # 排序结果
        combined = list(id_to_result.values())
        combined.sort(key=lambda x: x.score, reverse=True)
        
        return combined


class EnsembleRetriever(KnowledgeRetriever):
    """
    集成检索器 - 集成多个检索器的结果
    """
    
    def __init__(self, 
                retrievers: List[KnowledgeRetriever],
                weights: Optional[List[float]] = None,
                config: Optional[RetrievalConfig] = None):
        """
        初始化集成检索器
        
        Args:
            retrievers: 检索器列表
            weights: 检索器权重列表（和为1），如果为None则平均分配
            config: 检索配置，如果为None则使用默认配置
        """
        if not config:
            config = RetrievalConfig(strategy=RetrievalStrategy.ENSEMBLE)
        else:
            config.strategy = RetrievalStrategy.ENSEMBLE
            
        super().__init__(config)
        
        self.retrievers = retrievers
        
        # 验证权重
        if weights:
            if len(weights) != len(retrievers):
                raise ValueError(f"权重数量 ({len(weights)}) 与检索器数量 ({len(retrievers)}) 不匹配")
                
            if abs(sum(weights) - 1.0) > 0.001:
                logger.warning(f"权重和 ({sum(weights)}) 不为1，将进行归一化")
                weights = [w / sum(weights) for w in weights]
                
            self.weights = weights
        else:
            # 平均分配权重
            self.weights = [1.0 / len(retrievers)] * len(retrievers)
    
    def initialize(self) -> None:
        """初始化检索器"""
        for retriever in self.retrievers:
            retriever.initialize()
            
        self._initialized = True
        logger.info(f"集成检索器初始化完成，包含 {len(self.retrievers)} 个检索器")
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        根据查询检索知识
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果列表
        """
        self.ensure_initialized()
        
        all_results = []
        
        # 从每个检索器获取结果
        for i, retriever in enumerate(self.retrievers):
            weight = self.weights[i]
            
            # 获取检索结果
            results = retriever.retrieve(query)
            
            # 应用权重
            for result in results:
                result.score *= weight
                
            all_results.extend(results)
        
        # 合并结果
        combined_results = self._combine_results(all_results)
        
        # 处理结果
        processed_results = self.process_results(combined_results)
        
        return processed_results
    
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
        self.ensure_initialized()
        
        all_results = []
        
        # 从每个检索器获取结果
        for i, retriever in enumerate(self.retrievers):
            weight = self.weights[i]
            
            # 获取检索结果
            try:
                results = retriever.retrieve_with_filter(query, filter)
            except Exception as e:
                logger.warning(f"检索器 {i} 过滤检索失败: {str(e)}，使用普通检索")
                results = retriever.retrieve(query)
            
            # 应用权重
            for result in results:
                result.score *= weight
                
            all_results.extend(results)
        
        # 合并结果
        combined_results = self._combine_results(all_results)
        
        # 处理结果
        processed_results = self.process_results(combined_results)
        
        return processed_results
    
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
        k = k or self.config.top_k
        
        # 获取检索结果
        results = self.retrieve(query)
        
        # 提取文档文本
        return [result.text for result in results[:k]]
    
    def get_retriever_type(self) -> str:
        """
        获取检索器类型
        
        Returns:
            检索器类型
        """
        return "ensemble_retriever"
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """
        获取检索器信息
        
        Returns:
            检索器信息字典
        """
        retriever_infos = [
            {
                "type": retriever.get_retriever_type(),
                "weight": weight
            }
            for retriever, weight in zip(self.retrievers, self.weights)
        ]
        
        return {
            "type": self.get_retriever_type(),
            "strategy": self.config.strategy.name,
            "retriever_count": len(self.retrievers),
            "retrievers": retriever_infos,
            "top_k": self.config.top_k
        }
    
    def _combine_results(self, all_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        合并检索结果
        
        Args:
            all_results: 所有检索结果
            
        Returns:
            合并后的结果列表
        """
        # 创建ID到结果的映射
        id_to_result = {}
        
        # 处理所有结果
        for result in all_results:
            if result.id in id_to_result:
                # 如果已经存在，取最高分数
                if result.score > id_to_result[result.id].score:
                    id_to_result[result.id] = result
            else:
                # 如果不存在，添加新结果
                id_to_result[result.id] = result
        
        # 排序结果
        combined = list(id_to_result.values())
        combined.sort(key=lambda x: x.score, reverse=True)
        
        return combined


class ReRankingRetriever(KnowledgeRetriever):
    """
    重排序检索器 - 先检索后重排序
    """
    
    def __init__(self, 
                base_retriever: KnowledgeRetriever,
                reranker: Optional[Callable[[str, List[str]], List[float]]] = None,
                config: Optional[RetrievalConfig] = None):
        """
        初始化重排序检索器
        
        Args:
            base_retriever: 基础检索器
            reranker: 重排序函数，接收查询和文档列表，返回分数列表
            config: 检索配置，如果为None则使用默认配置
        """
        if not config:
            config = RetrievalConfig(strategy=RetrievalStrategy.RERANKING)
        else:
            config.strategy = RetrievalStrategy.RERANKING
            
        super().__init__(config)
        
        self.base_retriever = base_retriever
        self.reranker = reranker
    
    def initialize(self) -> None:
        """初始化检索器"""
        self.base_retriever.initialize()
        
        self._initialized = True
        logger.info(f"重排序检索器初始化完成，基础检索器: {self.base_retriever.get_retriever_type()}")
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        根据查询检索知识
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果列表
        """
        self.ensure_initialized()
        
        # 获取基础检索结果
        base_results = self.base_retriever.retrieve(query)
        
        # 如果没有设置重排序函数，直接返回基础结果
        if not self.reranker:
            return self.process_results(base_results)
            
        # 进行重排序
        docs = [result.text for result in base_results]
        reranked_scores = self.reranker(query, docs)
        
        # 创建重排序结果
        reranked_results = []
        for i, (result, score) in enumerate(zip(base_results, reranked_scores)):
            # 创建新的结果对象
            reranked_result = RetrievalResult(
                text=result.text,
                score=score,
                id=result.id,
                metadata=result.metadata,
                raw_result=result.raw_result,
                source="reranked"
            )
            reranked_results.append(reranked_result)
            
        # 按新分数排序
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # 处理结果
        processed_results = self.process_results(reranked_results)
        
        return processed_results
    
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
        self.ensure_initialized()
        
        # 获取基础检索结果
        base_results = self.base_retriever.retrieve_with_filter(query, filter)
        
        # 如果没有设置重排序函数，直接返回基础结果
        if not self.reranker:
            return self.process_results(base_results)
            
        # 进行重排序
        docs = [result.text for result in base_results]
        reranked_scores = self.reranker(query, docs)
        
        # 创建重排序结果
        reranked_results = []
        for i, (result, score) in enumerate(zip(base_results, reranked_scores)):
            # 创建新的结果对象
            reranked_result = RetrievalResult(
                text=result.text,
                score=score,
                id=result.id,
                metadata=result.metadata,
                raw_result=result.raw_result,
                source="reranked_filtered"
            )
            reranked_results.append(reranked_result)
            
        # 按新分数排序
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # 处理结果
        processed_results = self.process_results(reranked_results)
        
        return processed_results
    
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
        k = k or self.config.top_k
        
        # 获取检索结果
        results = self.retrieve(query)
        
        # 提取文档文本
        return [result.text for result in results[:k]]
    
    def get_retriever_type(self) -> str:
        """
        获取检索器类型
        
        Returns:
            检索器类型
        """
        return "reranking_retriever"
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """
        获取检索器信息
        
        Returns:
            检索器信息字典
        """
        return {
            "type": self.get_retriever_type(),
            "strategy": self.config.strategy.name,
            "base_retriever": self.base_retriever.get_retriever_info(),
            "has_reranker": self.reranker is not None,
            "top_k": self.config.top_k
        }
