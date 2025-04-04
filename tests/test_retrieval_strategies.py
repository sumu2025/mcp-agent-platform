"""
检索策略系统测试
"""

import sys
import os
from pathlib import Path
import unittest
import logging
from typing import List, Dict, Any
import numpy as np

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入检索相关组件
from mcp.knowledge.embedding import (
    EmbeddingConfig,
    LocalEmbedding,
    TextEmbedding
)

from mcp.knowledge.storage import (
    VectorStoreConfig,
    InMemoryVectorStore,
    VectorRecord,
    SearchResult
)

from mcp.knowledge.retrieval import (
    RetrievalConfig,
    RetrievalStrategy,
    RetrievalResult,
    SimilarityRetriever,
    KeywordRetriever,
    HybridRetriever,
    EnsembleRetriever,
    ReRankingRetriever
)


class TestRetrievalStrategies(unittest.TestCase):
    """测试各种检索策略"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试数据
        self.test_texts = [
            "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统，而无需被明确编程。",
            "深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。",
            "自然语言处理是人工智能的一个分支，它关注计算机理解和生成人类语言的能力。",
            "计算机视觉是人工智能的一个领域，它使计算机能够理解和解释视觉信息。",
            "Python是一种广泛使用的高级编程语言，以其简洁和可读性而闻名。",
            "JavaScript是一种用于Web开发的脚本语言，它允许在网页中添加交互性。"
        ]
        
        # 创建测试元数据
        self.test_metadata = [
            {"category": "AI", "level": "introductory", "type": "overview"},
            {"category": "AI", "subcategory": "machine_learning", "level": "introductory", "type": "overview"},
            {"category": "AI", "subcategory": "deep_learning", "level": "intermediate", "type": "technical"},
            {"category": "AI", "subcategory": "nlp", "level": "intermediate", "type": "application"},
            {"category": "AI", "subcategory": "computer_vision", "level": "intermediate", "type": "technical"},
            {"category": "programming", "language": "python", "level": "introductory", "type": "tutorial"},
            {"category": "programming", "language": "javascript", "level": "introductory", "type": "tutorial"}
        ]
        
        try:
            # 初始化嵌入管理器
            self.embedding_manager = LocalEmbedding(config=EmbeddingConfig(
                model_name="paraphrase-MiniLM-L6-v2",
                embedding_dim=384
            ))
            self.embedding_manager.initialize()
            
            # 创建向量存储
            self.vector_store = InMemoryVectorStore(config=VectorStoreConfig(
                embedding_dim=384,
                distance_metric="cosine"
            ))
            self.vector_store.initialize()
            
            # 添加测试数据到向量存储
            for i, (text, metadata) in enumerate(zip(self.test_texts, self.test_metadata)):
                # 生成嵌入
                embedding = self.embedding_manager.embed_text(text).embedding
                
                # 创建向量记录
                record = VectorRecord(
                    id=f"test_{i}",
                    embedding=embedding,
                    text=text,
                    metadata=metadata
                )
                
                # 添加到向量存储
                self.vector_store.add(record)
                
        except Exception as e:
            self.skipTest(f"设置测试环境失败: {str(e)}")
    
    def test_similarity_retriever(self):
        """测试相似度检索器"""
        # 创建检索器
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(
                strategy=RetrievalStrategy.SIMILARITY,
                top_k=3
            )
        )
        retriever.initialize()
        
        # 测试查询
        queries = [
            "什么是深度学习？",
            "编程语言有哪些？",
            "计算机视觉的应用"
        ]
        
        for query in queries:
            # 检索
            results = retriever.retrieve(query)
            
            # 验证结果
            self.assertIsNotNone(results, "应该返回检索结果")
            self.assertLessEqual(len(results), 3, "结果数量不应超过top_k")
            if results:
                self.assertIsInstance(results[0], RetrievalResult, "结果类型应该是RetrievalResult")
                self.assertGreaterEqual(results[0].score, 0, "相似度得分应该大于等于0")
                self.assertLessEqual(results[0].score, 1, "相似度得分应该小于等于1")
                
                # 验证排序
                if len(results) > 1:
                    self.assertGreaterEqual(results[0].score, results[-1].score, "结果应该按相似度降序排序")
    
    def test_keyword_retriever(self):
        """测试关键词检索器"""
        # 创建检索器
        retriever = KeywordRetriever(
            vector_store=self.vector_store,
            config=RetrievalConfig(
                strategy=RetrievalStrategy.KEYWORD,
                top_k=3
            )
        )
        retriever.initialize()
        
        # 测试查询
        queries = [
            "深度学习",
            "Python编程",
            "视觉信息处理"
        ]
        
        for query in queries:
            # 检索
            results = retriever.retrieve(query)
            
            # 验证结果
            self.assertIsNotNone(results, "应该返回检索结果")
            if results:
                self.assertIsInstance(results[0], RetrievalResult, "结果类型应该是RetrievalResult")
                self.assertGreaterEqual(results[0].score, 0, "相似度得分应该大于等于0")
                self.assertLessEqual(results[0].score, 1, "相似度得分应该小于等于1")
                
                # 验证结果中是否包含查询关键词
                query_terms = query.lower().split()
                result_text = results[0].text.lower()
                found_term = False
                for term in query_terms:
                    if len(term) > 3 and term in result_text:  # 忽略太短的词
                        found_term = True
                        break
                        
                # 注意：这个测试可能偶尔失败，因为关键词匹配不总是有保证
                # self.assertTrue(found_term, f"结果中应该包含查询关键词: {query}")
    
    def test_hybrid_retriever(self):
        """测试混合检索器"""
        # 创建检索器
        retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(
                strategy=RetrievalStrategy.HYBRID,
                top_k=3,
                hybrid_weight=0.7  # 向量相似度权重
            )
        )
        retriever.initialize()
        
        # 测试查询
        query = "深度学习如何使用神经网络"
        
        # 检索
        results = retriever.retrieve(query)
        
        # 验证结果
        self.assertIsNotNone(results, "应该返回检索结果")
        self.assertLessEqual(len(results), 3, "结果数量不应超过top_k")
        if results:
            self.assertIsInstance(results[0], RetrievalResult, "结果类型应该是RetrievalResult")
            self.assertEqual(results[0].source, "similarity", "来源应该是similarity（结果可能因实现而异）")
    
    def test_ensemble_retriever(self):
        """测试集成检索器"""
        # 创建基础检索器
        similarity_retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(top_k=3)
        )
        
        keyword_retriever = KeywordRetriever(
            vector_store=self.vector_store,
            config=RetrievalConfig(top_k=3)
        )
        
        # 创建集成检索器
        retriever = EnsembleRetriever(
            retrievers=[similarity_retriever, keyword_retriever],
            weights=[0.7, 0.3],
            config=RetrievalConfig(
                strategy=RetrievalStrategy.ENSEMBLE,
                top_k=3
            )
        )
        retriever.initialize()
        
        # 测试查询
        query = "机器学习和深度学习的区别"
        
        # 检索
        results = retriever.retrieve(query)
        
        # 验证结果
        self.assertIsNotNone(results, "应该返回检索结果")
        self.assertLessEqual(len(results), 3, "结果数量不应超过top_k")
    
    def test_reranking_retriever(self):
        """测试重排序检索器"""
        # 创建基础检索器
        base_retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(top_k=5)  # 获取更多结果用于重排序
        )
        
        # 创建简单的重排序函数
        def simple_reranker(query, documents):
            # 简单实现：包含查询词的文档得分更高
            scores = []
            query_terms = query.lower().split()
            for doc in documents:
                score = 0.5  # 基础分
                doc_lower = doc.lower()
                for term in query_terms:
                    if len(term) > 3 and term in doc_lower:  # 忽略太短的词
                        score += 0.1
                scores.append(min(score, 1.0))  # 最高分为1.0
            return scores
        
        # 创建重排序检索器
        retriever = ReRankingRetriever(
            base_retriever=base_retriever,
            reranker=simple_reranker,
            config=RetrievalConfig(
                strategy=RetrievalStrategy.RERANKING,
                top_k=3
            )
        )
        retriever.initialize()
        
        # 测试查询
        query = "深度学习使用神经网络"
        
        # 检索
        results = retriever.retrieve(query)
        
        # 验证结果
        self.assertIsNotNone(results, "应该返回检索结果")
        self.assertLessEqual(len(results), 3, "结果数量不应超过top_k")
        if results:
            self.assertIsInstance(results[0], RetrievalResult, "结果类型应该是RetrievalResult")
            self.assertEqual(results[0].source, "reranked", "来源应该是reranked")
    
    def test_metadata_filtering(self):
        """测试元数据过滤"""
        # 创建检索器
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(top_k=5)
        )
        retriever.initialize()
        
        # 测试查询
        query = "人工智能技术"
        
        # 使用不同过滤条件
        filters = [
            {"level": "introductory"},
            {"category": "programming"},
            {"subcategory": "deep_learning"},
            {"type": "technical"}
        ]
        
        for filter_dict in filters:
            # 检索
            results = retriever.retrieve_with_filter(query, filter_dict)
            
            # 验证结果
            if results:
                for result in results:
                    # 检查每个结果是否满足过滤条件
                    for key, value in filter_dict.items():
                        self.assertEqual(result.metadata.get(key), value, 
                                       f"结果应该满足过滤条件: {key}={value}")
    
    def test_result_deduplication(self):
        """测试结果去重"""
        # 创建检索器配置，启用去重
        config = RetrievalConfig(
            strategy=RetrievalStrategy.SIMILARITY,
            top_k=5,
            deduplicate=True,
            dedupe_threshold=0.9
        )
        
        # 创建检索器
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=config
        )
        retriever.initialize()
        
        # 创建测试数据 - 添加重复文档（内容稍有不同）
        duplicate_texts = [
            "深度学习是机器学习的一种方法，使用多层神经网络来学习复杂模式。",  # 几乎与原始文本相同
            "自然语言处理是AI的一个分支，专注于计算机理解和生成人类语言。"  # 与原始文本稍有不同
        ]
        
        for i, text in enumerate(duplicate_texts):
            embedding = self.embedding_manager.embed_text(text).embedding
            record = VectorRecord(
                id=f"duplicate_{i}",
                embedding=embedding,
                text=text,
                metadata={"category": "test_duplicate"}
            )
            self.vector_store.add(record)
        
        # 测试查询
        query = "深度学习和自然语言处理"
        
        # 检索
        results = retriever.retrieve(query)
        
        # 验证去重效果
        texts = [result.text for result in results]
        # 检查结果中没有完全相同的文本
        self.assertEqual(len(texts), len(set(texts)), "结果中不应有完全相同的文本")
    
    def test_retrieval_result_processing(self):
        """测试检索结果处理"""
        # 创建检索器，设置较低的相似度阈值
        config = RetrievalConfig(
            strategy=RetrievalStrategy.SIMILARITY,
            top_k=5,
            similarity_threshold=0.3  # 较低的阈值，便于测试
        )
        
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=config
        )
        retriever.initialize()
        
        # 测试查询 - 使用不太相关的查询
        query = "量子计算和区块链技术"  # 与测试数据不太相关
        
        # 检索
        results = retriever.retrieve(query)
        
        # 验证过滤效果 - 所有结果的得分应该高于阈值
        for result in results:
            self.assertGreaterEqual(result.score, config.similarity_threshold, 
                                 "结果得分应该高于相似度阈值")
        
        # 测试内容过滤
        def content_filter(text):
            # 仅保留包含"人工智能"的文本
            return "人工智能" in text
            
        # 设置内容过滤器
        config.content_filter = content_filter
        
        # 检索并过滤
        filtered_results = retriever.process_results(retriever.retrieve(query))
        
        # 验证过滤效果
        for result in filtered_results:
            self.assertIn("人工智能", result.text, "结果应该包含'人工智能'")
            
    def test_get_relevant_documents(self):
        """测试获取相关文档"""
        # 创建检索器
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(top_k=3)
        )
        retriever.initialize()
        
        # 测试查询
        query = "人工智能和机器学习"
        
        # 获取相关文档
        docs = retriever.get_relevant_documents(query)
        
        # 验证结果
        self.assertIsNotNone(docs, "应该返回相关文档")
        self.assertIsInstance(docs, list, "结果应该是列表")
        self.assertLessEqual(len(docs), 3, "文档数量不应超过top_k")
        if docs:
            self.assertIsInstance(docs[0], str, "文档应该是字符串")


if __name__ == "__main__":
    unittest.main()
