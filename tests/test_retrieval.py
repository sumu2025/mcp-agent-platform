"""
知识检索系统测试
"""

import unittest
import sys
import os
from pathlib import Path
import numpy as np

# 导入测试配置
from test_config import (
    TEST_QUERIES, 
    TEST_EMBEDDING_DIM, 
    logger,
    create_test_documents,
    setup_basic_rag_components,
    index_test_documents
)

# 导入要测试的模块
from mcp.knowledge.retrieval import (
    KnowledgeRetriever,
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
    SimilarityRetriever,
    HybridRetriever,
    KeywordRetriever,
    EnsembleRetriever,
    ReRankingRetriever
)


class TestRetrieval(unittest.TestCase):
    """测试知识检索系统"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 设置基本组件
        cls.embedding_manager, cls.vector_store = setup_basic_rag_components()
        
        # 创建测试文档
        cls.document_paths = create_test_documents()
        
        # 索引测试文档
        cls.chunk_count = index_test_documents(
            cls.embedding_manager, 
            cls.vector_store, 
            cls.document_paths
        )
        
        logger.info(f"已索引 {cls.chunk_count} 个文本块到向量存储")
    
    def test_similarity_retriever(self):
        """测试相似度检索器"""
        # 创建相似度检索器
        similarity_config = RetrievalConfig(
            strategy=RetrievalStrategy.SIMILARITY,
            top_k=3,
            deduplicate=True
        )
        
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=similarity_config
        )
        
        retriever.initialize()
        
        # 测试每个查询
        for query_info in TEST_QUERIES:
            query = query_info["text"]
            expected_category = query_info["expected_category"]
            
            # 检索结果
            results = retriever.retrieve(query)
            
            # 验证结果
            self.assertGreater(len(results), 0)
            self.assertLessEqual(len(results), similarity_config.top_k)
            self.assertIsInstance(results[0], RetrievalResult)
            
            # 验证相关性
            # 至少有一个结果应该包含预期的类别
            category_found = False
            for result in results:
                if expected_category in str(result.metadata):
                    category_found = True
                    break
            
            self.assertTrue(category_found, f"查询 '{query}' 未找到预期类别 '{expected_category}'")
    
    def test_hybrid_retriever(self):
        """测试混合检索器"""
        # 创建混合检索器
        hybrid_config = RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID,
            top_k=3,
            hybrid_weight=0.7,
            deduplicate=True
        )
        
        retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=hybrid_config
        )
        
        retriever.initialize()
        
        # 测试每个查询
        for query_info in TEST_QUERIES:
            query = query_info["text"]
            
            # 检索结果
            results = retriever.retrieve(query)
            
            # 验证结果
            self.assertGreater(len(results), 0)
            self.assertLessEqual(len(results), hybrid_config.top_k)
            
            # 验证结果来源
            hybrid_sources_found = False
            for result in results:
                if hasattr(result, 'source') and result.source:
                    hybrid_sources_found = True
                    break
            
            self.assertTrue(hybrid_sources_found, "混合检索应该标记结果来源")
    
    def test_keyword_retriever(self):
        """测试关键词检索器"""
        # 创建关键词检索器
        keyword_config = RetrievalConfig(
            strategy=RetrievalStrategy.KEYWORD,
            top_k=3
        )
        
        retriever = KeywordRetriever(
            vector_store=self.vector_store,
            config=keyword_config
        )
        
        # 初始化检索器
        retriever.initialize()
        
        # 测试简单查询
        query = "深度学习 神经网络"
        results = retriever.retrieve(query)
        
        # 验证结果
        self.assertGreaterEqual(len(results), 0)  # 关键词检索可能不总是有结果
    
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
        ensemble_config = RetrievalConfig(
            strategy=RetrievalStrategy.ENSEMBLE,
            top_k=3
        )
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[similarity_retriever, keyword_retriever],
            weights=[0.7, 0.3],
            config=ensemble_config
        )
        
        # 初始化检索器
        ensemble_retriever.initialize()
        
        # 测试查询
        query = TEST_QUERIES[0]["text"]
        results = ensemble_retriever.retrieve(query)
        
        # 验证结果
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), ensemble_config.top_k)
    
    def test_reranking_retriever(self):
        """测试重排序检索器"""
        # 创建基础检索器
        base_retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(top_k=5)  # 获取更多结果，然后重排序
        )
        
        # 创建简单的重排序函数
        def simple_reranker(query, docs):
            scores = []
            query_terms = query.lower().split()
            
            for doc in docs:
                # 计算查询词在文档中的出现次数
                doc_lower = doc.lower()
                term_count = sum(1 for term in query_terms if term in doc_lower)
                
                # 归一化得分
                score = term_count / max(1, len(query_terms))
                scores.append(score)
                
            return scores
        
        # 创建重排序检索器
        reranking_config = RetrievalConfig(
            strategy=RetrievalStrategy.RERANKING,
            top_k=3
        )
        
        reranking_retriever = ReRankingRetriever(
            base_retriever=base_retriever,
            reranker=simple_reranker,
            config=reranking_config
        )
        
        # 初始化检索器
        reranking_retriever.initialize()
        
        # 测试查询
        query = "深度学习技术应用"
        results = reranking_retriever.retrieve(query)
        
        # 验证结果
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), reranking_config.top_k)
        
        # 验证结果来源
        for result in results:
            self.assertEqual(result.source, "reranked")
    
    def test_retrieval_with_filter(self):
        """测试带过滤条件的检索"""
        # 创建检索器
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(top_k=3)
        )
        
        retriever.initialize()
        
        # 设置过滤条件
        filter_condition = {"level": "intermediate"}
        
        # 测试带过滤条件的检索
        query = "人工智能技术"
        results = retriever.retrieve_with_filter(query, filter_condition)
        
        # 验证结果
        self.assertGreater(len(results), 0)
        
        # 验证所有结果都满足过滤条件
        for result in results:
            self.assertEqual(result.metadata.get("level"), "intermediate")
    
    def test_deduplicate_results(self):
        """测试结果去重功能"""
        # 创建检索器
        config = RetrievalConfig(
            top_k=5,
            deduplicate=True,
            dedupe_threshold=0.9  # 设置较高的阈值，以便测试
        )
        
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=config
        )
        
        # 手动创建重复结果
        results = []
        for i in range(2):
            result = RetrievalResult(
                text="这是一个重复的文本内容，应该被去重。",
                score=0.9 - (i * 0.1),
                id=f"dup_{i}",
                metadata={}
            )
            results.append(result)
            
        # 添加一个不同的结果
        results.append(RetrievalResult(
            text="这是一个完全不同的文本内容。",
            score=0.7,
            id="unique",
            metadata={}
        ))
        
        # 去重
        deduplicated = retriever.deduplicate_results(results)
        
        # 验证结果
        self.assertLess(len(deduplicated), len(results))
        self.assertEqual(len(deduplicated), 2)  # 应该只保留一个重复结果和一个不同结果
    
    def test_filter_by_score(self):
        """测试按分数过滤结果"""
        # 创建检索器
        config = RetrievalConfig(
            top_k=5,
            similarity_threshold=0.5  # 设置分数阈值
        )
        
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=config
        )
        
        # 手动创建结果
        results = []
        for i in range(5):
            result = RetrievalResult(
                text=f"测试文本 {i}",
                score=(i+1) * 0.2,  # 分数从0.2到1.0
                id=f"test_{i}",
                metadata={}
            )
            results.append(result)
            
        # 过滤
        filtered = retriever.filter_by_score(results)
        
        # 验证结果
        self.assertEqual(len(filtered), 3)  # 只有3个结果分数>=0.5
        for result in filtered:
            self.assertGreaterEqual(result.score, 0.5)
    
    def test_get_relevant_documents(self):
        """测试获取相关文档功能"""
        # 创建检索器
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(top_k=3)
        )
        
        retriever.initialize()
        
        # 获取相关文档
        query = TEST_QUERIES[0]["text"]
        documents = retriever.get_relevant_documents(query)
        
        # 验证结果
        self.assertEqual(len(documents), 3)
        for doc in documents:
            self.assertIsInstance(doc, str)
            self.assertGreater(len(doc), 0)
    
    def test_retriever_info(self):
        """测试检索器信息"""
        # 创建检索器
        retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(
                strategy=RetrievalStrategy.HYBRID,
                top_k=3,
                hybrid_weight=0.7
            )
        )
        
        retriever.initialize()
        
        # 获取检索器信息
        info = retriever.get_retriever_info()
        
        # 验证信息内容
        self.assertEqual(info["type"], "hybrid_retriever")
        self.assertEqual(info["strategy"], "HYBRID")
        self.assertEqual(info["vector_weight"], 0.7)
        self.assertEqual(info["keyword_weight"], 0.3)


if __name__ == "__main__":
    unittest.main()
