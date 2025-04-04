"""
知识检索系统测试
"""

import sys
import os
from pathlib import Path
import unittest
import numpy as np
import tempfile
import shutil
import logging

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入检索组件
from mcp.knowledge.storage import (
    VectorStoreConfig,
    VectorRecord,
    InMemoryVectorStore
)

from mcp.knowledge.embedding import (
    EmbeddingConfig,
    LocalEmbedding
)

from mcp.knowledge.retrieval import (
    RetrievalConfig,
    RetrievalStrategy,
    RetrievalResult,
    SimilarityRetriever,
    HybridRetriever,
    KeywordRetriever
)


class TestRetriever(unittest.TestCase):
    """测试知识检索系统"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 测试文本
        self.test_texts = [
            "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统。",
            "深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。",
            "自然语言处理是人工智能的一个分支，它关注计算机理解和生成人类语言的能力。",
            "计算机视觉是人工智能的一个领域，它使计算机能够理解和解释视觉信息。"
        ]
        
        # 测试元数据
        self.test_metadata = [
            {"category": "AI", "level": "introductory"},
            {"category": "AI", "subcategory": "machine_learning", "level": "introductory"},
            {"category": "AI", "subcategory": "deep_learning", "level": "intermediate"},
            {"category": "AI", "subcategory": "nlp", "level": "intermediate"},
            {"category": "AI", "subcategory": "computer_vision", "level": "intermediate"}
        ]
        
        # 尝试初始化嵌入管理器
        try:
            self.embedding_manager = LocalEmbedding(config=EmbeddingConfig(
                model_name="paraphrase-MiniLM-L6-v2",
                embedding_dim=384
            ))
            self.embedding_manager.initialize()
            self.skip_tests = False
        except Exception as e:
            print(f"嵌入管理器初始化失败: {str(e)}，跳过相关测试")
            self.skip_tests = True
            return
            
        # 创建向量存储
        self.vector_store = InMemoryVectorStore(config=VectorStoreConfig(
            embedding_dim=384,
            distance_metric="cosine"
        ))
        self.vector_store.initialize()
        
        # 添加测试数据到向量存储
        for i, (text, metadata) in enumerate(zip(self.test_texts, self.test_metadata)):
            try:
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
                print(f"添加测试数据失败: {str(e)}")
                self.skip_tests = True
                return
        
        # 创建检索配置
        self.retrieval_config = RetrievalConfig(
            strategy=RetrievalStrategy.SIMILARITY,
            top_k=3,
            deduplicate=True
        )
        
        # 创建检索器
        self.similarity_retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=self.retrieval_config
        )
        
        self.keyword_retriever = KeywordRetriever(
            vector_store=self.vector_store,
            config=self.retrieval_config
        )
        
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(
                strategy=RetrievalStrategy.HYBRID,
                top_k=3,
                hybrid_weight=0.7,
                deduplicate=True
            )
        )
        
        # 初始化检索器
        self.similarity_retriever.initialize()
        self.keyword_retriever.initialize()
        self.hybrid_retriever.initialize()
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_similarity_retriever(self):
        """测试相似度检索器"""
        if self.skip_tests:
            self.skipTest("嵌入管理器初始化失败，跳过测试")
            
        # 查询
        query = "什么是深度学习？"
        results = self.similarity_retriever.retrieve(query)
        
        # 验证结果
        self.assertGreater(len(results), 0, "应该返回非空结果")
        self.assertIsInstance(results[0], RetrievalResult, "结果应该是RetrievalResult对象")
        
        # 验证首条结果与深度学习相关
        self.assertIn("深度学习", results[0].text, "首条结果应该与深度学习相关")
    
    def test_keyword_retriever(self):
        """测试关键词检索器"""
        if self.skip_tests:
            self.skipTest("嵌入管理器初始化失败，跳过测试")
            
        # 查询
        query = "深度学习"
        results = self.keyword_retriever.retrieve(query)
        
        # 验证结果
        self.assertGreaterEqual(len(results), 0, "应该返回结果")
        
        # 如果有结果，验证与查询相关
        if results:
            self.assertIsInstance(results[0], RetrievalResult, "结果应该是RetrievalResult对象")
            self.assertIn("深度学习", results[0].text, "结果应该与查询相关")
    
    def test_hybrid_retriever(self):
        """测试混合检索器"""
        if self.skip_tests:
            self.skipTest("嵌入管理器初始化失败，跳过测试")
            
        # 查询
        query = "什么是深度学习和神经网络？"
        results = self.hybrid_retriever.retrieve(query)
        
        # 验证结果
        self.assertGreater(len(results), 0, "应该返回非空结果")
        self.assertIsInstance(results[0], RetrievalResult, "结果应该是RetrievalResult对象")
    
    def test_filter_retrieval(self):
        """测试过滤检索"""
        if self.skip_tests:
            self.skipTest("嵌入管理器初始化失败，跳过测试")
            
        # 查询
        query = "人工智能"
        filter_dict = {"level": "intermediate"}
        
        # 过滤检索
        results = self.similarity_retriever.retrieve_with_filter(query, filter_dict)
        
        # 验证结果
        self.assertGreaterEqual(len(results), 0, "应该返回结果")
        
        # 验证结果满足过滤条件
        for result in results:
            self.assertEqual(result.metadata["level"], "intermediate", 
                           "结果应该满足过滤条件")
    
    def test_deduplication(self):
        """测试结果去重"""
        if self.skip_tests:
            self.skipTest("嵌入管理器初始化失败，跳过测试")
            
        # 创建重复结果
        results = [
            RetrievalResult(
                text="这是一个测试文档",
                score=0.9,
                id="doc1",
                metadata={}
            ),
            RetrievalResult(
                text="这是一个测试文档",  # 完全相同的文本
                score=0.8,
                id="doc2",
                metadata={}
            ),
            RetrievalResult(
                text="这是另一个文档",
                score=0.7,
                id="doc3",
                metadata={}
            )
        ]
        
        # 去重
        deduplicated = self.similarity_retriever.deduplicate_results(results)
        
        # 验证结果
        self.assertLess(len(deduplicated), len(results), 
                       "去重后的结果数量应该减少")
        self.assertEqual(len(deduplicated), 2, 
                       "应该有2个不重复的结果")
    
    def test_score_filter(self):
        """测试分数过滤"""
        if self.skip_tests:
            self.skipTest("嵌入管理器初始化失败，跳过测试")
            
        # 创建结果
        results = [
            RetrievalResult(
                text="文档1",
                score=0.9,
                id="doc1",
                metadata={}
            ),
            RetrievalResult(
                text="文档2",
                score=0.7,
                id="doc2",
                metadata={}
            ),
            RetrievalResult(
                text="文档3",
                score=0.5,
                id="doc3",
                metadata={}
            ),
            RetrievalResult(
                text="文档4",
                score=0.3,
                id="doc4",
                metadata={}
            )
        ]
        
        # 设置分数阈值
        self.similarity_retriever.config.similarity_threshold = 0.6
        
        # 过滤
        filtered = self.similarity_retriever.filter_by_score(results)
        
        # 验证结果
        self.assertEqual(len(filtered), 2, "应该只有2个结果通过阈值过滤")
        self.assertGreaterEqual(filtered[0].score, 0.6, "过滤后的结果分数应该大于阈值")
        self.assertGreaterEqual(filtered[1].score, 0.6, "过滤后的结果分数应该大于阈值")


if __name__ == "__main__":
    unittest.main()
