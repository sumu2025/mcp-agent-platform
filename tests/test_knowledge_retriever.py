"""
知识检索系统测试
"""

import sys
import os
from pathlib import Path
import unittest
import numpy as np

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入所需组件
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
    SimilarityRetriever,
    HybridRetriever,
    KeywordRetriever,
    RetrievalStrategy,
    RetrievalResult
)


class MockEmbeddingManager:
    """模拟嵌入管理器"""
    
    def __init__(self):
        self.dim = 384
        self._initialized = True
    
    def initialize(self):
        pass
    
    def ensure_initialized(self):
        pass
    
    def embed_query(self, query):
        # 根据查询内容返回不同的向量，使得不同查询匹配不同文档
        if "深度学习" in query:
            vec = np.zeros(self.dim)
            vec[0:self.dim//3] = 1  # 偏向第一组向量
        elif "机器学习" in query:
            vec = np.zeros(self.dim)
            vec[self.dim//3:2*self.dim//3] = 1  # 偏向第二组向量
        elif "人工智能" in query:
            vec = np.zeros(self.dim)
            vec[2*self.dim//3:] = 1  # 偏向第三组向量
        else:
            vec = np.random.random(self.dim)  # 随机向量
            
        return vec / np.linalg.norm(vec)  # 归一化
    
    def get_manager_name(self):
        return "mock_embedding_manager"


class TestKnowledgeRetriever(unittest.TestCase):
    """测试知识检索器功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试数据
        self.test_texts = [
            "深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。",
            "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统。",
            "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。",
            "神经网络是深度学习的核心组件，它模拟人脑的结构和功能。",
            "监督学习是机器学习的一种方法，它使用标记数据进行训练。",
            "强化学习是机器学习的一种方法，它通过奖励机制学习最佳行动。"
        ]
        
        self.test_metadata = [
            {"category": "AI", "subcategory": "deep_learning", "level": "intermediate"},
            {"category": "AI", "subcategory": "machine_learning", "level": "introductory"},
            {"category": "AI", "level": "introductory"},
            {"category": "AI", "subcategory": "deep_learning", "level": "intermediate"},
            {"category": "AI", "subcategory": "machine_learning", "level": "intermediate"},
            {"category": "AI", "subcategory": "machine_learning", "level": "advanced"}
        ]
        
        # 创建向量存储
        self.vector_store = InMemoryVectorStore(config=VectorStoreConfig(
            embedding_dim=384,
            distance_metric="cosine",
            normalize_vectors=True
        ))
        self.vector_store.initialize()
        
        # 添加测试数据
        dim = 384
        for i, (text, metadata) in enumerate(zip(self.test_texts, self.test_metadata)):
            # 创建向量，使不同类别的文档有不同的向量模式
            vec = np.zeros(dim)
            
            if "deep_learning" in metadata.get("subcategory", ""):
                vec[0:dim//3] = 1  # 第一组向量
            elif "machine_learning" in metadata.get("subcategory", ""):
                vec[dim//3:2*dim//3] = 1  # 第二组向量
            else:
                vec[2*dim//3:] = 1  # 第三组向量
                
            # 添加随机噪声
            vec += np.random.random(dim) * 0.1
            
            # 归一化向量
            vec = vec / np.linalg.norm(vec)
            
            # 创建记录
            record = VectorRecord(
                id=f"test_{i}",
                embedding=vec,
                text=text,
                metadata=metadata
            )
            
            # 添加到向量存储
            self.vector_store.add(record)
            
        # 创建模拟嵌入管理器
        self.mock_embedding_manager = MockEmbeddingManager()
        
        # 尝试创建真实嵌入管理器（如果可能）
        try:
            self.real_embedding_manager = LocalEmbedding(config=EmbeddingConfig(
                model_name="paraphrase-MiniLM-L6-v2",
                embedding_dim=384
            ))
            self.real_embedding_manager.initialize()
        except Exception as e:
            print(f"真实嵌入管理器初始化失败: {str(e)}，将仅使用模拟管理器")
            self.real_embedding_manager = None
    
    def test_similarity_retriever(self):
        """测试相似度检索器"""
        # 创建检索器
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.mock_embedding_manager,
            config=RetrievalConfig(top_k=2)
        )
        retriever.initialize()
        
        # 查询
        query = "深度学习是什么"
        results = retriever.retrieve(query)
        
        # 验证结果
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RetrievalResult)
        
        # 验证结果内容（应该与深度学习相关）
        found_deep_learning = False
        for result in results:
            if "深度学习" in result.text:
                found_deep_learning = True
                break
                
        self.assertTrue(found_deep_learning)
    
    def test_retriever_with_filter(self):
        """测试带过滤的检索器"""
        # 创建检索器
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.mock_embedding_manager,
            config=RetrievalConfig(top_k=3)
        )
        retriever.initialize()
        
        # 带过滤条件的查询
        query = "机器学习方法"
        filter_dict = {"level": "intermediate"}
        results = retriever.retrieve_with_filter(query, filter_dict)
        
        # 验证结果
        for result in results:
            self.assertEqual(result.metadata.get("level"), "intermediate")
    
    def test_get_relevant_documents(self):
        """测试获取相关文档"""
        # 创建检索器
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.mock_embedding_manager,
            config=RetrievalConfig(top_k=2)
        )
        retriever.initialize()
        
        # 获取相关文档
        query = "人工智能应用"
        docs = retriever.get_relevant_documents(query)
        
        # 验证结果
        self.assertEqual(len(docs), 2)
        self.assertIsInstance(docs[0], str)
    
    def test_hybrid_retriever(self):
        """测试混合检索器"""
        # 跳过测试如果没有真实嵌入管理器
        if self.real_embedding_manager is None:
            self.skipTest("真实嵌入管理器不可用，跳过混合检索器测试")
            
        # 创建混合检索器
        retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.real_embedding_manager,
            config=RetrievalConfig(
                strategy=RetrievalStrategy.HYBRID,
                top_k=3,
                hybrid_weight=0.7
            )
        )
        retriever.initialize()
        
        # 查询
        query = "神经网络在深度学习中的应用"
        results = retriever.retrieve(query)
        
        # 验证结果
        self.assertGreaterEqual(len(results), 1)
        
        # 结果应该包含相关关键词
        found_keyword = False
        keywords = ["神经网络", "深度学习"]
        for result in results:
            for keyword in keywords:
                if keyword in result.text:
                    found_keyword = True
                    break
        
        self.assertTrue(found_keyword)
    
    def test_deduplicate_results(self):
        """测试结果去重功能"""
        # 创建两个相似的结果
        result1 = RetrievalResult(
            text="这是一个关于人工智能的文本",
            score=0.9,
            id="1",
            metadata={}
        )
        
        result2 = RetrievalResult(
            text="这是一个关于人工智能技术的文本",
            score=0.8,
            id="2",
            metadata={}
        )
        
        # 创建一个不同的结果
        result3 = RetrievalResult(
            text="这是一个关于机器学习的完全不同的文本",
            score=0.7,
            id="3",
            metadata={}
        )
        
        # 创建检索器
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.mock_embedding_manager,
            config=RetrievalConfig(
                top_k=3,
                deduplicate=True,
                dedupe_threshold=0.5  # 设置较低的阈值以便测试
            )
        )
        retriever.initialize()
        
        # 去重
        original = [result1, result2, result3]
        deduplicated = retriever.deduplicate_results(original)
        
        # 验证结果
        self.assertLess(len(deduplicated), len(original))


if __name__ == "__main__":
    unittest.main()
