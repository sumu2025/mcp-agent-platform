"""
RAG系统端到端测试脚本
测试完整的RAG流程，从文档处理到回答生成
"""

import sys
import os
import unittest
import logging
from pathlib import Path
from typing import List, Dict, Any
import glob

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入所需组件
from mcp.knowledge.indexing import (
    MarkdownProcessor,
    RecursiveTextChunker,
    ObsidianMetadataExtractor
)

from mcp.knowledge.embedding import (
    EmbeddingConfig,
    LocalEmbedding
)

from mcp.knowledge.storage import (
    VectorStoreConfig,
    InMemoryVectorStore,
    VectorRecord
)

from mcp.knowledge.retrieval import (
    RetrievalConfig,
    SimilarityRetriever,
    HybridRetriever,
    RetrievalStrategy
)

from mcp.knowledge.augmentation import (
    AugmentationConfig,
    AugmentationMode,
    StructuredAugmenter
)

from mcp.knowledge.rag_engine import (
    RAGEngine,
    RAGConfig,
    RAGStrategy
)


class MockLLMClient:
    """模拟LLM客户端，用于测试"""
    
    def __init__(self):
        """初始化模拟客户端"""
        self.last_prompt = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """模拟生成回答"""
        self.last_prompt = prompt
        
        # 根据提示内容生成简单回答
        if "深度学习" in prompt:
            return "深度学习是机器学习的一种方法，使用多层神经网络来学习复杂模式。"
        elif "机器学习" in prompt:
            return "机器学习是AI的一个子领域，关注从数据中学习的系统。"
        elif "自然语言处理" in prompt:
            return "自然语言处理应用包括机器翻译、情感分析、文本摘要和聊天机器人。"
        elif "计算机视觉" in prompt:
            return "计算机视觉使计算机能够理解视觉信息，应用于自动驾驶汽车等领域。"
        else:
            return "这是对您问题的模拟回答。"


class TestRAGEndToEnd(unittest.TestCase):
    """测试RAG端到端流程"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 设置日志级别
        logging.basicConfig(level=logging.INFO)
        
        # 测试数据目录
        cls.test_data_dir = Path(__file__).parent / "test_data"
        
        # 确保测试数据存在
        test_files = list(cls.test_data_dir.glob("*.md"))
        if len(test_files) < 3:
            cls.skipTest(cls, "测试数据不足，请先创建测试文档")
        
        try:
            # 初始化嵌入管理器
            cls.embedding_manager = LocalEmbedding(config=EmbeddingConfig(
                model_name="paraphrase-MiniLM-L6-v2",
                embedding_dim=384
            ))
            cls.embedding_manager.initialize()
            
            # 初始化向量存储
            cls.vector_store = InMemoryVectorStore(config=VectorStoreConfig(
                embedding_dim=384,
                distance_metric="cosine"
            ))
            cls.vector_store.initialize()
            
            # 初始化文档处理器
            cls.chunker = RecursiveTextChunker(
                chunk_size=512,
                chunk_overlap=100
            )
            
            cls.metadata_extractor = ObsidianMetadataExtractor()
            
            cls.processor = MarkdownProcessor(
                chunker=cls.chunker,
                metadata_extractor=cls.metadata_extractor
            )
            
            # 处理文档
            for file_path in test_files:
                # 处理文档
                chunks = cls.processor.process(file_path)
                
                for chunk in chunks:
                    # 生成嵌入
                    embedding = cls.embedding_manager.embed_text(chunk.text).embedding
                    
                    # 创建向量记录
                    record = VectorRecord(
                        id=f"{file_path.stem}_{chunks.index(chunk)}",
                        embedding=embedding,
                        text=chunk.text,
                        metadata=chunk.metadata
                    )
                    
                    # 添加到向量存储
                    cls.vector_store.add(record)
            
            logging.info(f"已添加 {cls.vector_store.count()} 个文档块到向量存储")
            
        except Exception as e:
            cls.skipTest(cls, f"测试环境设置失败: {str(e)}")
    
    def setUp(self):
        """设置每个测试用例"""
        # 创建LLM客户端
        self.llm_client = MockLLMClient()
    
    def test_basic_rag(self):
        """测试基本RAG流程"""
        # 创建RAG引擎
        rag_engine = RAGEngine(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            llm_client=self.llm_client,
            config=RAGConfig(
                strategy=RAGStrategy.BASIC,
                retrieval_config=RetrievalConfig(
                    strategy=RetrievalStrategy.SIMILARITY,
                    top_k=3
                ),
                augmentation_config=AugmentationConfig(
                    mode=AugmentationMode.BASIC
                )
            )
        )
        
        rag_engine.initialize()
        
        # 测试查询
        query = "什么是深度学习？"
        result = rag_engine.generate(query)
        
        # 验证结果
        self.assertIsNotNone(result.answer, "应该生成有效的回答")
        self.assertEqual(result.query, query, "查询应该与输入一致")
        self.assertGreaterEqual(len(result.retrieval_results), 1, "应该有至少1个检索结果")
        self.assertGreater(result.processing_time, 0, "处理时间应该大于0")
    
    def test_advanced_rag(self):
        """测试高级RAG流程"""
        # 创建RAG引擎
        rag_engine = RAGEngine(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            llm_client=self.llm_client,
            config=RAGConfig(
                strategy=RAGStrategy.ADVANCED,
                retrieval_config=RetrievalConfig(
                    strategy=RetrievalStrategy.HYBRID,
                    top_k=3,
                    hybrid_weight=0.7
                ),
                augmentation_config=AugmentationConfig(
                    mode=AugmentationMode.STRUCTURED
                )
            )
        )
        
        rag_engine.initialize()
        
        # 测试查询
        query = "解释机器学习和人工智能的区别"
        result = rag_engine.generate(query)
        
        # 验证结果
        self.assertIsNotNone(result.answer, "应该生成有效的回答")
        self.assertGreaterEqual(len(result.retrieval_results), 1, "应该有至少1个检索结果")
        
        # 检查是否检索到包含"机器学习"的文档
        has_ml_doc = False
        for res in result.retrieval_results:
            if "机器学习" in res.text:
                has_ml_doc = True
                break
        
        self.assertTrue(has_ml_doc, "应该检索到包含'机器学习'的文档")
    
    def test_metadata_filtering(self):
        """测试元数据过滤"""
        # 创建RAG引擎
        rag_engine = RAGEngine(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            llm_client=self.llm_client,
            config=RAGConfig(
                strategy=RAGStrategy.BASIC,
                retrieval_config=RetrievalConfig(
                    strategy=RetrievalStrategy.SIMILARITY,
                    top_k=3,
                    metadata_filters={"level": "intermediate"}
                )
            )
        )
        
        rag_engine.initialize()
        
        # 测试查询
        query = "自然语言处理有哪些应用?"
        result = rag_engine.generate(query)
        
        # 验证结果
        for res in result.retrieval_results:
            self.assertEqual(res.metadata.get("level"), "intermediate", 
                            "检索结果应该只包含intermediate级别的文档")
    
    def test_result_caching(self):
        """测试结果缓存"""
        # 创建RAG引擎
        rag_engine = RAGEngine(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            llm_client=self.llm_client,
            config=RAGConfig(
                strategy=RAGStrategy.BASIC,
                enable_cache=True,
                cache_size=10
            )
        )
        
        rag_engine.initialize()
        
        # 首次查询
        query = "什么是计算机视觉？"
        result1 = rag_engine.generate(query)
        
        # 记录处理时间
        time1 = result1.processing_time
        
        # 重复查询
        result2 = rag_engine.generate(query)
        time2 = result2.processing_time
        
        # 验证结果
        self.assertEqual(result1.answer, result2.answer, "缓存结果应该与首次查询一致")
        self.assertLessEqual(time2, time1, "使用缓存后处理时间应减少")


if __name__ == "__main__":
    unittest.main()
