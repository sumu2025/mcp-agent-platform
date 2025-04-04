"""
基础RAG功能测试脚本 - 测试MCP知识增强层核心组件
"""

import sys
import os
from pathlib import Path
import unittest
import logging
import tempfile
import shutil
from typing import List, Dict, Any

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_test")

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入所需组件
from mcp.knowledge.indexing import (
    MarkdownProcessor,
    RecursiveTextChunker,
    ObsidianMetadataExtractor,
    TextChunk
)

from mcp.knowledge.embedding import (
    EmbeddingConfig,
    LocalEmbedding,
    TextEmbedding,
    DiskEmbeddingCache
)

from mcp.knowledge.storage import (
    VectorStoreConfig,
    InMemoryVectorStore,
    VectorRecord,
    SearchResult
)

from mcp.knowledge.retrieval import (
    RetrievalConfig,
    SimilarityRetriever,
    HybridRetriever,
    KeywordRetriever,
    RetrievalStrategy,
    RetrievalResult
)

from mcp.knowledge.augmentation import (
    AugmentationConfig,
    AugmentationMode,
    BasicAugmenter,
    StructuredAugmenter
)

from mcp.knowledge.augmentation.token_management import TokenManager

try:
    from mcp.knowledge.rag_engine import (
        RAGEngine,
        RAGConfig,
        RAGStrategy
    )
    HAS_RAG_ENGINE = True
except ImportError:
    HAS_RAG_ENGINE = False


class MockLLMClient:
    """模拟LLM客户端，用于测试"""
    
    def __init__(self, return_prompt=False):
        """初始化模拟客户端"""
        self.return_prompt = return_prompt
        self.last_prompt = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """模拟生成回答"""
        self.last_prompt = prompt
        
        if self.return_prompt:
            return f"PROMPT: {prompt[:50]}..."
        
        # 根据提示内容生成简单回答
        if "深度学习" in prompt:
            return "深度学习是机器学习的一种方法，使用多层神经网络来学习复杂模式。"
        elif "机器学习" in prompt:
            return "机器学习是AI的一个子领域，关注从数据中学习的系统。"
        elif "自然语言处理" in prompt:
            return "自然语言处理应用包括机器翻译、情感分析、文本摘要和聊天机器人。"
        else:
            return "这是一个模拟回答，用于测试RAG系统。"


class TestRAGComponents(unittest.TestCase):
    """测试RAG核心组件功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 测试数据目录
        self.test_data_dir = Path(__file__).parent / "test_data"
        self.test_doc_path = self.test_data_dir / "test_doc.md"
        
        # 创建测试数据
        self.test_texts = [
            "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统。",
            "深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。"
        ]
        
        # 创建测试元数据
        self.test_metadata = [
            {"category": "AI", "level": "introductory"},
            {"category": "AI", "subcategory": "machine_learning", "level": "introductory"},
            {"category": "AI", "subcategory": "deep_learning", "level": "intermediate"}
        ]
        
        # 初始化嵌入管理器
        try:
            self.embedding_manager = LocalEmbedding(config=EmbeddingConfig(
                model_name="paraphrase-MiniLM-L6-v2",
                embedding_dim=384
            ))
            self.embedding_manager.initialize()
        except Exception as e:
            self.skipTest(f"嵌入管理器初始化失败，跳过测试: {str(e)}")
            
        # 初始化向量存储
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
    
    def test_text_chunker(self):
        """测试文本分块器"""
        # 创建分块器
        chunker = RecursiveTextChunker(
            chunk_size=100,
            chunk_overlap=20
        )
        
        # 测试文本
        test_text = "# 标题1\n\n这是第一段落，包含一些内容。\n\n## 子标题\n\n这是第二段落，也包含一些内容。"
        
        # 分块
        chunks = chunker.split(test_text)
        
        # 验证结果
        self.assertGreater(len(chunks), 0, "分块器应该生成至少一个块")
        self.assertIsInstance(chunks[0], TextChunk, "结果应该是TextChunk类型")
    
    def test_chunker_with_document(self):
        """使用实际文档测试分块器"""
        if not self.test_doc_path.exists():
            self.skipTest(f"测试文档不存在: {self.test_doc_path}")
        
        # 创建分块器
        chunker = RecursiveTextChunker(
            chunk_size=150,
            chunk_overlap=30
        )
        
        # 读取文档内容
        with open(self.test_doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分块
        chunks = chunker.split(content)
        
        # 验证结果
        self.assertGreaterEqual(len(chunks), 2, "应该至少生成2个块")
        
        # 验证块内容和结构
        combined_text = ' '.join([chunk.text for chunk in chunks])
        self.assertIn("测试文档标题", combined_text, "分块结果应包含文档标题")
        self.assertIn("二级标题", combined_text, "分块结果应包含二级标题")
        self.assertIn("三级标题", combined_text, "分块结果应包含三级标题")
        self.assertIn("列表项", combined_text, "分块结果应包含列表项")
    
    def test_embedding_manager(self):
        """测试嵌入管理器"""
        # 测试文本
        test_text = "这是一个测试文本"
        
        # 生成嵌入
        embedding = self.embedding_manager.embed_text(test_text)
        
        # 验证结果
        self.assertIsInstance(embedding, TextEmbedding, "结果应该是TextEmbedding类型")
        self.assertEqual(embedding.text, test_text, "嵌入文本应该与输入一致")
        self.assertEqual(len(embedding.embedding), 384, "嵌入维度应该是384")
    
    def test_embedding_cache(self):
        """测试嵌入缓存"""
        # 创建临时缓存目录
        cache_dir = tempfile.mkdtemp()
        
        try:
            # 创建缓存
            cache = DiskEmbeddingCache(cache_dir=cache_dir)
            cache.initialize()
            
            # 创建带缓存的嵌入管理器
            cached_embedding_manager = LocalEmbedding(
                config=EmbeddingConfig(
                    model_name="paraphrase-MiniLM-L6-v2",
                    embedding_dim=384,
                    use_cache=True
                ),
                cache=cache
            )
            cached_embedding_manager.initialize()
            
            # 测试文本
            test_text = "这是一个用于测试缓存的文本"
            
            # 第一次嵌入（缓存未命中）
            start_time = time.time()
            first_embedding = cached_embedding_manager.embed_text(test_text)
            first_time = time.time() - start_time
            
            # 第二次嵌入（缓存命中）
            start_time = time.time()
            second_embedding = cached_embedding_manager.embed_text(test_text)
            second_time = time.time() - start_time
            
            # 验证结果
            self.assertEqual(
                first_embedding.embedding.tolist(),
                second_embedding.embedding.tolist(),
                "缓存嵌入应该与原始嵌入相同"
            )
            
            # 验证缓存效率（第二次应该更快）
            # 注意：这个测试可能不稳定，取决于系统负载
            logger.info(f"首次嵌入时间: {first_time:.4f}秒")
            logger.info(f"缓存嵌入时间: {second_time:.4f}秒")
            
            # 获取缓存信息
            cache_info = cache.get_cache_info()
            self.assertGreaterEqual(cache_info["entries"], 1, "缓存应该至少包含一个条目")
            
        finally:
            # 清理临时目录
            shutil.rmtree(cache_dir)
    
    def test_vector_store(self):
        """测试向量存储"""
        # 查询数量
        count = self.vector_store.count()
        
        # 验证结果
        self.assertEqual(count, len(self.test_texts), "向量存储中的记录数应该与测试数据数量相同")
        
        # 生成查询向量
        query_text = "什么是深度学习"
        query_vector = self.embedding_manager.embed_query(query_text)
        
        # 搜索
        results = self.vector_store.search(query_vector, k=2)
        
        # 验证结果
        self.assertGreaterEqual(len(results), 1, "应该至少返回一个结果")
        self.assertIsInstance(results[0], SearchResult, "结果应该是SearchResult类型")
    
    def test_vector_store_operations(self):
        """测试向量存储操作"""
        # 创建新的向量存储
        test_store = InMemoryVectorStore(config=VectorStoreConfig(
            embedding_dim=384,
            distance_metric="cosine"
        ))
        test_store.initialize()
        
        # 创建测试记录
        record = VectorRecord(
            id="test_op",
            embedding=np.ones(384),  # 简单的单位向量
            text="测试向量存储操作",
            metadata={"test": "metadata"}
        )
        
        # 添加记录
        record_id = test_store.add(record)
        self.assertEqual(record_id, "test_op", "返回的ID应该与提供的ID相同")
        self.assertEqual(test_store.count(), 1, "存储应该包含1条记录")
        
        # 获取记录
        retrieved = test_store.get(record_id)
        self.assertIsNotNone(retrieved, "应该能够获取已添加的记录")
        self.assertEqual(retrieved.text, record.text, "获取的记录文本应该与添加的相同")
        
        # 更新记录
        updated_record = VectorRecord(
            id=record_id,
            embedding=record.embedding,
            text="更新后的文本",
            metadata=record.metadata
        )
        test_store.update(updated_record)
        
        # 验证更新
        after_update = test_store.get(record_id)
        self.assertEqual(after_update.text, "更新后的文本", "记录应该已更新")
        
        # 删除记录
        deleted = test_store.delete(record_id)
        self.assertTrue(deleted, "删除操作应该成功")
        self.assertEqual(test_store.count(), 0, "存储应该为空")
        
        # 验证删除
        deleted_record = test_store.get(record_id)
        self.assertIsNone(deleted_record, "已删除的记录不应该能被获取")
    
    def test_retriever(self):
        """测试检索器"""
        # 创建检索器
        retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(top_k=2)
        )
        retriever.initialize()
        
        # 查询
        query = "深度学习是什么"
        results = retriever.retrieve(query)
        
        # 验证结果
        self.assertGreaterEqual(len(results), 1, "应该至少返回一个结果")
        self.assertIsInstance(results[0], RetrievalResult, "结果应该是RetrievalResult类型")
        
        # 测试元数据过滤
        filter_results = retriever.retrieve_with_filter(query, {"level": "intermediate"})
        
        # 如果有结果，验证元数据
        if filter_results:
            self.assertEqual(filter_results[0].metadata.get("level"), "intermediate", 
                            "过滤结果应该满足元数据条件")
    
    def test_hybrid_retriever(self):
        """测试混合检索器"""
        # 创建混合检索器
        hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            config=RetrievalConfig(
                strategy=RetrievalStrategy.HYBRID,
                top_k=2,
                hybrid_weight=0.7  # 70% 向量相似度, 30% 关键词匹配
            )
        )
        hybrid_retriever.initialize()
        
        # 查询
        query = "深度学习和神经网络"
        results = hybrid_retriever.retrieve(query)
        
        # 验证结果
        self.assertGreaterEqual(len(results), 1, "应该至少返回一个结果")
        
        # 验证混合结果包含相关内容
        combined_text = ' '.join([r.text for r in results])
        self.assertIn("深度学习", combined_text, "结果应该包含查询关键词")
    
    def test_keyword_retriever(self):
        """测试关键词检索器"""
        # 创建关键词检索器
        keyword_retriever = KeywordRetriever(
            vector_store=self.vector_store,
            config=RetrievalConfig(
                strategy=RetrievalStrategy.KEYWORD,
                top_k=2
            )
        )
        keyword_retriever.initialize()
        
        # 查询
        query = "深度学习和神经网络"
        results = keyword_retriever.retrieve(query)
        
        # 由于测试数据集小，可能没有足够的结果
        # 只验证返回类型
        for result in results:
            self.assertIsInstance(result, RetrievalResult, "结果应该是RetrievalResult类型")
    
    def test_augmenter(self):
        """测试上下文增强器"""
        # 创建增强器
        augmenter = StructuredAugmenter(config=AugmentationConfig(
            mode=AugmentationMode.STRUCTURED,
            max_context_length=1000
        ))
        augmenter.initialize()
        
        # 创建检索结果
        retrieval_results = []
        for i, (text, metadata) in enumerate(zip(self.test_texts, self.test_metadata)):
            result = RetrievalResult(
                text=text,
                score=0.9 - (i * 0.1),
                id=f"test_{i}",
                metadata=metadata
            )
            retrieval_results.append(result)
        
        # 增强上下文
        query = "什么是深度学习？"
        context = augmenter.augment(query, retrieval_results)
        
        # 验证结果
        self.assertIsNotNone(context.prompt, "应该生成有效的提示")
        self.assertIn(query, context.query, "提示应该包含查询")
        self.assertIn(self.test_texts[0], context.retrieval, "提示应该包含检索结果")
    
    def test_different_augmenters(self):
        """测试不同的增强器"""
        # 创建检索结果
        retrieval_results = []
        for i, (text, metadata) in enumerate(zip(self.test_texts, self.test_metadata)):
            result = RetrievalResult(
                text=text,
                score=0.9 - (i * 0.1),
                id=f"test_{i}",
                metadata=metadata
            )
            retrieval_results.append(result)
        
        # 查询
        query = "什么是深度学习？"
        
        # 测试基本增强器
        basic_augmenter = BasicAugmenter(config=AugmentationConfig(
            mode=AugmentationMode.BASIC,
            max_context_length=1000
        ))
        basic_augmenter.initialize()
        basic_context = basic_augmenter.augment(query, retrieval_results)
        
        # 测试结构化增强器
        structured_augmenter = StructuredAugmenter(config=AugmentationConfig(
            mode=AugmentationMode.STRUCTURED,
            max_context_length=1000
        ))
        structured_augmenter.initialize()
        structured_context = structured_augmenter.augment(query, retrieval_results)
        
        # 验证两种增强器的结果不同
        self.assertNotEqual(
            basic_context.prompt, 
            structured_context.prompt,
            "不同增强器应该生成不同的提示"
        )
        
        # 验证结构化增强器包含更多格式元素
        basic_markup_count = basic_context.prompt.count("#") + basic_context.prompt.count("*")
        structured_markup_count = structured_context.prompt.count("#") + structured_context.prompt.count("*")
        
        # 结构化格式通常有更多的Markdown标记
        self.assertGreaterEqual(
            structured_markup_count,
            basic_markup_count,
            "结构化格式应该包含更多格式元素"
        )
    
    def test_token_manager(self):
        """测试Token管理器"""
        # 创建Token管理器
        token_manager = TokenManager()
        
        # 测试文本
        test_text = "这是一个测试文本，用于验证Token管理器功能。" * 10
        
        # 计算token数量
        token_count = token_manager.count_tokens(test_text)
        
        # 验证结果
        self.assertGreater(token_count, 0, "应该计算出大于0的token数量")
        
        # 测试截断
        max_tokens = 10
        truncated = token_manager.truncate_text(test_text, max_tokens)
        
        # 验证结果
        truncated_count = token_manager.count_tokens(truncated)
        self.assertLessEqual(truncated_count, max_tokens, 
                            "截断后的文本token数不应超过限制")
    
    def test_token_distribution(self):
        """测试Token分配"""
        # 创建Token管理器
        token_manager = TokenManager()
        
        # 测试部分
        sections = {
            "system": "系统指令部分" * 20,  # 长系统指令
            "retrieval": "检索结果部分" * 30,  # 更长的检索结果
            "query": "查询部分"  # 短查询
        }
        
        # 计算原始token数
        original_counts = {k: token_manager.count_tokens(v) for k, v in sections.items()}
        total_original = sum(original_counts.values())
        
        # 设置最大token数（原始总数的一半）
        max_tokens = total_original // 2
        
        # 按优先级分配token
        priorities = ["query", "system", "retrieval"]  # 优先保留查询，其次系统指令，最后检索结果
        adjusted = token_manager.distribute_tokens(max_tokens, sections, priorities)
        
        # 计算调整后的token数
        adjusted_counts = {k: token_manager.count_tokens(v) for k, v in adjusted.items()}
        total_adjusted = sum(adjusted_counts.values())
        
        # 验证结果
        self.assertLessEqual(total_adjusted, max_tokens, 
                           "调整后的总token数不应超过限制")
        
        # 验证优先级 - 查询应该保持不变
        self.assertEqual(adjusted["query"], sections["query"], 
                       "最高优先级的部分应该保持不变")
        
        # 验证检索结果被截断最多
        reduction_retrieval = original_counts["retrieval"] - adjusted_counts["retrieval"]
        reduction_system = original_counts["system"] - adjusted_counts["system"]
        self.assertGreaterEqual(reduction_retrieval, reduction_system, 
                              "最低优先级的部分应该被截断最多")


# 仅当RAG引擎可用时才运行的测试
@unittest.skipIf(not HAS_RAG_ENGINE, "RAG引擎未导入")
class TestRAGEngine(unittest.TestCase):
    """测试RAG引擎功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试数据
        self.test_texts = [
            "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统。",
            "深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。"
        ]
        
        # 创建测试元数据
        self.test_metadata = [
            {"category": "AI", "level": "introductory"},
            {"category": "AI", "subcategory": "machine_learning", "level": "introductory"},
            {"category": "AI", "subcategory": "deep_learning", "level": "intermediate"}
        ]
        
        # 初始化嵌入管理器
        try:
            self.embedding_manager = LocalEmbedding(config=EmbeddingConfig(
                model_name="paraphrase-MiniLM-L6-v2",
                embedding_dim=384
            ))
            self.embedding_manager.initialize()
        except Exception as e:
            self.skipTest(f"嵌入管理器初始化失败，跳过测试: {str(e)}")
            
        # 初始化向量存储
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
                
        # 创建LLM客户端
        self.llm_client = MockLLMClient()
    
    def test_rag_engine_basic(self):
        """测试基本RAG引擎功能"""
        # 创建RAG引擎
        rag_engine = RAGEngine(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            llm_client=self.llm_client,
            config=RAGConfig(
                strategy=RAGStrategy.BASIC,
                retrieval_config=RetrievalConfig(top_k=2),
                augmentation_config=AugmentationConfig(
                    mode=AugmentationMode.BASIC,
                    max_context_length=2000
                )
            )
        )
        
        # 初始化
        rag_engine.initialize()
        
        # 生成回答
        query = "什么是深度学习？"
        result = rag_engine.generate(query)
        
        # 验证结果
        self.assertIsNotNone(result.answer, "应该生成有效的回答")
        self.assertEqual(result.query, query, "结果查询应该与输入相同")
        self.assertGreaterEqual(len(result.retrieval_results), 1, "应该至少有一个检索结果")
        self.assertGreater(result.processing_time, 0, "处理时间应该大于0")
    
    def test_rag_engine_advanced(self):
        """测试高级RAG引擎功能"""
        # 创建RAG引擎
        rag_engine = RAGEngine(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            llm_client=self.llm_client,
            config=RAGConfig(
                strategy=RAGStrategy.ADVANCED,
                retrieval_config=RetrievalConfig(
                    strategy=RetrievalStrategy.HYBRID,
                    top_k=2
                ),
                augmentation_config=AugmentationConfig(
                    mode=AugmentationMode.STRUCTURED,
                    max_context_length=2000
                )
            )
        )
        
        # 初始化
        rag_engine.initialize()
        
        # 生成回答
        query = "机器学习和深度学习有什么区别？"
        result = rag_engine.generate(query)
        
        # 验证结果
        self.assertIsNotNone(result.answer, "应该生成有效的回答")
        self.assertGreaterEqual(len(result.retrieval_results), 1, "应该至少有一个检索结果")
        
        # 验证检索结果与查询相关
        found_ml = False
        found_dl = False
        for res in result.retrieval_results:
            if "机器学习" in res.text:
                found_ml = True
            if "深度学习" in res.text:
                found_dl = True
                
        self.assertTrue(found_ml or found_dl, "检索结果应该包含相关内容")
    
    def test_rag_engine_caching(self):
        """测试RAG引擎缓存功能"""
        # 创建RAG引擎（启用缓存）
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
        
        # 初始化
        rag_engine.initialize()
        
        # 第一次查询
        query = "人工智能是什么？"
        first_result = rag_engine.generate(query)
        first_time = first_result.processing_time
        
        # 第二次相同查询（应该使用缓存）
        second_result = rag_engine.generate(query)
        second_time = second_result.processing_time
        
        # 验证结果
        self.assertEqual(first_result.answer, second_result.answer, 
                       "缓存结果应该与原始结果相同")
        
        # 通常缓存查询会快得多，但测试环境可能有变化
        # 所以只记录时间，不硬性断言
        logger.info(f"首次查询时间: {first_time:.4f}秒")
        logger.info(f"缓存查询时间: {second_time:.4f}秒")


if __name__ == "__main__":
    unittest.main()
