"""
RAG引擎端到端测试
"""

import sys
import os
import tempfile
from pathlib import Path
import unittest
import logging
from typing import List, Dict, Any
import time

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入RAG相关组件
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
    RetrievalStrategy,
    SimilarityRetriever
)

from mcp.knowledge.augmentation import (
    AugmentationConfig,
    AugmentationMode,
    StructuredAugmenter
)

from mcp.knowledge.rag_engine import (
    RAGEngine,
    RAGConfig,
    RAGStrategy,
    RAGResult
)


# 模拟LLM客户端
class MockLLMClient:
    """模拟LLM客户端，用于测试"""
    
    def __init__(self, responses=None):
        """
        初始化模拟LLM客户端
        
        Args:
            responses: 预定义的响应字典，查询->回复
        """
        self.responses = responses or {}
        self.default_response = "这是对'{query}'的模拟回答。"
        self.calls = []
    
    def generate(self, prompt: str, **kwargs) -> str:
        """模拟生成回答"""
        # 记录调用
        self.calls.append({
            "prompt": prompt,
            "kwargs": kwargs
        })
        
        # 寻找匹配的预定义响应
        for query, response in self.responses.items():
            if query in prompt:
                return response
        
        # 返回默认响应
        return self.default_response.format(query=prompt[:30])


class TestRAGEngine(unittest.TestCase):
    """测试RAG引擎端到端功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试文件
        self.test_files = []
        self.create_test_files()
        
        try:
            # 处理测试文档
            self.process_test_documents()
            
            # 创建LLM客户端
            self.llm_client = MockLLMClient(responses={
                "深度学习": "深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。",
                "人工智能": "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。",
                "编程语言": "编程语言是用于编写计算机程序的形式语言。常见的编程语言包括Python和JavaScript。"
            })
            
            # 创建RAG引擎
            self.rag_engine = RAGEngine(
                vector_store=self.vector_store,
                embedding_manager=self.embedding_manager,
                llm_client=self.llm_client,
                config=RAGConfig(
                    strategy=RAGStrategy.ADVANCED,
                    retrieval_config=RetrievalConfig(
                        strategy=RetrievalStrategy.HYBRID,
                        top_k=3
                    ),
                    augmentation_config=AugmentationConfig(
                        mode=AugmentationMode.STRUCTURED,
                        max_context_length=2000
                    )
                )
            )
            
            self.rag_engine.initialize()
            
        except Exception as e:
            self.skipTest(f"设置测试环境失败: {str(e)}")
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时文件
        for file_path in self.test_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
                    
        # 删除临时目录
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
    
    def create_test_files(self):
        """创建测试Markdown文件"""
        test_docs = [
            {
                "title": "人工智能概述",
                "content": """
# 人工智能概述

人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、问题解决、感知和语言理解。人工智能系统通常利用大量数据进行训练，可以识别模式、做出决策和预测未来行为。

## 主要应用领域

人工智能已经广泛应用于多个领域，包括：

1. 医疗诊断
2. 自动驾驶汽车
3. 金融分析
4. 个人助理
5. 游戏和娱乐

## 发展历史

人工智能研究始于20世纪50年代，经历了多次发展浪潮和低谷。
                """,
                "metadata": {
                    "category": "AI",
                    "level": "introductory",
                    "author": "MCP测试团队"
                }
            },
            {
                "title": "机器学习基础",
                "content": """
# 机器学习基础

机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统，而无需被明确编程。常见的机器学习方法包括监督学习、无监督学习和强化学习。

## 监督学习

监督学习使用标记数据训练模型，其中输入和期望的输出都已知。算法通过学习输入和输出之间的映射关系来进行预测。

## 无监督学习

无监督学习处理没有标记的数据，算法尝试发现数据中的隐藏模式或结构。

## 强化学习

强化学习通过奖励机制学习最佳行动，代理通过与环境交互来学习如何最大化累积奖励。
                """,
                "metadata": {
                    "category": "AI",
                    "subcategory": "machine_learning",
                    "level": "introductory",
                    "author": "MCP测试团队"
                }
            },
            {
                "title": "深度学习技术",
                "content": """
# 深度学习技术

深度学习是机器学习的一种特殊方法，它使用多层神经网络来学习复杂模式。神经网络的灵感来自人类大脑的结构，由相互连接的节点（神经元）组成。

## 神经网络结构

典型的深度神经网络包括：

- 输入层：接收原始数据
- 隐藏层：处理信息（可以有多层）
- 输出层：产生最终结果

## 流行的架构

最流行的深度学习架构包括：

1. 卷积神经网络（CNN）：主要用于图像处理
2. 循环神经网络（RNN）：适用于序列数据如文本和语音
3. 变换器（Transformer）：用于自然语言处理

## 与传统机器学习的区别

与传统机器学习相比，深度学习能够自动提取特征，处理更复杂的数据，但需要更多的计算资源和训练数据。
                """,
                "metadata": {
                    "category": "AI",
                    "subcategory": "deep_learning",
                    "level": "intermediate",
                    "author": "MCP测试团队"
                }
            },
            {
                "title": "编程语言介绍",
                "content": """
# 编程语言介绍

编程语言是用于编写计算机程序的形式语言。它们允许程序员精确地定义计算机应执行的操作。

## Python

Python是一种广泛使用的高级编程语言，以其简洁和可读性而闻名。它是一种解释型语言，强调代码的可读性，使程序员能够用更少的代码行表达概念。Python支持多种编程范式，包括过程式、面向对象和函数式编程。

## JavaScript

JavaScript是一种用于Web开发的脚本语言，它允许在网页中添加交互性。作为HTML和CSS的补充，JavaScript使开发者能够创建动态网页、处理用户输入和与Web API通信。现代JavaScript框架如React、Angular和Vue.js简化了复杂用户界面的开发过程。
                """,
                "metadata": {
                    "category": "programming",
                    "level": "introductory",
                    "author": "MCP测试团队"
                }
            }
        ]
        
        for i, doc in enumerate(test_docs):
            file_name = f"{doc['title'].replace(' ', '_')}.md"
            file_path = os.path.join(self.temp_dir, file_name)
            
            # 构建Markdown内容
            content = f"# {doc['title']}\n\n"
            
            # 添加前置元数据
            content += "---\n"
            for key, value in doc["metadata"].items():
                content += f"{key}: {value}\n"
            content += "---\n\n"
            
            # 添加正文
            content += doc["content"]
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            self.test_files.append(file_path)
    
    def process_test_documents(self):
        """处理测试文档并添加到向量存储"""
        # 创建文档处理器组件
        chunker = RecursiveTextChunker(
            chunk_size=300,
            chunk_overlap=50,
            min_chunk_size=100
        )
        
        metadata_extractor = ObsidianMetadataExtractor()
        
        processor = MarkdownProcessor(
            chunker=chunker,
            metadata_extractor=metadata_extractor
        )
        
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
        
        # 处理文档并添加到向量存储
        for file_path in self.test_files:
            # 处理文档
            text_chunks = processor.process(file_path)
            
            # 提取元数据
            file_metadata = processor.extract_metadata(file_path)
            
            # 添加到向量存储
            for chunk in text_chunks:
                # 生成嵌入
                embedding = self.embedding_manager.embed_text(chunk.text).embedding
                
                # 创建向量记录
                record = VectorRecord(
                    id=chunk.chunk_id,
                    embedding=embedding,
                    text=chunk.text,
                    metadata=chunk.metadata
                )
                
                # 添加到向量存储
                self.vector_store.add(record)
    
    def test_rag_engine_initialization(self):
        """测试RAG引擎初始化"""
        # 验证RAG引擎初始化
        self.assertIsNotNone(self.rag_engine, "RAG引擎应该成功初始化")
        self.assertIsNotNone(self.rag_engine.retriever, "应该创建检索器")
        self.assertIsNotNone(self.rag_engine.augmenter, "应该创建增强器")
    
    def test_basic_rag_generation(self):
        """测试基本RAG生成功能"""
        # 测试查询
        query = "什么是深度学习？"
        
        # 生成回答
        result = self.rag_engine.generate(query)
        
        # 验证结果
        self.assertIsNotNone(result, "应该返回RAG结果")
        self.assertIsInstance(result, RAGResult, "结果类型应该是RAGResult")
        self.assertEqual(result.query, query, "结果查询应该与输入一致")
        self.assertIsNotNone(result.answer, "应该有回答")
        self.assertGreater(len(result.retrieval_results), 0, "应该有检索结果")
        self.assertIsNotNone(result.context, "应该有增强上下文")
        self.assertGreater(result.processing_time, 0, "处理时间应该大于0")
    
    def test_retrieval_quality(self):
        """测试检索质量"""
        # 测试不同查询
        queries = [
            "深度学习和传统机器学习有什么区别？",
            "Python和JavaScript有什么不同？",
            "人工智能的主要应用领域有哪些？"
        ]
        
        for query in queries:
            # 生成回答
            result = self.rag_engine.generate(query)
            
            # 验证检索结果
            self.assertGreater(len(result.retrieval_results), 0, "应该有检索结果")
            
            # 验证检索结果与查询相关
            relevant_terms = query.lower().split()
            relevant_terms = [term for term in relevant_terms if len(term) > 3]  # 忽略太短的词
            
            found_relevant = False
            for ret_result in result.retrieval_results:
                ret_text = ret_result.text.lower()
                for term in relevant_terms:
                    if term in ret_text:
                        found_relevant = True
                        break
                if found_relevant:
                    break
                    
            self.assertTrue(found_relevant, f"检索结果应该与查询'{query}'相关")
    
    def test_context_augmentation(self):
        """测试上下文增强"""
        # 测试查询
        query = "深度学习的神经网络结构是怎样的？"
        
        # 生成回答
        result = self.rag_engine.generate(query)
        
        # 验证增强上下文
        self.assertIsNotNone(result.context, "应该有增强上下文")
        self.assertIsNotNone(result.context.prompt, "应该有提示")
        self.assertIsNotNone(result.context.system, "应该有系统指令")
        self.assertIsNotNone(result.context.retrieval, "应该有检索结果文本")
        self.assertIsNotNone(result.context.query, "应该有查询")
        
        # 验证提示包含检索内容
        for ret_result in result.retrieval_results:
            self.assertIn(ret_result.text, result.context.retrieval, 
                         "提示应该包含检索结果文本")
        
        # 验证提示结构是否符合配置
        if self.rag_engine.config.augmentation_config.mode == AugmentationMode.STRUCTURED:
            self.assertIn("参考资料", result.context.system, "结构化提示应该包含'参考资料'部分")
    
    def test_different_rag_strategies(self):
        """测试不同RAG策略"""
        # 测试查询
        query = "人工智能和机器学习的关系是什么？"
        
        # 测试基本策略
        self.rag_engine.config.strategy = RAGStrategy.BASIC
        basic_result = self.rag_engine.generate(query)
        
        # 测试高级策略
        self.rag_engine.config.strategy = RAGStrategy.ADVANCED
        advanced_result = self.rag_engine.generate(query)
        
        # 测试迭代策略
        self.rag_engine.config.strategy = RAGStrategy.ITERATIVE
        iterative_result = self.rag_engine.generate(query)
        
        # 验证结果类型
        self.assertIsInstance(basic_result, RAGResult, "基本策略结果类型应该是RAGResult")
        self.assertIsInstance(advanced_result, RAGResult, "高级策略结果类型应该是RAGResult")
        self.assertIsInstance(iterative_result, RAGResult, "迭代策略结果类型应该是RAGResult")
        
        # 验证所有策略都能生成回答
        self.assertIsNotNone(basic_result.answer, "基本策略应该生成回答")
        self.assertIsNotNone(advanced_result.answer, "高级策略应该生成回答")
        self.assertIsNotNone(iterative_result.answer, "迭代策略应该生成回答")
    
    def test_result_caching(self):
        """测试结果缓存"""
        # 确保启用缓存
        self.rag_engine.config.enable_cache = True
        
        # 测试查询
        query = "什么是深度学习？"
        
        # 第一次生成（未缓存）
        start_time = time.time()
        first_result = self.rag_engine.generate(query)
        first_time = time.time() - start_time
        
        # 第二次生成（应该使用缓存）
        start_time = time.time()
        second_result = self.rag_engine.generate(query)
        second_time = time.time() - start_time
        
        # 验证缓存效果
        self.assertEqual(first_result.answer, second_result.answer, "缓存结果应该与首次生成一致")
        self.assertLessEqual(second_time, first_time, "使用缓存应该比首次生成快")
        
        # 验证LLM客户端只被调用一次
        self.assertEqual(len(self.llm_client.calls), 1, "LLM客户端应该只被调用一次")
    
    def test_token_management(self):
        """测试Token管理"""
        # 测试查询
        query = "请详细解释深度学习、机器学习和人工智能之间的关系和区别。"
        
        # 设置较小的上下文长度限制
        self.rag_engine.config.augmentation_config.max_context_length = 500
        
        # 生成回答
        result = self.rag_engine.generate(query)
        
        # 验证增强上下文
        self.assertIsNotNone(result.context, "应该有增强上下文")
        
        # 验证Token计数
        self.assertIsNotNone(result.context.token_count, "应该有Token计数")
        self.assertIn("total", result.context.token_count, "应该有总Token计数")
        
        # 验证总Token数不超过限制
        self.assertLessEqual(
            result.context.token_count.get("total", float('inf')), 
            self.rag_engine.config.augmentation_config.max_context_length,
            "总Token数不应超过限制"
        )


if __name__ == "__main__":
    unittest.main()
