"""
嵌入管理系统测试 - 测试文本嵌入和缓存功能
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import shutil
import numpy as np

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

from mcp.knowledge.embedding import (
    EmbeddingConfig,
    LocalEmbedding,
    TextEmbedding,
    EmbeddingCache,
    DiskEmbeddingCache
)


class TestEmbedding(unittest.TestCase):
    """测试嵌入管理系统"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建缓存目录
        self.cache_dir = os.path.join(self.temp_dir, "embedding_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 创建测试文本
        self.test_texts = [
            "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统。",
            "深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。"
        ]
        
        # 创建嵌入配置
        try:
            self.config = EmbeddingConfig(
                model_name="paraphrase-MiniLM-L6-v2",
                embedding_dim=384,
                batch_size=2,
                use_cache=True,
                cache_dir=self.cache_dir
            )
            
            # 创建嵌入缓存
            self.cache = DiskEmbeddingCache(cache_dir=self.cache_dir)
            
            # 创建嵌入管理器
            self.embedding_manager = LocalEmbedding(
                config=self.config,
                cache=self.cache
            )
            
            # 尝试初始化嵌入管理器
            try:
                self.embedding_manager.initialize()
                self.skip_tests = False
            except ImportError:
                self.skip_tests = True
                print("警告: sentence-transformers未安装，将跳过嵌入测试")
        except Exception as e:
            self.skip_tests = True
            print(f"警告: 嵌入管理器初始化失败: {str(e)}，将跳过嵌入测试")
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_embedding_initialization(self):
        """测试嵌入管理器初始化"""
        if self.skip_tests:
            self.skipTest("嵌入管理器初始化失败，跳过测试")
            
        self.assertTrue(self.embedding_manager._initialized)
        self.assertEqual(self.embedding_manager.config.model_name, "paraphrase-MiniLM-L6-v2")
        self.assertEqual(self.embedding_manager.config.embedding_dim, 384)
    
    def test_embed_text(self):
        """测试文本嵌入"""
        if self.skip_tests:
            self.skipTest("嵌入管理器初始化失败，跳过测试")
            
        # 嵌入单个文本
        text = self.test_texts[0]
        embedding = self.embedding_manager.embed_text(text)
        
        # 验证结果
        self.assertIsInstance(embedding, TextEmbedding)
        self.assertEqual(embedding.text, text)
        self.assertIsInstance(embedding.embedding, np.ndarray)
        self.assertEqual(embedding.embedding.shape[0], 384)
    
    def test_embed_texts_batch(self):
        """测试批量文本嵌入"""
        if self.skip_tests:
            self.skipTest("嵌入管理器初始化失败，跳过测试")
            
        # 批量嵌入文本
        embeddings = self.embedding_manager.embed_texts(self.test_texts)
        
        # 验证结果
        self.assertEqual(len(embeddings), len(self.test_texts))
        for i, embedding in enumerate(embeddings):
            self.assertIsInstance(embedding, TextEmbedding)
            self.assertEqual(embedding.text, self.test_texts[i])
            self.assertIsInstance(embedding.embedding, np.ndarray)
            self.assertEqual(embedding.embedding.shape[0], 384)
    
    def test_embed_query(self):
        """测试查询嵌入"""
        if self.skip_tests:
            self.skipTest("嵌入管理器初始化失败，跳过测试")
            
        # 嵌入查询
        query = "什么是深度学习？"
        query_vector = self.embedding_manager.embed_query(query)
        
        # 验证结果
        self.assertIsInstance(query_vector, np.ndarray)
        self.assertEqual(query_vector.shape[0], 384)
    
    def test_similarity(self):
        """测试相似度计算"""
        if self.skip_tests:
            self.skipTest("嵌入管理器初始化失败，跳过测试")
            
        # 计算相似度
        text1 = "深度学习是机器学习的一种方法"
        text2 = "深度学习使用神经网络进行学习"
        text3 = "气候变化是一个全球性问题"
        
        sim12 = self.embedding_manager.similarity(text1, text2)
        sim13 = self.embedding_manager.similarity(text1, text3)
        
        # 验证结果
        self.assertGreater(sim12, 0)
        self.assertLess(sim12, 1.1)  # 可能因为浮点误差略大于1
        self.assertGreater(sim13, 0)
        self.assertLess(sim13, 1.1)
        
        # 相关文本的相似度应该高于不相关文本
        self.assertGreater(sim12, sim13)
    
    def test_embedding_cache(self):
        """测试嵌入缓存"""
        if self.skip_tests:
            self.skipTest("嵌入管理器初始化失败，跳过测试")
            
        # 确保缓存目录存在
        self.assertTrue(os.path.exists(self.cache_dir))
        
        # 首次嵌入应该写入缓存
        text = "这是一个测试文本，用于验证缓存功能"
        first_embedding = self.embedding_manager.embed_text(text)
        
        # 验证缓存文件创建
        cache_files = os.listdir(self.cache_dir)
        self.assertGreater(len(cache_files), 0)
        
        # 记录当前缓存文件修改时间
        cache_mod_times = {}
        for file in cache_files:
            if file.endswith('.npy'):
                file_path = os.path.join(self.cache_dir, file)
                cache_mod_times[file] = os.path.getmtime(file_path)
        
        # 再次嵌入同一文本，应该使用缓存
        second_embedding = self.embedding_manager.embed_text(text)
        
        # 验证两次嵌入结果相同
        np.testing.assert_array_equal(
            first_embedding.embedding,
            second_embedding.embedding
        )
        
        # 缓存文件不应该被修改
        for file in cache_files:
            if file.endswith('.npy'):
                file_path = os.path.join(self.cache_dir, file)
                self.assertEqual(
                    cache_mod_times.get(file),
                    os.path.getmtime(file_path)
                )
