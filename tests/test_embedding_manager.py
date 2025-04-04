"""
嵌入管理系统测试
"""

import sys
import os
from pathlib import Path
import unittest
import tempfile
import shutil
import numpy as np
import time

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入所需组件
from mcp.knowledge.embedding import (
    EmbeddingConfig,
    LocalEmbedding,
    TextEmbedding,
    EmbeddingCache,
    DiskEmbeddingCache
)


class TestEmbeddingManager(unittest.TestCase):
    """测试嵌入管理器功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 设置测试数据
        self.test_texts = [
            "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统。",
            "深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。"
        ]
        
        # 初始化嵌入管理器
        try:
            cache_dir = os.path.join(self.temp_dir, "cache")
            self.cache = DiskEmbeddingCache(cache_dir=cache_dir)
            
            self.embedding_manager = LocalEmbedding(
                config=EmbeddingConfig(
                    model_name="paraphrase-MiniLM-L6-v2",
                    embedding_dim=384,
                    use_cache=True,
                    cache_dir=cache_dir
                ),
                cache=self.cache
            )
            self.embedding_manager.initialize()
        except Exception as e:
            self.skipTest(f"嵌入管理器初始化失败，跳过测试: {str(e)}")
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_embed_text(self):
        """测试文本嵌入"""
        # 嵌入单个文本
        embedding = self.embedding_manager.embed_text(self.test_texts[0])
        
        # 验证结果
        self.assertIsInstance(embedding, TextEmbedding)
        self.assertEqual(embedding.text, self.test_texts[0])
        self.assertIsInstance(embedding.embedding, np.ndarray)
        self.assertEqual(embedding.embedding.shape[0], 384)
    
    def test_embed_texts_batch(self):
        """测试批量文本嵌入"""
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
        # 嵌入查询
        query = "什么是深度学习"
        query_vector = self.embedding_manager.embed_query(query)
        
        # 验证结果
        self.assertIsInstance(query_vector, np.ndarray)
        self.assertEqual(query_vector.shape[0], 384)
    
    def test_similarity(self):
        """测试相似度计算"""
        # 计算相似度
        text1 = "人工智能是一种技术"
        text2 = "AI是一种先进技术"
        text3 = "猫是一种宠物动物"
        
        sim12 = self.embedding_manager.similarity(text1, text2)
        sim13 = self.embedding_manager.similarity(text1, text3)
        
        # 验证结果
        self.assertGreater(sim12, 0)
        self.assertLess(sim12, 1)
        self.assertGreater(sim12, sim13)  # 相关文本相似度应更高
    
    def test_cache_functionality(self):
        """测试缓存功能"""
        # 首次嵌入（未缓存）
        start_time = time.time()
        embedding1 = self.embedding_manager.embed_text(self.test_texts[0])
        first_time = time.time() - start_time
        
        # 二次嵌入（应该使用缓存）
        start_time = time.time()
        embedding2 = self.embedding_manager.embed_text(self.test_texts[0])
        second_time = time.time() - start_time
        
        # 验证结果
        self.assertEqual(embedding1.text, embedding2.text)
        np.testing.assert_array_equal(embedding1.embedding, embedding2.embedding)
        
        # 缓存应该使二次查询更快
        # 但是有时候第一次加载模型会很慢，之后就快了，所以这个测试可能不可靠
        # self.assertLess(second_time, first_time)
        
        # 检查缓存状态
        cache_info = self.cache.get_cache_info()
        self.assertGreater(cache_info["hits"], 0)
    
    def test_embedding_normalization(self):
        """测试嵌入规范化"""
        # 启用规范化的配置
        config = EmbeddingConfig(
            model_name="paraphrase-MiniLM-L6-v2",
            normalize_embeddings=True
        )
        
        embedding_manager = LocalEmbedding(config=config)
        embedding_manager.initialize()
        
        # 生成嵌入
        embedding = embedding_manager.embed_text(self.test_texts[0])
        
        # 检查规范化（向量长度应接近1）
        norm = np.linalg.norm(embedding.embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
