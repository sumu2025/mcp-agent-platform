"""
增强的嵌入测试
更全面地测试嵌入管理器和缓存功能
"""

import sys
import os
import unittest
import tempfile
import shutil
import time
from pathlib import Path
import numpy as np

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入所需组件
from mcp.knowledge.embedding import (
    EmbeddingConfig,
    LocalEmbedding,
    TextEmbedding,
    DiskEmbeddingCache,
    MemoryEmbeddingCache
)


class TestEmbeddingEnhanced(unittest.TestCase):
    """增强的嵌入测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        try:
            # 初始化嵌入管理器
            cls.embedding_manager = LocalEmbedding(config=EmbeddingConfig(
                model_name="paraphrase-MiniLM-L6-v2",
                embedding_dim=384,
                use_cache=False  # 先禁用缓存进行基础测试
            ))
            cls.embedding_manager.initialize()
        except Exception as e:
            raise unittest.SkipTest(f"嵌入管理器初始化失败，跳过测试: {str(e)}")
        
        # 测试数据
        cls.test_texts = [
            "人工智能是计算机科学的一个分支。",
            "机器学习是人工智能的一个子领域。",
            "深度学习是机器学习的一种方法。"
        ]
    
    def setUp(self):
        """每个测试前的设置"""
        # 创建临时目录用于缓存测试
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """每个测试后的清理"""
        # 清理临时目录
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_embedding(self):
        """测试基本嵌入功能"""
        # 嵌入单个文本
        text = "这是一个测试文本"
        embedding = self.embedding_manager.embed_text(text)
        
        # 验证结果
        self.assertIsInstance(embedding, TextEmbedding, "结果应该是TextEmbedding类型")
        self.assertEqual(embedding.text, text, "嵌入文本应该与输入一致")
        self.assertEqual(len(embedding.embedding), 384, "嵌入维度应该是384")
        
        # 测试嵌入向量是否规范化
        norm = np.linalg.norm(embedding.embedding)
        self.assertAlmostEqual(norm, 1.0, delta=0.01, msg="嵌入向量应该被规范化")
    
    def test_batch_embedding(self):
        """测试批量嵌入功能"""
        # 批量嵌入
        embeddings = self.embedding_manager.embed_texts(self.test_texts)
        
        # 验证结果
        self.assertEqual(len(embeddings), len(self.test_texts), "应该返回与输入相同数量的嵌入")
        
        for i, embedding in enumerate(embeddings):
            self.assertEqual(embedding.text, self.test_texts[i], "嵌入文本应该与输入一致")
            self.assertEqual(len(embedding.embedding), 384, "嵌入维度应该是384")
    
    def test_query_embedding(self):
        """测试查询嵌入功能"""
        # 嵌入查询
        query = "什么是深度学习？"
        query_vector = self.embedding_manager.embed_query(query)
        
        # 验证结果
        self.assertIsInstance(query_vector, np.ndarray, "结果应该是numpy数组")
        self.assertEqual(len(query_vector), 384, "嵌入维度应该是384")
        
        # 测试嵌入向量是否规范化
        norm = np.linalg.norm(query_vector)
        self.assertAlmostEqual(norm, 1.0, delta=0.01, msg="嵌入向量应该被规范化")
    
    def test_similarity_calculation(self):
        """测试相似度计算功能"""
        # 计算相似文本的相似度
        text1 = "深度学习是机器学习的一种方法"
        text2 = "深度学习是一种基于神经网络的机器学习方法"
        
        similarity = self.embedding_manager.similarity(text1, text2)
        
        # 验证结果
        self.assertIsInstance(similarity, float, "结果应该是浮点数")
        self.assertGreaterEqual(similarity, 0.0, "相似度应该大于等于0")
        self.assertLessEqual(similarity, 1.0, "相似度应该小于等于1")
        self.assertGreater(similarity, 0.7, "相似文本的相似度应该较高")
        
        # 计算不相似文本的相似度
        text3 = "Python是一种编程语言"
        similarity = self.embedding_manager.similarity(text1, text3)
        
        # 验证结果
        self.assertLess(similarity, 0.7, "不相似文本的相似度应该较低")
    
    def test_memory_cache(self):
        """测试内存缓存功能"""
        # 创建带缓存的嵌入管理器
        cache = MemoryEmbeddingCache(max_size=100)
        
        embedding_manager = LocalEmbedding(
            config=EmbeddingConfig(use_cache=True),
            cache=cache
        )
        embedding_manager.initialize()
        
        # 第一次嵌入（未缓存）
        start_time = time.time()
        embedding1 = embedding_manager.embed_text(self.test_texts[0])
        first_time = time.time() - start_time
        
        # 第二次嵌入（应使用缓存）
        start_time = time.time()
        embedding2 = embedding_manager.embed_text(self.test_texts[0])
        second_time = time.time() - start_time
        
        # 验证结果
        self.assertEqual(embedding1.text, embedding2.text, "缓存结果应该与首次嵌入一致")
        self.assertTrue(np.array_equal(embedding1.embedding, embedding2.embedding), 
                        "缓存结果的向量应该与首次嵌入一致")
        
        # 检查缓存命中信息
        cache_info = cache.get_cache_info()
        self.assertEqual(cache_info["hits"], 1, "应该有1次缓存命中")
        self.assertEqual(cache_info["misses"], 1, "应该有1次缓存未命中")
    
    def test_disk_cache(self):
        """测试磁盘缓存功能"""
        # 创建带磁盘缓存的嵌入管理器
        cache = DiskEmbeddingCache(cache_dir=self.temp_dir)
        
        embedding_manager = LocalEmbedding(
            config=EmbeddingConfig(use_cache=True),
            cache=cache
        )
        embedding_manager.initialize()
        
        # 嵌入所有测试文本
        for text in self.test_texts:
            embedding_manager.embed_text(text)
        
        # 验证缓存目录中的文件
        cache_files = list(Path(self.temp_dir).glob("*.npy"))
        self.assertEqual(len(cache_files), len(self.test_texts), 
                        "缓存目录应该包含与测试文本数量相同的文件")
        
        # 清空缓存
        cache.clear()
        
        # 验证缓存目录是否已清空
        cache_files = list(Path(self.temp_dir).glob("*.npy"))
        self.assertEqual(len(cache_files), 0, "缓存目录应该为空")
    
    def test_batch_with_cache(self):
        """测试带缓存的批量嵌入功能"""
        # 创建带缓存的嵌入管理器
        cache = MemoryEmbeddingCache(max_size=100)
        
        embedding_manager = LocalEmbedding(
            config=EmbeddingConfig(use_cache=True),
            cache=cache
        )
        embedding_manager.initialize()
        
        # 第一次批量嵌入
        embeddings1 = embedding_manager.embed_texts(self.test_texts)
        
        # 部分修改测试文本
        mixed_texts = self.test_texts.copy()
        mixed_texts[1] = "这是一个新的测试文本"  # 替换一个文本
        
        # 第二次批量嵌入（部分应使用缓存）
        embeddings2 = embedding_manager.embed_texts(mixed_texts)
        
        # 验证结果
        self.assertEqual(embeddings1[0].text, embeddings2[0].text, "第一个文本应该相同")
        self.assertEqual(embeddings1[2].text, embeddings2[2].text, "第三个文本应该相同")
        self.assertNotEqual(embeddings1[1].text, embeddings2[1].text, "第二个文本应该不同")
        
        # 检查缓存命中信息
        cache_info = cache.get_cache_info()
        self.assertEqual(cache_info["hits"], 2, "应该有2次缓存命中（第1和第3个文本）")
        self.assertEqual(cache_info["misses"], 4, "应该有4次缓存未命中（首次3个文本和第二次的新文本）")


if __name__ == "__main__":
    unittest.main()
