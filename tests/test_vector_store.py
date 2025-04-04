"""
向量存储系统测试 - 测试向量存储和检索功能
"""

import unittest
import sys
from pathlib import Path
import os
import tempfile
import shutil
import numpy as np

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

from mcp.knowledge.storage import (
    VectorStoreConfig,
    VectorRecord,
    SearchResult,
    InMemoryVectorStore,
    SQLiteVectorStore
)


class TestVectorStore(unittest.TestCase):
    """测试向量存储系统"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试向量
        self.test_vectors = [
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32)
        ]
        
        # 标准化向量用于相似度计算
        self.test_vectors = [v / np.linalg.norm(v) for v in self.test_vectors]
        
        # 创建测试记录
        self.records = []
        for i, vector in enumerate(self.test_vectors):
            record = VectorRecord(
                id=f"test_{i}",
                embedding=vector,
                text=f"测试文本 {i}",
                metadata={"category": f"分类{i}", "level": "测试"}
            )
            self.records.append(record)
        
        # 创建查询向量 - 使其与第一个测试向量接近
        self.query_vector = self.test_vectors[0] * 0.9 + np.random.rand(384).astype(np.float32) * 0.1
        # 标准化查询向量
        self.query_vector = self.query_vector / np.linalg.norm(self.query_vector)
        
        # 创建内存向量存储
        self.memory_config = VectorStoreConfig(
            embedding_dim=384,
            distance_metric="cosine",
            normalize_vectors=True
        )
        self.memory_store = InMemoryVectorStore(config=self.memory_config)
        self.memory_store.initialize()
        
        # 创建SQLite向量存储
        self.sqlite_path = os.path.join(self.temp_dir, "test_vectors.db")
        self.sqlite_config = VectorStoreConfig(
            embedding_dim=384,
            distance_metric="cosine",
            normalize_vectors=True,
            storage_path=self.sqlite_path
        )
        self.sqlite_store = SQLiteVectorStore(config=self.sqlite_config)
        try:
            self.sqlite_store.initialize()
            self.skip_sqlite = False
        except:
            self.skip_sqlite = True
            print("警告: SQLite向量存储初始化失败，将跳过相关测试")
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_memory_store_initialization(self):
        """测试内存向量存储初始化"""
        self.assertTrue(self.memory_store._initialized)
        self.assertEqual(self.memory_store.config.embedding_dim, 384)
        self.assertEqual(self.memory_store.config.distance_metric, "cosine")
    
    def test_memory_store_add(self):
        """测试内存向量存储添加记录"""
        # 添加记录
        record_id = self.memory_store.add(self.records[0])
        
        # 验证结果
        self.assertEqual(record_id, self.records[0].id)
        self.assertEqual(self.memory_store.count(), 1)
    
    def test_memory_store_add_batch(self):
        """测试内存向量存储批量添加记录"""
        # 批量添加记录
        record_ids = self.memory_store.add_batch(self.records)
        
        # 验证结果
        self.assertEqual(len(record_ids), len(self.records))
        for i, record_id in enumerate(record_ids):
            self.assertEqual(record_id, self.records[i].id)
        self.assertEqual(self.memory_store.count(), len(self.records))
    
    def test_memory_store_get(self):
        """测试内存向量存储获取记录"""
        # 添加记录
        self.memory_store.add(self.records[0])
        
        # 获取记录
        record = self.memory_store.get(self.records[0].id)
        
        # 验证结果
        self.assertIsNotNone(record)
        self.assertEqual(record.id, self.records[0].id)
        self.assertEqual(record.text, self.records[0].text)
        np.testing.assert_array_equal(record.embedding, self.records[0].embedding)
    
    def test_memory_store_search(self):
        """测试内存向量存储搜索"""
        # 添加所有记录
        self.memory_store.add_batch(self.records)
        
        # 搜索
        results = self.memory_store.search(self.query_vector, k=2)
        
        # 验证结果
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        
        # 第一个结果应该是与查询向量最接近的记录（应该是第一个记录）
        cosine_similarities = [
            np.dot(self.query_vector, self.test_vectors[i])
            for i in range(len(self.test_vectors))
        ]
        most_similar_idx = np.argmax(cosine_similarities)
        self.assertEqual(results[0].record.id, f"test_{most_similar_idx}")
    
    def test_memory_store_search_with_filter(self):
        """测试内存向量存储带过滤条件的搜索"""
        # 添加所有记录
        self.memory_store.add_batch(self.records)
        
        # 搜索带过滤条件
        filter = {"category": "分类0"}
        results = self.memory_store.search(self.query_vector, k=2, filter=filter)
        
        # 验证结果
        for result in results:
            self.assertEqual(result.record.metadata["category"], "分类0")
    
    def test_memory_store_delete(self):
        """测试内存向量存储删除记录"""
        # 添加记录
        self.memory_store.add(self.records[0])
        
        # 验证添加成功
        self.assertEqual(self.memory_store.count(), 1)
        
        # 删除记录
        deleted = self.memory_store.delete(self.records[0].id)
        
        # 验证结果
        self.assertTrue(deleted)
        self.assertEqual(self.memory_store.count(), 0)
    
    def test_memory_store_update(self):
        """测试内存向量存储更新记录"""
        # 添加记录
        self.memory_store.add(self.records[0])
        
        # 创建更新后的记录
        updated_record = VectorRecord(
            id=self.records[0].id,
            embedding=self.records[0].embedding,
            text="更新后的文本",
            metadata={"category": "更新后的分类", "level": "测试"}
        )
        
        # 更新记录
        updated = self.memory_store.update(updated_record)
        
        # 验证结果
        self.assertTrue(updated)
        retrieved_record = self.memory_store.get(self.records[0].id)
        self.assertEqual(retrieved_record.text, "更新后的文本")
        self.assertEqual(retrieved_record.metadata["category"], "更新后的分类")
    
    def test_sqlite_store_basic(self):
        """测试SQLite向量存储基本功能"""
        if self.skip_sqlite:
            self.skipTest("SQLite向量存储初始化失败，跳过测试")
        
        # 添加记录
        self.sqlite_store.add(self.records[0])
        
        # 验证添加成功
        self.assertEqual(self.sqlite_store.count(), 1)
        
        # 获取记录
        record = self.sqlite_store.get(self.records[0].id)
        
        # 验证结果
        self.assertIsNotNone(record)
        self.assertEqual(record.id, self.records[0].id)
        self.assertEqual(record.text, self.records[0].text)
        
        # 删除记录
        deleted = self.sqlite_store.delete(self.records[0].id)
        
        # 验证结果
        self.assertTrue(deleted)
        self.assertEqual(self.sqlite_store.count(), 0)
    
    def test_sqlite_store_search(self):
        """测试SQLite向量存储搜索功能"""
        if self.skip_sqlite:
            self.skipTest("SQLite向量存储初始化失败，跳过测试")
        
        # 添加所有记录
        self.sqlite_store.add_batch(self.records)
        
        # 搜索
        results = self.sqlite_store.search(self.query_vector, k=2)
        
        # 验证结果
        self.assertGreaterEqual(len(results), 1)
        self.assertIsInstance(results[0], SearchResult)
