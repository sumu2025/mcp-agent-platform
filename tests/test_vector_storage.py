"""
向量存储系统测试
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import numpy as np

# 导入测试配置
from test_config import TEST_EMBEDDING_DIM, logger, setup_basic_rag_components

# 导入要测试的模块
from mcp.knowledge.storage import (
    VectorStore,
    VectorStoreConfig,
    VectorRecord,
    SearchResult,
    InMemoryVectorStore,
    SQLiteVectorStore
)


class TestVectorStorage(unittest.TestCase):
    """测试向量存储系统"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 设置基本组件
        cls.embedding_manager, _ = setup_basic_rag_components()
        
        # 创建测试数据
        cls.test_records = []
        cls.test_texts = [
            "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统。",
            "深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。",
            "自然语言处理是人工智能的一个分支，它关注计算机理解和生成人类语言的能力。",
            "计算机视觉是人工智能的一个领域，它使计算机能够理解和解释视觉信息。"
        ]
        
        cls.test_metadata = [
            {"category": "AI", "level": "introductory"},
            {"category": "AI", "subcategory": "machine_learning", "level": "introductory"},
            {"category": "AI", "subcategory": "deep_learning", "level": "intermediate"},
            {"category": "AI", "subcategory": "nlp", "level": "intermediate"},
            {"category": "AI", "subcategory": "computer_vision", "level": "intermediate"}
        ]
        
        # 创建测试记录
        for i, (text, metadata) in enumerate(zip(cls.test_texts, cls.test_metadata)):
            # 生成嵌入
            embedding = cls.embedding_manager.embed_text(text).embedding
            
            # 创建向量记录
            record = VectorRecord(
                id=f"test_{i}",
                embedding=embedding,
                text=text,
                metadata=metadata
            )
            
            cls.test_records.append(record)
    
    def setUp(self):
        """每个测试前的设置"""
        # 创建内存向量存储
        self.memory_store = InMemoryVectorStore(config=VectorStoreConfig(
            embedding_dim=TEST_EMBEDDING_DIM,
            distance_metric="cosine"
        ))
        self.memory_store.initialize()
        
        # 添加测试记录
        for record in self.test_records:
            self.memory_store.add(record)
        
        # 创建临时SQLite向量存储
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        
        self.sqlite_store = SQLiteVectorStore(config=VectorStoreConfig(
            embedding_dim=TEST_EMBEDDING_DIM,
            distance_metric="cosine",
            storage_path=self.temp_db.name
        ))
        self.sqlite_store.initialize()
        
        # 添加测试记录
        for record in self.test_records:
            self.sqlite_store.add(record)
    
    def tearDown(self):
        """每个测试后的清理"""
        # 清理临时SQLite文件
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_store_initialization(self):
        """测试向量存储初始化"""
        self.assertIsInstance(self.memory_store, VectorStore)
        self.assertIsInstance(self.sqlite_store, VectorStore)
    
    def test_record_count(self):
        """测试记录数量"""
        self.assertEqual(self.memory_store.count(), len(self.test_records))
        self.assertEqual(self.sqlite_store.count(), len(self.test_records))
    
    def test_add_and_get(self):
        """测试添加和获取记录"""
        # 创建新记录
        new_text = "这是一个新的测试记录。"
        new_embedding = self.embedding_manager.embed_text(new_text).embedding
        new_record = VectorRecord(
            id="new_record",
            embedding=new_embedding,
            text=new_text,
            metadata={"test": True}
        )
        
        # 添加到内存存储
        self.memory_store.add(new_record)
        
        # 验证记录数量
        self.assertEqual(self.memory_store.count(), len(self.test_records) + 1)
        
        # 获取记录
        retrieved = self.memory_store.get("new_record")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, "new_record")
        self.assertEqual(retrieved.text, new_text)
        self.assertTrue(np.array_equal(retrieved.embedding, new_embedding))
        self.assertEqual(retrieved.metadata["test"], True)
    
    def test_batch_add(self):
        """测试批量添加记录"""
        # 创建新记录
        new_records = []
        for i in range(3):
            text = f"批量添加测试记录 {i}"
            embedding = self.embedding_manager.embed_text(text).embedding
            record = VectorRecord(
                id=f"batch_{i}",
                embedding=embedding,
                text=text,
                metadata={"batch": i}
            )
            new_records.append(record)
        
        # 批量添加到SQLite存储
        self.sqlite_store.add_batch(new_records)
        
        # 验证记录数量
        self.assertEqual(self.sqlite_store.count(), len(self.test_records) + len(new_records))
        
        # 验证记录内容
        for i in range(3):
            record_id = f"batch_{i}"
            retrieved = self.sqlite_store.get(record_id)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.id, record_id)
            self.assertEqual(retrieved.metadata["batch"], i)
    
    def test_delete(self):
        """测试删除记录"""
        # 删除内存存储中的记录
        record_id = self.test_records[0].id
        self.memory_store.delete(record_id)
        
        # 验证记录数量
        self.assertEqual(self.memory_store.count(), len(self.test_records) - 1)
        
        # 验证记录已删除
        self.assertIsNone(self.memory_store.get(record_id))
    
    def test_update(self):
        """测试更新记录"""
        # 获取记录
        record_id = self.test_records[1].id
        record = self.sqlite_store.get(record_id)
        
        # 修改记录
        updated_record = VectorRecord(
            id=record.id,
            embedding=record.embedding,
            text=record.text,
            metadata={"updated": True, **record.metadata}
        )
        
        # 更新记录
        self.sqlite_store.update(updated_record)
        
        # 验证更新
        retrieved = self.sqlite_store.get(record_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, record_id)
        self.assertTrue(retrieved.metadata["updated"])
    
    def test_search_similarity(self):
        """测试相似度搜索"""
        # 创建查询向量
        query_text = "机器学习和深度学习的区别是什么？"
        query_vector = self.embedding_manager.embed_query(query_text)
        
        # 内存存储搜索
        memory_results = self.memory_store.search(query_vector, k=2)
        
        # 验证结果
        self.assertEqual(len(memory_results), 2)
        self.assertIsInstance(memory_results[0], SearchResult)
        self.assertGreater(memory_results[0].score, 0.5)  # 相似度应该较高
        
        # 验证结果排序
        self.assertGreaterEqual(memory_results[0].score, memory_results[1].score)
        
        # SQLite存储搜索
        sqlite_results = self.sqlite_store.search(query_vector, k=2)
        
        # 验证结果
        self.assertEqual(len(sqlite_results), 2)
        self.assertIsInstance(sqlite_results[0], SearchResult)
    
    def test_search_with_filter(self):
        """测试带过滤条件的搜索"""
        # 创建查询向量
        query_text = "深度学习技术"
        query_vector = self.embedding_manager.embed_query(query_text)
        
        # 设置过滤条件
        filter_condition = {"level": "intermediate"}
        
        # 内存存储搜索
        memory_results = self.memory_store.search(
            query_vector, 
            k=3, 
            filter=filter_condition
        )
        
        # 验证结果
        self.assertGreater(len(memory_results), 0)
        for result in memory_results:
            self.assertEqual(result.record.metadata["level"], "intermediate")
    
    def test_clear(self):
        """测试清空存储"""
        # 清空内存存储
        self.memory_store.clear()
        
        # 验证记录数量
        self.assertEqual(self.memory_store.count(), 0)
    
    def test_save_load(self):
        """测试保存和加载"""
        # 保存SQLite存储
        self.sqlite_store.save()
        
        # 创建新的SQLite存储实例
        new_store = SQLiteVectorStore(config=VectorStoreConfig(
            embedding_dim=TEST_EMBEDDING_DIM,
            distance_metric="cosine",
            storage_path=self.temp_db.name
        ))
        
        # 加载数据
        new_store.load()
        
        # 验证记录数量
        self.assertEqual(new_store.count(), self.sqlite_store.count())
        
        # 验证搜索功能
        query_text = "人工智能技术"
        query_vector = self.embedding_manager.embed_query(query_text)
        
        results = new_store.search(query_vector, k=2)
        self.assertEqual(len(results), 2)
    
    def test_store_info(self):
        """测试存储信息"""
        # 获取存储信息
        memory_info = self.memory_store.get_store_info()
        sqlite_info = self.sqlite_store.get_store_info()
        
        # 验证信息内容
        self.assertEqual(memory_info["type"], "memory_vector_store")
        self.assertEqual(memory_info["embedding_dim"], TEST_EMBEDDING_DIM)
        
        self.assertEqual(sqlite_info["type"], "sqlite_vector_store")
        self.assertEqual(sqlite_info["embedding_dim"], TEST_EMBEDDING_DIM)


if __name__ == "__main__":
    unittest.main()
