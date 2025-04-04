"""
向量存储演示脚本

此脚本演示如何使用MCP知识增强层的向量存储功能，包括:
1. 使用InMemoryVectorStore进行内存存储
2. 使用SQLiteVectorStore进行持久化存储
3. 向量相似度搜索
4. 元数据过滤

用法:
python vector_store_demo.py
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from pprint import pprint

# 确保可以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

from mcp.knowledge.storage import (
    VectorStoreConfig,
    VectorRecord,
    InMemoryVectorStore,
    SQLiteVectorStore
)

from mcp.knowledge.embedding import (
    EmbeddingConfig,
    LocalEmbedding
)


def main():
    """主函数"""
    print("向量存储演示")
    print("-" * 50)
    
    # 创建嵌入管理器
    embedding_config = EmbeddingConfig(
        model_name="paraphrase-MiniLM-L6-v2",
        embedding_dim=384
    )
    
    print("初始化嵌入管理器...")
    embedding_manager = LocalEmbedding(config=embedding_config)
    embedding_manager.initialize()
    
    # 示例文本
    texts = [
        {"text": "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。", 
         "metadata": {"category": "AI", "level": "introductory"}},
        {"text": "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统。", 
         "metadata": {"category": "AI", "subcategory": "machine_learning", "level": "introductory"}},
        {"text": "深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。", 
         "metadata": {"category": "AI", "subcategory": "deep_learning", "level": "intermediate"}},
        {"text": "自然语言处理是人工智能的一个分支，它关注计算机理解和生成人类语言的能力。", 
         "metadata": {"category": "AI", "subcategory": "nlp", "level": "intermediate"}},
        {"text": "计算机视觉是人工智能的一个领域，它关注使计算机能够理解和解释视觉信息。", 
         "metadata": {"category": "AI", "subcategory": "computer_vision", "level": "intermediate"}},
        {"text": "Python是一种广泛使用的高级编程语言，以其简洁和可读性而闻名。", 
         "metadata": {"category": "programming", "language": "python", "level": "introductory"}},
        {"text": "JavaScript是一种用于Web开发的脚本语言，它允许在网页中添加交互性。", 
         "metadata": {"category": "programming", "language": "javascript", "level": "introductory"}},
        {"text": "数据库是用于存储和管理数据的系统，可以高效地检索和更新信息。", 
         "metadata": {"category": "database", "level": "introductory"}}
    ]
    
    # 生成嵌入
    print("\n生成嵌入向量...")
    records = []
    
    for i, item in enumerate(texts):
        print(f"处理文本 {i+1}/{len(texts)}")
        embedding = embedding_manager.embed_text(item["text"]).embedding
        
        record = VectorRecord(
            id=f"record_{i}",
            embedding=embedding,
            text=item["text"],
            metadata=item["metadata"]
        )
        
        records.append(record)
    
    # ========== 内存向量存储演示 ==========
    print("\n内存向量存储演示")
    print("-" * 30)
    
    # 创建内存向量存储
    memory_config = VectorStoreConfig(
        embedding_dim=384,
        distance_metric="cosine",
        normalize_vectors=True
    )
    
    memory_store = InMemoryVectorStore(config=memory_config)
    memory_store.initialize()
    
    # 添加向量记录
    print("添加向量记录...")
    for record in records:
        memory_store.add(record)
    
    print(f"已存储 {memory_store.count()} 条记录")
    
    # 获取存储信息
    store_info = memory_store.get_store_info()
    print("\n存储信息:")
    pprint(store_info)
    
    # 相似度搜索演示
    print("\n相似度搜索演示:")
    
    # 示例查询
    query_text = "AI技术进展"
    print(f"查询: '{query_text}'")
    
    query_embedding = embedding_manager.embed_query(query_text)
    
    # 搜索相似向量
    results = memory_store.search(query_embedding, k=3)
    
    # 显示结果
    print("\n搜索结果:")
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  文本: {result.record.text}")
        print(f"  相似度: {result.score:.4f}")
        print(f"  类别: {result.record.metadata.get('category')}")
        print()
    
    # 元数据过滤演示
    print("\n元数据过滤演示:")
    
    # 使用过滤条件搜索
    filter = {"level": "intermediate"}
    print(f"查询: '{query_text}' + 过滤: {filter}")
    
    results = memory_store.search(query_embedding, k=3, filter=filter)
    
    # 显示结果
    print("\n搜索结果:")
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  文本: {result.record.text}")
        print(f"  相似度: {result.score:.4f}")
        print(f"  类别: {result.record.metadata.get('category')}")
        print(f"  级别: {result.record.metadata.get('level')}")
        print()
    
    # ========== SQLite向量存储演示 ==========
    print("\nSQLite向量存储演示")
    print("-" * 30)
    
    # 创建SQLite向量存储
    db_path = os.path.join(os.path.expanduser("~"), ".mcp", "vector_store", "demo.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    sqlite_config = VectorStoreConfig(
        embedding_dim=384,
        distance_metric="cosine",
        normalize_vectors=True,
        storage_path=db_path
    )
    
    sqlite_store = SQLiteVectorStore(config=sqlite_config)
    sqlite_store.initialize()
    
    # 清空存储（确保演示从干净的状态开始）
    sqlite_store.clear()
    
    # 添加向量记录
    print("添加向量记录...")
    sqlite_store.add_batch(records)
    
    print(f"已存储 {sqlite_store.count()} 条记录")
    
    # 创建元数据索引
    print("创建元数据索引...")
    sqlite_store.add_metadata_index("category")
    sqlite_store.add_metadata_index("level")
    
    # 获取存储信息
    store_info = sqlite_store.get_store_info()
    print("\n存储信息:")
    pprint(store_info)
    
    # 相似度搜索演示
    print("\n相似度搜索示例:")
    
    # 搜索相似向量
    start_time = time.time()
    results = sqlite_store.search(query_embedding, k=3)
    search_time = time.time() - start_time
    
    # 显示结果
    print(f"\n搜索结果 (耗时: {search_time:.4f}秒):")
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  文本: {result.record.text}")
        print(f"  相似度: {result.score:.4f}")
        print(f"  类别: {result.record.metadata.get('category')}")
        print()
    
    # 复杂过滤演示
    print("\n复杂过滤示例:")
    
    # 使用多个过滤条件
    filter = {"category": "programming"}
    print(f"查询: '{query_text}' + 过滤: {filter}")
    
    results = sqlite_store.search(query_embedding, k=2, filter=filter)
    
    # 显示结果
    print("\n搜索结果:")
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  文本: {result.record.text}")
        print(f"  相似度: {result.score:.4f}")
        print(f"  类别: {result.record.metadata.get('category')}")
        print(f"  语言: {result.record.metadata.get('language')}")
        print()
    
    # 更新记录演示
    print("\n更新记录演示:")
    
    # 获取一个现有记录
    record = sqlite_store.get("record_0")
    
    if record:
        # 更新记录
        print(f"原始文本: {record.text}")
        print(f"原始元数据: {record.metadata}")
        
        # 修改记录
        record.metadata["importance"] = "high"
        record.metadata["updated"] = True
        
        # 更新存储
        sqlite_store.update(record)
        
        # 验证更新
        updated_record = sqlite_store.get("record_0")
        print(f"\n更新后元数据: {updated_record.metadata}")
    
    # 删除记录演示
    print("\n删除记录演示:")
    
    # 删除一个记录
    delete_id = "record_7"
    print(f"删除记录 ID: {delete_id}")
    
    if sqlite_store.delete(delete_id):
        print(f"记录 {delete_id} 已删除")
    
    # 验证删除
    print(f"当前记录数: {sqlite_store.count()}")
    
    # 优化存储
    print("\n优化存储...")
    sqlite_store.optimize()
    
    # 备份存储
    print("\n备份存储...")
    backup_path = sqlite_store.backup()
    print(f"备份已创建: {backup_path}")
    
    print("\n演示完成!")


if __name__ == "__main__":
    main()
