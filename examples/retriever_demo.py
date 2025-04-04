"""
知识检索演示脚本

此脚本演示如何使用MCP知识增强层的知识检索功能，包括:
1. 使用SimilarityRetriever进行向量相似度检索
2. 使用HybridRetriever结合向量和关键词检索
3. 使用元数据过滤提高相关性
4. 多种检索策略的比较

用法:
python retriever_demo.py
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from pprint import pprint
import re
import logging
from typing import List

# 设置日志级别
logging.basicConfig(level=logging.INFO)

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

from mcp.knowledge.retrieval import (
    RetrievalConfig,
    SimilarityRetriever,
    HybridRetriever,
    KeywordRetriever,
    EnsembleRetriever,
    ReRankingRetriever,
    RetrievalStrategy
)


# 示例文档
DOCUMENTS = [
    {
        "title": "人工智能概述",
        "text": "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、问题解决、感知和语言理解。人工智能系统通常利用大量数据进行训练，可以识别模式、做出决策和预测未来行为。",
        "metadata": {"category": "AI", "level": "introductory", "type": "overview", "year": 2023}
    },
    {
        "title": "机器学习基础",
        "text": "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统，而无需被明确编程。常见的机器学习方法包括监督学习、无监督学习和强化学习。监督学习使用标记数据训练模型，无监督学习寻找数据中的隐藏模式，而强化学习则通过奖励系统学习最佳行动。",
        "metadata": {"category": "AI", "subcategory": "machine_learning", "level": "introductory", "type": "overview", "year": 2022}
    },
    {
        "title": "深度学习技术",
        "text": "深度学习是机器学习的一种特殊方法，它使用多层神经网络来学习复杂模式。神经网络的灵感来自人类大脑的结构，由相互连接的节点（神经元）组成。深度学习在图像识别、自然语言处理和游戏中取得了显著成功。最流行的深度学习架构包括卷积神经网络和循环神经网络。",
        "metadata": {"category": "AI", "subcategory": "deep_learning", "level": "intermediate", "type": "technical", "year": 2022}
    },
    {
        "title": "自然语言处理应用",
        "text": "自然语言处理（NLP）是人工智能的一个分支，它关注计算机理解和生成人类语言的能力。NLP应用包括机器翻译、情感分析、文本摘要和聊天机器人。现代NLP系统通常基于Transformer架构，如GPT（生成式预训练Transformer）和BERT（来自Transformers的双向编码器表示）。",
        "metadata": {"category": "AI", "subcategory": "nlp", "level": "intermediate", "type": "application", "year": 2023}
    },
    {
        "title": "计算机视觉技术",
        "text": "计算机视觉是人工智能的一个领域，它使计算机能够理解和解释视觉信息。计算机视觉系统可以检测对象、识别人脸、分析场景和跟踪运动。它们广泛应用于自动驾驶汽车、监控系统、医学成像和增强现实。卷积神经网络已成为计算机视觉中最成功的架构之一。",
        "metadata": {"category": "AI", "subcategory": "computer_vision", "level": "intermediate", "type": "technical", "year": 2023}
    },
    {
        "title": "Python编程入门",
        "text": "Python是一种广泛使用的高级编程语言，以其简洁和可读性而闻名。它是一种解释型语言，强调代码的可读性和简洁的语法，使程序员能够用更少的代码行表达概念。Python支持多种编程范式，包括过程式、面向对象和函数式编程。它拥有庞大的标准库和活跃的开发者社区。",
        "metadata": {"category": "programming", "language": "python", "level": "introductory", "type": "tutorial", "year": 2021}
    },
    {
        "title": "JavaScript网页开发",
        "text": "JavaScript是一种用于Web开发的脚本语言，它允许在网页中添加交互性。作为HTML和CSS的补充，JavaScript使开发者能够创建动态网页、处理用户输入和与Web API通信。现代JavaScript框架如React、Angular和Vue.js简化了复杂用户界面的开发过程。",
        "metadata": {"category": "programming", "language": "javascript", "level": "introductory", "type": "tutorial", "year": 2021}
    },
    {
        "title": "数据库管理系统",
        "text": "数据库是用于存储和管理数据的系统，可以高效地检索和更新信息。常见的数据库类型包括关系型数据库（如MySQL、PostgreSQL）和非关系型数据库（如MongoDB、Redis）。数据库管理系统（DBMS）提供了一个界面，用于创建、查询和管理数据库，同时确保数据完整性和安全性。",
        "metadata": {"category": "database", "level": "introductory", "type": "overview", "year": 2020}
    },
    {
        "title": "云计算服务",
        "text": "云计算是通过互联网提供计算资源（如服务器、存储、数据库、网络、软件）的模式。云服务提供商（如AWS、Microsoft Azure和Google Cloud）管理底层基础设施，使用户能够根据需求扩展资源并仅为所用资源付费。云计算的主要模式包括基础设施即服务（IaaS）、平台即服务（PaaS）和软件即服务（SaaS）。",
        "metadata": {"category": "cloud", "level": "introductory", "type": "overview", "year": 2022}
    },
    {
        "title": "区块链基础",
        "text": "区块链是一种分布式账本技术，它维护着一个不断增长的记录链，这些记录通过密码学相互链接并保护。区块链的关键特性包括去中心化、不可变性和透明性。比特币是第一个基于区块链的应用，但这项技术已扩展到金融服务之外，包括供应链、医疗保健和投票系统。",
        "metadata": {"category": "blockchain", "level": "introductory", "type": "overview", "year": 2021}
    }
]


def create_simple_reranker(query_terms_weight=2.0):
    """
    创建一个简单的基于关键词的重排序函数
    
    Args:
        query_terms_weight: 查询词在文档中出现时的权重
        
    Returns:
        重排序函数
    """
    def reranker(query, docs):
        # 提取查询关键词
        query_terms = re.findall(r'\b\w+\b', query.lower())
        query_terms = [term for term in query_terms if len(term) > 2]
        
        scores = []
        for doc in docs:
            doc_lower = doc.lower()
            
            # 基本分数（保留原始排序的影响）
            score = 0.5
            
            # 检查查询词在文档中的出现
            for term in query_terms:
                if term in doc_lower:
                    score += query_terms_weight / len(query_terms)
            
            # 文档长度惩罚（短文档更有可能是精确匹配）
            doc_length = len(doc.split())
            length_factor = min(1.0, 100 / max(50, doc_length))
            score *= (0.5 + 0.5 * length_factor)
            
            scores.append(score)
            
        # 归一化分数
        max_score = max(scores) if scores else 1.0
        normalized_scores = [s / max_score for s in scores]
        
        return normalized_scores
        
    return reranker


def setup_data(embedding_manager):
    """
    设置数据：创建向量存储并添加文档
    
    Args:
        embedding_manager: 嵌入管理器
        
    Returns:
        向量存储
    """
    # 创建向量存储
    vector_config = VectorStoreConfig(
        embedding_dim=384,
        distance_metric="cosine",
        normalize_vectors=True
    )
    
    vector_store = InMemoryVectorStore(config=vector_config)
    vector_store.initialize()
    
    # 生成嵌入并添加文档
    records = []
    for i, doc in enumerate(DOCUMENTS):
        # 生成嵌入
        text = f"{doc['title']}\n\n{doc['text']}"
        embedding = embedding_manager.embed_text(text).embedding
        
        # 创建向量记录
        record = VectorRecord(
            id=f"doc_{i}",
            embedding=embedding,
            text=text,
            metadata=doc["metadata"]
        )
        
        records.append(record)
    
    # 批量添加记录
    vector_store.add_batch(records)
    
    return vector_store


def main():
    """主函数"""
    print("知识检索演示")
    print("-" * 50)
    
    # 创建嵌入管理器
    embedding_config = EmbeddingConfig(
        model_name="paraphrase-MiniLM-L6-v2",
        embedding_dim=384
    )
    
    print("初始化嵌入管理器...")
    embedding_manager = LocalEmbedding(config=embedding_config)
    embedding_manager.initialize()
    
    # 设置数据
    print("\n设置示例数据...")
    vector_store = setup_data(embedding_manager)
    print(f"已添加 {vector_store.count()} 个文档")
    
    # ========== 相似度检索演示 ==========
    print("\n相似度检索演示")
    print("-" * 30)
    
    # 创建相似度检索器
    similarity_config = RetrievalConfig(
        strategy=RetrievalStrategy.SIMILARITY,
        top_k=3,
        deduplicate=True
    )
    
    similarity_retriever = SimilarityRetriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        config=similarity_config
    )
    
    similarity_retriever.initialize()
    
    # 示例查询
    query = "人工智能和深度学习的应用"
    print(f"查询: '{query}'")
    
    # 检索
    start_time = time.time()
    results = similarity_retriever.retrieve(query)
    retrieval_time = time.time() - start_time
    
    # 显示结果
    print(f"\n检索结果 (耗时: {retrieval_time:.4f}秒):")
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  标题: {result.text.splitlines()[0]}")
        print(f"  相似度: {result.score:.4f}")
        print(f"  类别: {result.metadata.get('category')}")
        print(f"  年份: {result.metadata.get('year')}")
        print()
    
    # ========== 元数据过滤演示 ==========
    print("\n元数据过滤演示")
    print("-" * 30)
    
    # 使用过滤条件
    filter = {"level": "intermediate"}
    print(f"查询: '{query}' + 过滤: {filter}")
    
    # 检索
    results = similarity_retriever.retrieve_with_filter(query, filter)
    
    # 显示结果
    print("\n检索结果:")
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  标题: {result.text.splitlines()[0]}")
        print(f"  相似度: {result.score:.4f}")
        print(f"  级别: {result.metadata.get('level')}")
        print(f"  类别: {result.metadata.get('category')}")
        print()
    
    # ========== 混合检索演示 ==========
    print("\n混合检索演示")
    print("-" * 30)
    
    # 创建混合检索器
    hybrid_config = RetrievalConfig(
        strategy=RetrievalStrategy.HYBRID,
        top_k=3,
        hybrid_weight=0.7,  # 向量相似度权重
        deduplicate=True
    )
    
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        config=hybrid_config
    )
    
    hybrid_retriever.initialize()
    
    # 示例查询
    query = "深度学习和神经网络技术"
    print(f"查询: '{query}'")
    
    # 检索
    start_time = time.time()
    results = hybrid_retriever.retrieve(query)
    retrieval_time = time.time() - start_time
    
    # 显示结果
    print(f"\n检索结果 (耗时: {retrieval_time:.4f}秒):")
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  标题: {result.text.splitlines()[0]}")
        print(f"  相似度: {result.score:.4f}")
        print(f"  类别: {result.metadata.get('category')}")
        print(f"  来源: {result.source}")
        print()
    
    # ========== 重排序检索演示 ==========
    print("\n重排序检索演示")
    print("-" * 30)
    
    # 创建重排序检索器
    reranking_config = RetrievalConfig(
        strategy=RetrievalStrategy.RERANKING,
        top_k=3,
        deduplicate=True
    )
    
    # 创建简单的重排序函数
    reranker = create_simple_reranker(query_terms_weight=2.0)
    
    reranking_retriever = ReRankingRetriever(
        base_retriever=similarity_retriever,
        reranker=reranker,
        config=reranking_config
    )
    
    reranking_retriever.initialize()
    
    # 示例查询
    query = "Python编程和数据库系统"
    print(f"查询: '{query}'")
    
    # 检索
    results = reranking_retriever.retrieve(query)
    
    # 显示结果
    print("\n重排序结果:")
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  标题: {result.text.splitlines()[0]}")
        print(f"  相似度: {result.score:.4f}")
        print(f"  类别: {result.metadata.get('category')}")
        print()
    
    # ========== 检索器比较 ==========
    print("\n检索器比较")
    print("-" * 30)
    
    # 多个检索器
    retrievers = [
        ("相似度检索", similarity_retriever),
        ("混合检索", hybrid_retriever),
        ("重排序检索", reranking_retriever)
    ]
    
    # 示例查询
    query = "编程语言和云计算技术"
    print(f"查询: '{query}'")
    
    # 比较结果
    print("\n检索结果比较:")
    for name, retriever in retrievers:
        print(f"\n{name}:")
        results = retriever.retrieve(query)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.text.splitlines()[0]} ({result.score:.4f})")
    
    print("\n演示完成!")


if __name__ == "__main__":
    main()
