"""
嵌入功能演示脚本

此脚本演示如何使用MCP知识增强层的嵌入功能，包括:
1. 使用LocalEmbedding生成文本嵌入
2. 使用EmbeddingCache缓存嵌入结果
3. 计算文本相似度

用法:
python embedding_demo.py
"""

import sys
import os
import time
from pathlib import Path
import numpy as np

# 确保可以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

from mcp.knowledge.embedding import (
    EmbeddingConfig,
    LocalEmbedding,
    DiskEmbeddingCache
)


def main():
    """主函数"""
    print("嵌入功能演示")
    print("-" * 50)
    
    # 创建缓存目录
    cache_dir = os.path.join(os.path.expanduser("~"), ".mcp", "embedding_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # 创建嵌入缓存
    cache = DiskEmbeddingCache(cache_dir=cache_dir)
    
    # 创建嵌入配置
    config = EmbeddingConfig(
        model_name="paraphrase-MiniLM-L6-v2",  # 小型嵌入模型
        embedding_dim=384,  # 嵌入维度
        batch_size=32,      # 批处理大小
        use_cache=True,     # 启用缓存
        cache_dir=cache_dir # 缓存目录
    )
    
    print(f"使用模型: {config.model_name}")
    print(f"嵌入维度: {config.embedding_dim}")
    print(f"缓存目录: {cache_dir}")
    
    # 创建嵌入管理器
    embedding_manager = LocalEmbedding(config=config, cache=cache)
    
    # 初始化
    print("\n初始化嵌入管理器...")
    start_time = time.time()
    embedding_manager.initialize()
    init_time = time.time() - start_time
    print(f"初始化完成，耗时: {init_time:.2f}秒")
    
    # 示例文本
    texts = [
        "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统。",
        "深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。",
        "自然语言处理是人工智能的一个分支，它关注计算机理解和生成人类语言的能力。",
        "计算机视觉是人工智能的一个领域，它关注使计算机能够理解和解释视觉信息。"
    ]
    
    # 生成嵌入
    print("\n生成嵌入...")
    start_time = time.time()
    embeddings = []
    
    for i, text in enumerate(texts, 1):
        print(f"处理文本 {i}/{len(texts)}")
        embedding = embedding_manager.embed_text(text)
        embeddings.append(embedding)
        print(f"  文本: {text[:50]}...")
        print(f"  嵌入形状: {embedding.embedding.shape}")
    
    embed_time = time.time() - start_time
    print(f"嵌入生成完成，耗时: {embed_time:.2f}秒")
    
    # 计算相似度矩阵
    print("\n计算相似度矩阵:")
    similarity_matrix = np.zeros((len(texts), len(texts)))
    
    for i in range(len(texts)):
        for j in range(len(texts)):
            similarity = embedding_manager.similarity(texts[i], texts[j])
            similarity_matrix[i][j] = similarity
    
    # 打印相似度矩阵
    print("\n相似度矩阵:")
    for i in range(len(texts)):
        row = " ".join([f"{similarity_matrix[i][j]:.2f}" for j in range(len(texts))])
        print(f"[{row}]")
    
    # 找出最相似的文本对
    max_similarity = 0
    max_pair = (0, 0)
    
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if similarity_matrix[i][j] > max_similarity:
                max_similarity = similarity_matrix[i][j]
                max_pair = (i, j)
    
    print(f"\n最相似的文本对 ({max_similarity:.2f}):")
    print(f"1: {texts[max_pair[0]]}")
    print(f"2: {texts[max_pair[1]]}")
    
    # 测试缓存效率
    print("\n测试缓存效率...")
    
    # 第一次查询（应该使用缓存）
    start_time = time.time()
    for text in texts:
        embedding_manager.embed_text(text)
    cached_time = time.time() - start_time
    
    # 缓存信息
    cache_info = cache.get_cache_info()
    print(f"\n缓存信息:")
    for key, value in cache_info.items():
        print(f"  {key}: {value}")
    
    print(f"\n首次嵌入时间: {embed_time:.2f}秒")
    print(f"使用缓存时间: {cached_time:.2f}秒")
    print(f"加速比: {embed_time/cached_time:.2f}倍")
    
    print("\n演示完成!")


if __name__ == "__main__":
    main()
