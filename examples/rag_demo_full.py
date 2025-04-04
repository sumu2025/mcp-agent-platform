"""
RAG完整流程演示脚本

此脚本演示如何使用MCP知识增强层的完整RAG流程，包括:
1. 文档处理与向量化
2. 检索相关知识
3. 上下文增强
4. 生成回答

用法:
python rag_demo_full.py [query]
"""

import sys
import os
import time
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Any

# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 确保可以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入所需组件
from mcp.knowledge.indexing import (
    DocumentProcessor,
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
    InMemoryVectorStore
)

from mcp.knowledge.retrieval import (
    RetrievalConfig,
    HybridRetriever,
    RetrievalStrategy
)

from mcp.knowledge.augmentation import (
    AugmentationConfig,
    AugmentationMode,
    StructuredAugmenter
)

from mcp.knowledge.rag_engine import (
    RAGEngine,
    RAGConfig,
    RAGStrategy
)

# 模拟LLM客户端
class MockLLMClient:
    """模拟LLM客户端，用于演示"""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """模拟生成回答"""
        print("\n============ LLM提示 ============")
        print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        print("=================================\n")
        
        # 根据提示生成简单回答
        return (
            "基于提供的参考资料，我可以回答您的问题。\n\n"
            "参考资料中提到...[此处是模拟的LLM回答]\n\n"
            "希望这个回答对您有所帮助！"
        )


# 示例文档
DOCUMENTS = [
    {
        "title": "人工智能概述",
        "content": "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、问题解决、感知和语言理解。人工智能系统通常利用大量数据进行训练，可以识别模式、做出决策和预测未来行为。",
        "metadata": {"category": "AI", "level": "introductory"}
    },
    {
        "title": "机器学习基础",
        "content": "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统，而无需被明确编程。常见的机器学习方法包括监督学习、无监督学习和强化学习。监督学习使用标记数据训练模型，无监督学习寻找数据中的隐藏模式，而强化学习则通过奖励系统学习最佳行动。",
        "metadata": {"category": "AI", "subcategory": "machine_learning", "level": "introductory"}
    },
    {
        "title": "深度学习技术",
        "content": "深度学习是机器学习的一种特殊方法，它使用多层神经网络来学习复杂模式。神经网络的灵感来自人类大脑的结构，由相互连接的节点（神经元）组成。深度学习在图像识别、自然语言处理和游戏中取得了显著成功。最流行的深度学习架构包括卷积神经网络和循环神经网络。",
        "metadata": {"category": "AI", "subcategory": "deep_learning", "level": "intermediate"}
    },
    {
        "title": "自然语言处理应用",
        "content": "自然语言处理（NLP）是人工智能的一个分支，它关注计算机理解和生成人类语言的能力。NLP应用包括机器翻译、情感分析、文本摘要和聊天机器人。现代NLP系统通常基于Transformer架构，如GPT（生成式预训练Transformer）和BERT（来自Transformers的双向编码器表示）。",
        "metadata": {"category": "AI", "subcategory": "nlp", "level": "intermediate"}
    },
    {
        "title": "计算机视觉技术",
        "content": "计算机视觉是人工智能的一个领域，它使计算机能够理解和解释视觉信息。计算机视觉系统可以检测对象、识别人脸、分析场景和跟踪运动。它们广泛应用于自动驾驶汽车、监控系统、医学成像和增强现实。卷积神经网络已成为计算机视觉中最成功的架构之一。",
        "metadata": {"category": "AI", "subcategory": "computer_vision", "level": "intermediate"}
    }
]


def setup_documents(temp_dir: str) -> List[str]:
    """
    设置临时文档
    
    Args:
        temp_dir: 临时目录
        
    Returns:
        文档路径列表
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    # 创建临时Markdown文件
    file_paths = []
    for doc in DOCUMENTS:
        file_name = f"{doc['title'].replace(' ', '_')}.md"
        file_path = os.path.join(temp_dir, file_name)
        
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
            
        file_paths.append(file_path)
        
    return file_paths


def process_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    处理文档
    
    Args:
        file_paths: 文档路径列表
        
    Returns:
        处理后的文档列表
    """
    # 创建文档处理器
    chunker = RecursiveTextChunker(
        chunk_size=512,       # 目标块大小（字符数）
        chunk_overlap=100,    # 块重叠（字符数）
        min_chunk_size=50     # 最小块大小（字符数）
    )
    
    metadata_extractor = ObsidianMetadataExtractor()
    
    processor = MarkdownProcessor(
        chunker=chunker,
        metadata_extractor=metadata_extractor
    )
    
    # 处理文档
    processed_docs = []
    
    for file_path in file_paths:
        print(f"处理文档: {file_path}")
        
        # 提取元数据
        metadata = processor.extract_metadata(file_path)
        
        # 处理文档
        text_chunks = processor.process(file_path)
        
        for chunk in text_chunks:
            processed_docs.append({
                "text": chunk.text,
                "metadata": chunk.metadata
            })
    
    return processed_docs


def setup_rag_system(docs: List[Dict[str, Any]]) -> RAGEngine:
    """
    设置RAG系统
    
    Args:
        docs: 处理后的文档列表
        
    Returns:
        RAG引擎
    """
    # 1. 创建嵌入管理器
    embedding_config = EmbeddingConfig(
        model_name="paraphrase-MiniLM-L6-v2",
        embedding_dim=384
    )
    
    embedding_manager = LocalEmbedding(config=embedding_config)
    embedding_manager.initialize()
    
    # 2. 创建向量存储
    vector_config = VectorStoreConfig(
        embedding_dim=384,
        distance_metric="cosine",
        normalize_vectors=True
    )
    
    vector_store = InMemoryVectorStore(config=vector_config)
    vector_store.initialize()
    
    # 3. 添加文档到向量存储
    for doc in docs:
        # 生成嵌入
        embedding = embedding_manager.embed_text(doc["text"]).embedding
        
        # 创建向量记录
        record = {
            "id": f"doc_{len(vector_store.records)}",
            "embedding": embedding,
            "text": doc["text"],
            "metadata": doc["metadata"]
        }
        
        # 添加到向量存储
        vector_store.add(record)
    
    print(f"已添加 {vector_store.count()} 个文档块到向量存储")
    
    # 4. 创建检索配置
    retrieval_config = RetrievalConfig(
        strategy=RetrievalStrategy.HYBRID,
        top_k=3,
        hybrid_weight=0.7,
        deduplicate=True
    )
    
    # 5. 创建增强配置
    augmentation_config = AugmentationConfig(
        mode=AugmentationMode.STRUCTURED,
        max_context_length=4000,
        include_metadata=True
    )
    
    # 6. 创建RAG配置
    rag_config = RAGConfig(
        strategy=RAGStrategy.ADVANCED,
        retrieval_config=retrieval_config,
        augmentation_config=augmentation_config,
        enable_logging=True
    )
    
    # 7. 创建模拟LLM客户端
    llm_client = MockLLMClient()
    
    # 8. 创建RAG引擎
    rag_engine = RAGEngine(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        llm_client=llm_client,
        config=rag_config
    )
    
    rag_engine.initialize()
    
    return rag_engine


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='RAG完整流程演示')
    parser.add_argument('query', nargs='?', default="什么是深度学习？它与机器学习有什么区别？", help='查询文本')
    args = parser.parse_args()
    
    print("RAG完整流程演示")
    print("-" * 50)
    
    # 设置临时目录
    temp_dir = "./temp_docs"
    
    # 创建临时文档
    file_paths = setup_documents(temp_dir)
    
    # 处理文档
    processed_docs = process_documents(file_paths)
    
    # 设置RAG系统
    rag_engine = setup_rag_system(processed_docs)
    
    # 用户查询
    query = args.query
    print(f"\n用户查询: {query}")
    
    # 生成回答
    start_time = time.time()
    result = rag_engine.generate(query)
    end_time = time.time()
    
    # 显示结果
    print(f"\n处理时间: {end_time - start_time:.2f}秒")
    print(f"\n检索到 {len(result.retrieval_results)} 个相关文档片段")
    
    print("\n--- 检索结果预览 ---")
    for i, res in enumerate(result.retrieval_results):
        print(f"{i+1}. {res.text[:100]}... (相似度: {res.score:.2f})")
    
    print("\n--- 生成回答 ---")
    print(result.answer)
    
    # 清理临时文件
    if os.path.exists(temp_dir):
        pass  # 保留文件以供检查，实际应用中可以删除
        # import shutil
        # shutil.rmtree(temp_dir)
    
    print("\n演示完成!")


if __name__ == "__main__":
    main()
