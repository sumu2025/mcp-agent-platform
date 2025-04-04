"""
RAG系统测试配置
定义测试使用的共享配置、测试数据和辅助函数
"""

import os
import sys
from pathlib import Path
import logging
import tempfile
import shutil

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_test")

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"
if not TEST_DATA_DIR.exists():
    TEST_DATA_DIR.mkdir(parents=True)

# 测试模型配置
TEST_EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
TEST_EMBEDDING_DIM = 384

# 测试文档内容
TEST_DOCUMENTS = [
    {
        "title": "人工智能概述",
        "content": "人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、问题解决、感知和语言理解。人工智能系统通常利用大量数据进行训练，可以识别模式、做出决策和预测未来行为。",
        "metadata": {"category": "AI", "level": "introductory", "tags": ["AI", "overview"]}
    },
    {
        "title": "机器学习基础",
        "content": "机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统，而无需被明确编程。常见的机器学习方法包括监督学习、无监督学习和强化学习。监督学习使用标记数据训练模型，无监督学习寻找数据中的隐藏模式，而强化学习则通过奖励系统学习最佳行动。",
        "metadata": {"category": "AI", "subcategory": "machine_learning", "level": "introductory", "tags": ["ML", "AI"]}
    },
    {
        "title": "深度学习技术",
        "content": "深度学习是机器学习的一种特殊方法，它使用多层神经网络来学习复杂模式。神经网络的灵感来自人类大脑的结构，由相互连接的节点（神经元）组成。深度学习在图像识别、自然语言处理和游戏中取得了显著成功。最流行的深度学习架构包括卷积神经网络和循环神经网络。",
        "metadata": {"category": "AI", "subcategory": "deep_learning", "level": "intermediate", "tags": ["DL", "neural networks"]}
    },
    {
        "title": "自然语言处理应用",
        "content": "自然语言处理（NLP）是人工智能的一个分支，它关注计算机理解和生成人类语言的能力。NLP应用包括机器翻译、情感分析、文本摘要和聊天机器人。现代NLP系统通常基于Transformer架构，如GPT（生成式预训练Transformer）和BERT（来自Transformers的双向编码器表示）。",
        "metadata": {"category": "AI", "subcategory": "nlp", "level": "intermediate", "tags": ["NLP", "language"]}
    },
    {
        "title": "计算机视觉技术",
        "content": "计算机视觉是人工智能的一个领域，它使计算机能够理解和解释视觉信息。计算机视觉系统可以检测对象、识别人脸、分析场景和跟踪运动。它们广泛应用于自动驾驶汽车、监控系统、医学成像和增强现实。卷积神经网络已成为计算机视觉中最成功的架构之一。",
        "metadata": {"category": "AI", "subcategory": "computer_vision", "level": "intermediate", "tags": ["CV", "vision"]}
    }
]

# 测试查询
TEST_QUERIES = [
    {"text": "什么是深度学习?", "expected_category": "deep_learning"},
    {"text": "解释机器学习和人工智能的区别", "expected_category": "machine_learning"},
    {"text": "自然语言处理有哪些应用?", "expected_category": "nlp"},
    {"text": "计算机视觉技术如何工作?", "expected_category": "computer_vision"}
]


def create_test_documents():
    """创建测试文档并返回它们的路径"""
    doc_paths = []
    
    # 清理旧文件
    for file in TEST_DATA_DIR.glob("*.md"):
        file.unlink()
    
    # 创建新文件
    for doc in TEST_DOCUMENTS:
        file_name = f"{doc['title'].replace(' ', '_')}.md"
        file_path = TEST_DATA_DIR / file_name
        
        # 创建Markdown内容
        content = f"# {doc['title']}\n\n"
        
        # 添加前置元数据
        content += "---\n"
        for key, value in doc["metadata"].items():
            if isinstance(value, list):
                content += f"{key}: {', '.join(value)}\n"
            else:
                content += f"{key}: {value}\n"
        content += "---\n\n"
        
        # 添加正文
        content += doc["content"]
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        doc_paths.append(str(file_path))
    
    return doc_paths


class MockLLMClient:
    """模拟LLM客户端，用于测试"""
    
    def __init__(self, return_prompt=False):
        """初始化模拟客户端"""
        self.return_prompt = return_prompt
        self.last_prompt = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """模拟生成回答"""
        self.last_prompt = prompt
        
        if self.return_prompt:
            return f"PROMPT: {prompt[:50]}..."
        
        # 根据提示内容生成简单回答
        if "深度学习" in prompt:
            return "深度学习是机器学习的一种方法，使用多层神经网络来学习复杂模式。"
        elif "机器学习" in prompt:
            return "机器学习是AI的一个子领域，关注从数据中学习的系统。"
        elif "自然语言处理" in prompt:
            return "自然语言处理应用包括机器翻译、情感分析、文本摘要和聊天机器人。"
        elif "计算机视觉" in prompt:
            return "计算机视觉使计算机能够理解视觉信息，应用于自动驾驶汽车等领域。"
        else:
            return "这是一个通用回答，因为无法确定具体的主题。"


def setup_basic_rag_components():
    """设置基本的RAG组件，用于测试"""
    from mcp.knowledge.embedding import EmbeddingConfig, LocalEmbedding
    from mcp.knowledge.storage import VectorStoreConfig, InMemoryVectorStore
    
    # 创建嵌入管理器
    embedding_config = EmbeddingConfig(
        model_name=TEST_EMBEDDING_MODEL,
        embedding_dim=TEST_EMBEDDING_DIM
    )
    embedding_manager = LocalEmbedding(config=embedding_config)
    embedding_manager.initialize()
    
    # 创建向量存储
    vector_config = VectorStoreConfig(
        embedding_dim=TEST_EMBEDDING_DIM,
        distance_metric="cosine"
    )
    vector_store = InMemoryVectorStore(config=vector_config)
    vector_store.initialize()
    
    return embedding_manager, vector_store


def process_test_documents(document_paths):
    """处理测试文档并返回处理结果"""
    from mcp.knowledge.indexing import (
        MarkdownProcessor,
        RecursiveTextChunker,
        ObsidianMetadataExtractor
    )
    
    # 创建文档处理器
    chunker = RecursiveTextChunker(
        chunk_size=512,
        chunk_overlap=100
    )
    metadata_extractor = ObsidianMetadataExtractor()
    processor = MarkdownProcessor(
        chunker=chunker,
        metadata_extractor=metadata_extractor
    )
    
    # 处理文档
    all_chunks = []
    for path in document_paths:
        chunks = processor.process(path)
        all_chunks.extend(chunks)
    
    return all_chunks


def index_test_documents(embedding_manager, vector_store, document_paths):
    """索引测试文档到向量存储"""
    from mcp.knowledge.storage import VectorRecord
    
    # 处理文档
    chunks = process_test_documents(document_paths)
    
    # 添加到向量存储
    for chunk in chunks:
        # 生成嵌入
        embedding = embedding_manager.embed_text(chunk.text).embedding
        
        # 添加到向量存储
        vector_store.add(VectorRecord(
            id=chunk.chunk_id,
            embedding=embedding,
            text=chunk.text,
            metadata=chunk.metadata
        ))
    
    return len(chunks)


def setup_rag_engine(embedding_manager=None, vector_store=None):
    """设置完整的RAG引擎，用于测试"""
    from mcp.knowledge.retrieval import RetrievalConfig, RetrievalStrategy
    from mcp.knowledge.augmentation import AugmentationConfig, AugmentationMode
    from mcp.knowledge.rag_engine import RAGEngine, RAGConfig, RAGStrategy
    
    # 如果没有提供，创建基本组件
    if embedding_manager is None or vector_store is None:
        embedding_manager, vector_store = setup_basic_rag_components()
        
        # 创建测试文档
        doc_paths = create_test_documents()
        
        # 索引文档
        index_test_documents(embedding_manager, vector_store, doc_paths)
    
    # 创建RAG配置
    retrieval_config = RetrievalConfig(
        strategy=RetrievalStrategy.HYBRID,
        top_k=3
    )
    
    augmentation_config = AugmentationConfig(
        mode=AugmentationMode.STRUCTURED
    )
    
    rag_config = RAGConfig(
        strategy=RAGStrategy.ADVANCED,
        retrieval_config=retrieval_config,
        augmentation_config=augmentation_config,
        enable_logging=False  # 关闭日志以避免测试输出过多
    )
    
    # 创建模拟LLM客户端
    llm_client = MockLLMClient()
    
    # 创建RAG引擎
    rag_engine = RAGEngine(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        llm_client=llm_client,
        config=rag_config
    )
    
    rag_engine.initialize()
    
    return rag_engine
