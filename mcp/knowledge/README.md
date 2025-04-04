# MCP知识增强层

## 概述

MCP知识增强层是MCP智能体中台的第三阶段开发内容，实现检索增强生成（RAG）系统与Obsidian知识库的深度集成，使模型能够基于用户的个人知识生成更相关、更准确的回答。

## 主要组件

### 文档处理系统 (Document Processing)

负责解析和处理多种格式的文档，将其转换为可检索的文本块。

- `DocumentProcessor`：文档处理器基类
- `TextChunker`：文本分块工具
- `MetadataExtractor`：元数据提取器

位置：`mcp/knowledge/indexing/`

### 嵌入管理系统 (Embedding Management)

负责将文本转换为向量嵌入，支持不同的嵌入模型。

- `EmbeddingManager`：嵌入管理器接口
- `LocalEmbedding`：本地嵌入模型实现
- `APIEmbedding`：API嵌入模型实现

位置：`mcp/knowledge/embedding/`

### 向量存储系统 (Vector Storage)

负责存储和管理向量嵌入，提供高效的相似度搜索。

- `VectorStore`：向量存储接口
- `InMemoryVectorStore`：内存向量存储
- `SQLiteVectorStore`：SQLite向量存储

位置：`mcp/knowledge/storage/`

### 知识检索系统 (Knowledge Retrieval)

负责根据查询在向量存储中搜索相关文档。

- `KnowledgeRetriever`：检索器接口
- `SimilarityRetriever`：相似度检索器
- `HybridRetriever`：混合检索器

位置：`mcp/knowledge/retrieval/`

### 上下文增强系统 (Context Augmentation)

负责构建适当的上下文提示，融合用户请求和检索结果。

- `ContextAugmenter`：上下文增强器
- `PromptBuilder`：提示构建器
- `TokenManager`：Token计数和管理

位置：`mcp/knowledge/augmentation/`

### RAG引擎 (RAG Engine)

核心引擎，协调各组件工作，实现端到端RAG流程。

- `RAGEngine`：主引擎类
- `RAGStrategy`：策略接口
- `BasicRAG`：基础RAG策略

位置：`mcp/knowledge/`

## 数据流程

知识增强层的数据流分为两个主要阶段：

### 索引阶段

```
文档 -> 文档处理器 -> 文本块 -> 嵌入管理器 -> 向量嵌入 -> 向量存储
```

### 查询阶段

```
用户查询 -> 嵌入 -> 检索器 -> 相关文档 -> 上下文增强器 -> 增强上下文 -> 模型生成
```

## 技术选型

- 嵌入模型：SentenceTransformers
- 向量存储：SQLite + Faiss
- 文档解析：python-markdown, pypdf
- 模型接口：复用阶段1-2 API

## 开发状态

阶段3正在规划设计中，详细信息请参考：
`/04-项目/01-进行中/MCP集成/阶段总结/阶段3-知识增强层设计文档.md`
