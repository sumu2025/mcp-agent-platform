"""
RAG 功能演示脚本

此脚本演示如何使用MCP知识增强层的文档处理功能，包括:
1. 使用MarkdownProcessor处理Markdown文档
2. 使用RecursiveTextChunker进行智能分块
3. 使用ObsidianMetadataExtractor提取元数据

用法:
python rag_demo.py /path/to/markdown/file.md
"""

import sys
import os
import json
from pathlib import Path
from pprint import pprint

# 确保可以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

from mcp.knowledge.indexing import (
    MarkdownProcessor,
    RecursiveTextChunker,
    ObsidianMetadataExtractor
)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python rag_demo.py /path/to/markdown/file.md")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        sys.exit(1)
    
    # 创建元数据提取器
    metadata_extractor = ObsidianMetadataExtractor()
    
    # 创建文本分块器
    chunker = RecursiveTextChunker(
        chunk_size=512,       # 目标块大小（字符数）
        chunk_overlap=100,    # 块重叠（字符数）
        min_chunk_size=50     # 最小块大小（字符数）
    )
    
    # 创建Markdown处理器
    processor = MarkdownProcessor(
        chunker=chunker,
        metadata_extractor=metadata_extractor
    )
    
    print(f"处理文件: {file_path}")
    
    # 提取元数据
    print("\n--- 文档元数据 ---")
    metadata = processor.extract_metadata(file_path)
    pprint(metadata)
    
    # 处理文档，获取文本块
    print("\n--- 文档分块 ---")
    chunks = processor.process(file_path)
    
    print(f"文档被分割为 {len(chunks)} 个文本块")
    
    # 显示每个块的信息
    for i, chunk in enumerate(chunks, 1):
        print(f"\n块 {i}/{len(chunks)}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  长度: {len(chunk.text)} 字符")
        
        # 显示块的前100个字符
        preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
        print(f"  预览: {preview}")
        
        # 显示块的元数据
        if "heading" in chunk.metadata:
            print(f"  标题: {chunk.metadata['heading']} (级别 {chunk.metadata['heading_level']})")
        
        if "tags" in chunk.metadata and chunk.metadata["tags"]:
            print(f"  标签: {', '.join(chunk.metadata['tags'])}")
    
    print("\n处理完成!")


if __name__ == "__main__":
    main()
