"""
文档处理系统测试 - 专注于测试MarkdownProcessor和RecursiveTextChunker
"""

import unittest
import os
import sys
from pathlib import Path
import tempfile

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

from mcp.knowledge.indexing import (
    MarkdownProcessor,
    RecursiveTextChunker,
    ObsidianMetadataExtractor,
    TextChunk
)


class TestDocumentProcessor(unittest.TestCase):
    """测试文档处理器功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时测试文件
        self.test_content = """---
title: 测试文档
category: AI
tags: test, markdown
---

# 测试标题

这是第一段落，包含一些测试内容。

## 二级标题

这是第二段落，也包含一些测试内容。

### 三级标题

1. 列表项1
2. 列表项2
"""
        # 使用临时文件
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = os.path.join(self.temp_dir.name, "test_doc.md")
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write(self.test_content)
            
        # 创建处理器组件
        self.chunker = RecursiveTextChunker(
            chunk_size=100,
            chunk_overlap=20
        )
        self.metadata_extractor = ObsidianMetadataExtractor()
        self.processor = MarkdownProcessor(
            chunker=self.chunker,
            metadata_extractor=self.metadata_extractor
        )
    
    def tearDown(self):
        """清理测试环境"""
        self.temp_dir.cleanup()
    
    def test_markdown_processor_supports(self):
        """测试MarkdownProcessor.supports方法"""
        # 应该支持.md文件
        self.assertTrue(self.processor.supports(self.test_file))
        
        # 不应该支持非Markdown文件
        txt_file = os.path.join(self.temp_dir.name, "test.txt")
        with open(txt_file, "w") as f:
            f.write("Plain text file")
        self.assertFalse(self.processor.supports(txt_file))
    
    def test_extract_metadata(self):
        """测试元数据提取"""
        metadata = self.processor.extract_metadata(self.test_file)
        
        # 验证YAML前置元数据
        self.assertIn("title", metadata)
        self.assertEqual(metadata["title"], "测试标题")
        self.assertIn("category", metadata)
        self.assertEqual(metadata["category"], "AI")
        
        # 验证文件元数据
        self.assertIn("filename", metadata)
        self.assertEqual(metadata["filename"], "test_doc.md")
    
    def test_document_processing(self):
        """测试文档处理过程"""
        chunks = self.processor.process(self.test_file)
        
        # 应该至少有一个块
        self.assertGreater(len(chunks), 0)
        
        # 验证块的类型和属性
        for chunk in chunks:
            self.assertIsInstance(chunk, TextChunk)
            self.assertIsNotNone(chunk.text)
            self.assertIsNotNone(chunk.metadata)
            self.assertIn("title", chunk.metadata)
    
    def test_chunker(self):
        """测试文本分块器"""
        text = "# 标题\n\n第一段内容\n\n## 子标题\n\n第二段内容"
        chunks = self.chunker.split(text)
        
        # 检查分块结果
        self.assertGreater(len(chunks), 0)
        # 检查内容完整性 - 所有块合起来应该包含原始内容的所有部分
        combined_text = " ".join([c.text for c in chunks])
        self.assertIn("第一段内容", combined_text)
        self.assertIn("第二段内容", combined_text)
        self.assertIn("子标题", combined_text)


if __name__ == "__main__":
    unittest.main()
