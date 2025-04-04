"""
增强的文档处理器测试
更全面地测试Markdown处理器和分块功能
"""

import sys
import os
import unittest
import tempfile
from pathlib import Path

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入所需组件
from mcp.knowledge.indexing import (
    MarkdownProcessor,
    RecursiveTextChunker,
    ObsidianMetadataExtractor,
    TextChunk
)


class TestDocumentProcessorEnhanced(unittest.TestCase):
    """增强的文档处理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 测试数据目录
        self.test_data_dir = Path(__file__).parent / "test_data"
        
        # 确保测试数据存在
        test_files = list(self.test_data_dir.glob("*.md"))
        if len(test_files) < 3:
            self.skipTest("测试数据不足，请先创建测试文档")
            
        # 创建分块器
        self.chunker = RecursiveTextChunker(
            chunk_size=200,
            chunk_overlap=50,
            min_chunk_size=20
        )
        
        # 创建元数据提取器
        self.metadata_extractor = ObsidianMetadataExtractor()
        
        # 创建处理器
        self.processor = MarkdownProcessor(
            chunker=self.chunker,
            metadata_extractor=self.metadata_extractor
        )
    
    def test_markdown_processing(self):
        """测试Markdown处理"""
        # 测试文件
        test_file = self.test_data_dir / "深度学习技术.md"
        
        # 处理文档
        chunks = self.processor.process(test_file)
        
        # 验证结果
        self.assertGreater(len(chunks), 0, "应该生成至少一个块")
        self.assertIsInstance(chunks[0], TextChunk, "结果应该是TextChunk类型")
        
        # 检查第一个块的内容
        self.assertIn("深度学习", chunks[0].text, "块内容应该包含'深度学习'")
        
        # 检查块的元数据
        self.assertIn("category", chunks[0].metadata, "元数据应该包含category字段")
        self.assertEqual("AI", chunks[0].metadata["category"], "category应该是AI")
        self.assertEqual("deep_learning", chunks[0].metadata["subcategory"], "subcategory应该是deep_learning")
    
    def test_metadata_extraction(self):
        """测试元数据提取"""
        # 测试文件
        test_file = self.test_data_dir / "自然语言处理应用.md"
        
        # 提取元数据
        metadata = self.processor.extract_metadata(test_file)
        
        # 验证元数据
        self.assertIn("category", metadata, "元数据应该包含category字段")
        self.assertIn("tags", metadata, "元数据应该包含tags字段")
        self.assertEqual("AI", metadata["category"], "category应该是AI")
        self.assertEqual("nlp", metadata["subcategory"], "subcategory应该是nlp")
        self.assertEqual("intermediate", metadata["level"], "level应该是intermediate")
        
        # 检查标签
        self.assertIsInstance(metadata["tags"], list, "tags应该是列表类型")
        self.assertIn("NLP", metadata["tags"], "tags应该包含NLP")
    
    def test_recursive_chunking(self):
        """测试递归分块"""
        # 创建测试文本（有明确的结构）
        test_text = """# 主标题

这是介绍段落，提供了整体概述。

## 第一部分

这是第一部分的内容。包含一些详细信息。
这是更多的详细信息，应该与上面的内容在同一个块中。

## 第二部分

这是第二部分的第一段内容。

这是第二部分的第二段内容，与上面的段落应该在不同的块中。

### 第二部分的子部分

这是子部分的内容。包含一些特定信息。
"""
        
        # 分块
        chunks = self.chunker.split(test_text)
        
        # 验证结果
        self.assertGreater(len(chunks), 2, "应该生成多个块")
        
        # 检查标题信息是否正确嵌入到元数据中
        has_main_title = False
        has_second_part = False
        
        for chunk in chunks:
            if chunk.metadata.get("heading") == "主标题":
                has_main_title = True
            elif chunk.metadata.get("heading") == "第二部分":
                has_second_part = True
                self.assertEqual(2, chunk.metadata.get("heading_level"), "第二部分的heading_level应该是2")
        
        self.assertTrue(has_main_title, "应该有主标题的块")
        self.assertTrue(has_second_part, "应该有第二部分的块")
    
    def test_handle_long_document(self):
        """测试处理长文档"""
        # 创建长文档
        long_text = "这是一个测试段落。" * 100
        
        # 创建临时Markdown文件
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", encoding="utf-8", delete=False) as f:
            f.write("# 长文档测试\n\n")
            f.write(long_text)
            temp_file = f.name
        
        try:
            # 处理文档
            chunks = self.processor.process(temp_file)
            
            # 验证结果
            self.assertGreater(len(chunks), 1, "长文档应该被分成多个块")
            
            # 检查块大小（不应超过chunk_size太多）
            for chunk in chunks:
                # 简单地计算字符数作为估计
                self.assertLessEqual(len(chunk.text), self.chunker.chunk_size * 1.5, 
                                   "每个块的大小不应该超过chunk_size的1.5倍")
                
        finally:
            # 清理临时文件
            os.unlink(temp_file)
    
    def test_supports_method(self):
        """测试supports方法"""
        # 支持的文件类型
        self.assertTrue(self.processor.supports("test.md"), "应该支持.md文件")
        self.assertTrue(self.processor.supports("test.markdown"), "应该支持.markdown文件")
        
        # 不支持的文件类型
        self.assertFalse(self.processor.supports("test.txt"), "不应该支持.txt文件")
        self.assertFalse(self.processor.supports("test.pdf"), "不应该支持.pdf文件")


if __name__ == "__main__":
    unittest.main()
