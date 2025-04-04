"""
文档处理系统测试 - 测试Markdown解析和智能分块功能
"""

import unittest
import os
import sys
from pathlib import Path
import tempfile
import shutil

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

from mcp.knowledge.indexing import (
    MarkdownProcessor,
    RecursiveTextChunker,
    ObsidianMetadataExtractor,
    TextChunk
)


class TestDocumentProcessing(unittest.TestCase):
    """测试文档处理系统"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试文档
        self.test_doc_path = os.path.join(self.temp_dir, "test_doc.md")
        with open(self.test_doc_path, 'w', encoding='utf-8') as f:
            f.write("""---
title: 测试文档
category: AI
tags: ['test', 'markdown']
---

# 测试标题

这是第一段落，包含一些测试内容。

## 二级标题

这是第二段落，也包含一些测试内容。

### 三级标题

1. 列表项1
2. 列表项2
3. 列表项3
""")
        
        # 创建分块器
        self.chunker = RecursiveTextChunker(
            chunk_size=100,
            chunk_overlap=20
        )
        
        # 创建元数据提取器
        self.metadata_extractor = ObsidianMetadataExtractor()
        
        # 创建文档处理器
        self.processor = MarkdownProcessor(
            chunker=self.chunker,
            metadata_extractor=self.metadata_extractor
        )
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_extract_metadata(self):
        """测试元数据提取"""
        metadata = self.processor.extract_metadata(self.test_doc_path)
        
        # 验证基本元数据
        self.assertIn('title', metadata)
        self.assertEqual(metadata['title'], '测试文档')
        self.assertIn('category', metadata)
        self.assertEqual(metadata['category'], 'AI')
        
        # 验证文件元数据
        self.assertIn('filename', metadata)
        self.assertEqual(metadata['filename'], 'test_doc.md')
    
    def test_process_document(self):
        """测试文档处理"""
        chunks = self.processor.process(self.test_doc_path)
        
        # 验证是否生成了块
        self.assertGreater(len(chunks), 0)
        
        # 验证块类型
        for chunk in chunks:
            self.assertIsInstance(chunk, TextChunk)
    
    def test_chunker_section_splitting(self):
        """测试分块器的章节分割功能"""
        text = """# 一级标题A

这是内容1。

## 二级标题A1

这是内容2。

## 二级标题A2

这是内容3。

# 一级标题B

这是内容4。
"""
        
        chunks = self.chunker.split(text)
        
        # 检查是否正确分割了章节
        sections_found = set()
        for chunk in chunks:
            chunk_text = chunk.text
            if "一级标题A" in chunk_text: sections_found.add("一级标题A")
            if "二级标题A1" in chunk_text: sections_found.add("二级标题A1")
            if "二级标题A2" in chunk_text: sections_found.add("二级标题A2")
            if "一级标题B" in chunk_text: sections_found.add("一级标题B")
        
        # 验证所有章节都被找到
        self.assertIn("一级标题A", sections_found)
        self.assertIn("二级标题A1", sections_found)
        self.assertIn("二级标题A2", sections_found)
        self.assertIn("一级标题B", sections_found)
    
    def test_chunker_overlap(self):
        """测试分块器的块重叠功能"""
        # 创建一个长文本，确保会被分成多个块
        long_text = "这是一个测试句子。" * 30
        
        # 设置较大的重叠，确保能明显观察到
        overlap_chunker = RecursiveTextChunker(
            chunk_size=100,
            chunk_overlap=50
        )
        
        chunks = overlap_chunker.split(long_text)
        
        # 确保至少有两个块
        self.assertGreaterEqual(len(chunks), 2)
        
        # 验证重叠
        for i in range(len(chunks) - 1):
            # 当前块的结尾部分
            current_end = chunks[i].text[-30:]
            # 下一个块的开头部分
            next_start = chunks[i+1].text[:30]
            
            # 应该有一些重叠内容
            overlap = any(phrase in next_start for phrase in current_end.split())
            self.assertTrue(overlap, f"块 {i} 和 {i+1} 之间没有预期的重叠")
    
    def test_obsidian_metadata_extractor(self):
        """测试Obsidian元数据提取器"""
        # 创建测试文档
        obsidian_doc_path = os.path.join(self.temp_dir, "obsidian_test.md")
        with open(obsidian_doc_path, 'w', encoding='utf-8') as f:
            f.write("""---
title: Obsidian测试
category: Test
tags: [obsidian, test]
---

# Obsidian测试

这是一个[[内部链接]]到其他笔记。

一些属性：
Status:: 完成
Priority:: 高
Author:: 测试者

> [!note]
> 这是一个Obsidian callout。

- [ ] 任务1
- [x] 任务2
""")
        
        # 提取元数据
        with open(obsidian_doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = self.metadata_extractor.extract(content, obsidian_doc_path)
        
        # 验证前置元数据
        self.assertEqual(metadata.get('title'), 'Obsidian测试')
        self.assertEqual(metadata.get('category'), 'Test')
        
        # 验证Obsidian特定元数据
        self.assertIn('links', metadata)
        links = metadata['links']
        self.assertTrue(any(link['target'] == '内部链接' for link in links))
        
        # 验证属性
        self.assertIn('properties', metadata)
        props = metadata['properties']
        self.assertEqual(props.get('Status'), '完成')
        self.assertEqual(props.get('Priority'), '高')
        self.assertEqual(props.get('Author'), '测试者')
        
        # 验证Callout
        self.assertIn('callouts', metadata)
        callouts = metadata['callouts']
        self.assertTrue(any(callout['type'] == 'note' for callout in callouts))
