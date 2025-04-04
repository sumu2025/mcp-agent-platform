"""
Markdown文档处理器 - 实现对Markdown文档的解析和处理
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import yaml
from datetime import datetime

from .base import DocumentProcessor, TextChunk, TextChunker, MetadataExtractor


class MarkdownProcessor(DocumentProcessor):
    """
    Markdown文档处理器，实现对Markdown文档的解析和处理
    """
    
    def __init__(self, 
                 chunker: Optional[TextChunker] = None,
                 metadata_extractor: Optional[MetadataExtractor] = None):
        """
        初始化Markdown处理器
        
        Args:
            chunker: 文本分块器，如果为None则使用默认分块器
            metadata_extractor: 元数据提取器，如果为None则使用默认提取器
        """
        self.chunker = chunker
        self.metadata_extractor = metadata_extractor
        
    def process(self, file_path: Union[str, Path]) -> List[TextChunk]:
        """
        处理Markdown文档，返回文本块列表
        
        Args:
            file_path: 文档路径
            
        Returns:
            文本块列表
        """
        file_path = Path(file_path)
        if not self.supports(file_path):
            raise ValueError(f"不支持处理文件：{file_path}")
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取元数据和正文
        metadata, body = self._extract_front_matter(content)
        
        # 添加文件元数据
        file_metadata = self._get_file_metadata(file_path)
        metadata.update(file_metadata)
        
        # 如果有自定义分块器，使用它分块
        if self.chunker:
            return self.chunker.split(body, metadata=metadata)
        
        # 否则使用简单分块策略
        return self._default_chunking(body, metadata, file_path)
    
    def supports(self, file_path: Union[str, Path]) -> bool:
        """
        检查是否支持处理该文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            是否支持
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in ['.md', '.markdown']
    
    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        提取文档元数据
        
        Args:
            file_path: 文档路径
            
        Returns:
            元数据字典
        """
        file_path = Path(file_path)
        if not self.supports(file_path):
            raise ValueError(f"不支持处理文件：{file_path}")
            
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取YAML元数据
        yaml_metadata, _ = self._extract_front_matter(content)
        
        # 添加文件元数据
        file_metadata = self._get_file_metadata(file_path)
        yaml_metadata.update(file_metadata)
        
        # 如果有元数据提取器，使用它提取额外元数据
        if self.metadata_extractor:
            extra_metadata = self.metadata_extractor.extract(content, file_path)
            yaml_metadata.update(extra_metadata)
            
        return yaml_metadata
    
    def get_processor_name(self) -> str:
        """
        获取处理器名称
        
        Returns:
            处理器名称
        """
        return "markdown_processor"
    
    def _extract_front_matter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """
        从Markdown内容中提取YAML元数据
        
        Args:
            content: Markdown内容
            
        Returns:
            (元数据字典, 正文内容)
        """
        metadata = {}
        body = content
        
        # 检查是否有YAML前置元数据（文档开头的---包围的YAML块）
        yaml_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
        yaml_match = yaml_pattern.match(content)
        
        if yaml_match:
            yaml_text = yaml_match.group(1)
            try:
                metadata = yaml.safe_load(yaml_text) or {}
                # 移除YAML块，只保留正文
                body = content[yaml_match.end():]
            except Exception as e:
                # YAML解析错误，忽略前置元数据
                print(f"YAML解析错误: {e}")
        
        return metadata, body
    
    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        获取文件相关元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件元数据
        """
        stats = file_path.stat()
        return {
            "source_path": str(file_path),
            "filename": file_path.name,
            "file_extension": file_path.suffix,
            "created_time": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "file_size": stats.st_size
        }
    
    def _default_chunking(self, text: str, metadata: Dict[str, Any], file_path: Path) -> List[TextChunk]:
        """
        默认的简单分块策略
        
        Args:
            text: 文本内容
            metadata: 元数据
            file_path: 文件路径
            
        Returns:
            文本块列表
        """
        # 简单地按段落分块
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        for i, paragraph in enumerate(paragraphs):
            # 创建一个唯一ID
            chunk_id = f"{file_path.stem}_{i}"
            
            # 创建元数据的副本，添加索引信息
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(paragraphs)
            })
            
            # 创建文本块
            chunk = TextChunk(
                text=paragraph,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                source_document=str(file_path),
                chunk_index=i
            )
            
            chunks.append(chunk)
            
        return chunks
