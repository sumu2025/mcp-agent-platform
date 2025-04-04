"""
元数据提取器 - 从文档中提取结构化元数据
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
from datetime import datetime
import os

from .base import MetadataExtractor


class MarkdownMetadataExtractor(MetadataExtractor):
    """
    Markdown元数据提取器 - 从Markdown文档中提取元数据
    """
    
    def __init__(self):
        """初始化提取器"""
        # 标签模式，匹配 #tag 或 #tag/subtag 格式
        self.tag_pattern = re.compile(r'#([a-zA-Z0-9_/\-]+)')
        
        # 链接模式，匹配 [[链接]] 或 [[链接|显示文本]] 格式
        self.link_pattern = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]')
        
        # 标题模式
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
    def extract(self, text: str, file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        从文本提取元数据
        
        Args:
            text: 文本内容
            file_path: 文件路径（可选）
            
        Returns:
            元数据字典
        """
        metadata = {}
        
        # 提取标签
        tags = self._extract_tags(text)
        if tags:
            metadata["tags"] = tags
        
        # 提取链接
        links = self._extract_links(text)
        if links:
            metadata["links"] = links
        
        # 提取标题结构
        headings = self._extract_headings(text)
        if headings:
            metadata["headings"] = headings
            # 使用第一个H1标题作为文档标题
            h1_headings = [h["text"] for h in headings if h["level"] == 1]
            if h1_headings:
                metadata["title"] = h1_headings[0]
        
        # 提取文件相关元数据
        if file_path:
            path = Path(file_path)
            file_metadata = self._get_file_metadata(path)
            metadata.update(file_metadata)
        
        return metadata
    
    def get_extractor_name(self) -> str:
        """
        获取提取器名称
        
        Returns:
            提取器名称
        """
        return "markdown_metadata_extractor"
    
    def _extract_tags(self, text: str) -> List[str]:
        """
        提取标签
        
        Args:
            text: 文本内容
            
        Returns:
            标签列表
        """
        tags = set()
        for match in self.tag_pattern.finditer(text):
            tag = match.group(1)
            tags.add(tag)
        return sorted(list(tags))
    
    def _extract_links(self, text: str) -> List[Dict[str, str]]:
        """
        提取文内链接
        
        Args:
            text: 文本内容
            
        Returns:
            链接列表，每个链接为字典，包含target和text字段
        """
        links = []
        for match in self.link_pattern.finditer(text):
            target = match.group(1)
            display_text = match.group(2) if match.group(2) else target
            links.append({
                "target": target,
                "text": display_text
            })
        return links
    
    def _extract_headings(self, text: str) -> List[Dict[str, Any]]:
        """
        提取标题结构
        
        Args:
            text: 文本内容
            
        Returns:
            标题列表，每个标题为字典，包含level和text字段
        """
        headings = []
        for match in self.heading_pattern.finditer(text):
            level = len(match.group(1))  # 标题级别（#的数量）
            heading_text = match.group(2)  # 标题文本
            headings.append({
                "level": level,
                "text": heading_text
            })
        return headings
    
    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        获取文件相关元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件元数据
        """
        try:
            stats = file_path.stat()
            metadata = {
                "source_path": str(file_path),
                "filename": file_path.name,
                "file_extension": file_path.suffix,
                "basename": file_path.stem
            }
            
            # 添加时间信息
            try:
                metadata["created_time"] = datetime.fromtimestamp(stats.st_ctime).isoformat()
                metadata["modified_time"] = datetime.fromtimestamp(stats.st_mtime).isoformat()
            except:
                # 时间戳可能无效
                pass
                
            # 添加文件大小
            metadata["file_size"] = stats.st_size
            
            # 添加目录信息
            metadata["directory"] = str(file_path.parent)
            
            return metadata
        except Exception as e:
            # 如果获取元数据失败，返回最基本的信息
            return {
                "source_path": str(file_path) if file_path else None
            }


class ObsidianMetadataExtractor(MarkdownMetadataExtractor):
    """
    Obsidian元数据提取器 - 专门处理Obsidian笔记的元数据
    """
    
    def __init__(self):
        """初始化提取器"""
        super().__init__()
        
        # Obsidian特有的属性模式，匹配Property:: Value
        self.property_pattern = re.compile(r'^([A-Za-z0-9_-]+)::\s*(.+)$', re.MULTILINE)
        
        # Callout模式
        self.callout_pattern = re.compile(r'>\s*\[!([A-Za-z0-9_-]+)\]([^\n]*)')
        
        # Dataview查询模式
        self.dataview_pattern = re.compile(r'```dataview\s(.*?)```', re.DOTALL)
        
    def extract(self, text: str, file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        从文本提取元数据
        
        Args:
            text: 文本内容
            file_path: 文件路径（可选）
            
        Returns:
            元数据字典
        """
        # 首先获取基础Markdown元数据
        metadata = super().extract(text, file_path)
        
        # 提取Obsidian特有的属性
        properties = self._extract_properties(text)
        if properties:
            metadata["properties"] = properties
            # 将顶级属性也添加到元数据中
            metadata.update(properties)
        
        # 提取Callout信息
        callouts = self._extract_callouts(text)
        if callouts:
            metadata["callouts"] = callouts
        
        # 提取Dataview查询
        dataviews = self._extract_dataviews(text)
        if dataviews:
            metadata["dataviews"] = dataviews
        
        # 检查是否有特殊文件夹结构（如Daily Notes）
        if file_path:
            path = Path(file_path)
            folder_metadata = self._extract_folder_metadata(path)
            metadata.update(folder_metadata)
        
        return metadata
    
    def get_extractor_name(self) -> str:
        """
        获取提取器名称
        
        Returns:
            提取器名称
        """
        return "obsidian_metadata_extractor"
    
    def _extract_properties(self, text: str) -> Dict[str, Any]:
        """
        提取Obsidian属性
        
        Args:
            text: 文本内容
            
        Returns:
            属性字典
        """
        properties = {}
        for match in self.property_pattern.finditer(text):
            key = match.group(1).strip()
            value = match.group(2).strip()
            
            # 尝试转换值类型
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            
            properties[key] = value
            
        return properties
    
    def _extract_callouts(self, text: str) -> List[Dict[str, str]]:
        """
        提取Callout信息
        
        Args:
            text: 文本内容
            
        Returns:
            Callout列表
        """
        callouts = []
        for match in self.callout_pattern.finditer(text):
            callout_type = match.group(1)
            callout_text = match.group(2).strip() if match.group(2) else ""
            callouts.append({
                "type": callout_type,
                "text": callout_text
            })
        return callouts
    
    def _extract_dataviews(self, text: str) -> List[str]:
        """
        提取Dataview查询
        
        Args:
            text: 文本内容
            
        Returns:
            Dataview查询列表
        """
        dataviews = []
        for match in self.dataview_pattern.finditer(text):
            dataview_query = match.group(1).strip()
            dataviews.append(dataview_query)
        return dataviews
    
    def _extract_folder_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        从文件路径提取额外元数据（基于文件夹结构）
        
        Args:
            file_path: 文件路径
            
        Returns:
            额外元数据
        """
        metadata = {}
        path_str = str(file_path)
        
        # 检测是否在Daily Notes文件夹中
        if '/Daily/' in path_str or '\\Daily\\' in path_str:
            metadata["is_daily_note"] = True
            
            # 尝试从文件名解析日期
            try:
                filename = file_path.stem
                # 常见的日期格式匹配
                date_patterns = [
                    # YYYY-MM-DD
                    r'(\d{4})-(\d{2})-(\d{2})',
                    # YYYYMMDD
                    r'(\d{4})(\d{2})(\d{2})'
                ]
                
                for pattern in date_patterns:
                    match = re.search(pattern, filename)
                    if match:
                        year, month, day = match.groups()
                        metadata["date"] = f"{year}-{month}-{day}"
                        break
            except:
                pass
                
        return metadata
