"""
上下文格式化器 - 负责格式化RAG上下文
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging

# 导入依赖模块
from ..retrieval import RetrievalResult

# 设置日志
logger = logging.getLogger(__name__)


class ContextFormatter:
    """
    上下文格式化器基类，负责格式化RAG上下文
    """
    
    def __init__(self):
        """初始化格式化器"""
        pass
    
    def format_retrieval_result(self, result: RetrievalResult, index: int) -> str:
        """
        格式化单个检索结果
        
        Args:
            result: 检索结果
            index: 结果索引
            
        Returns:
            格式化后的文本
        """
        # 基本格式：[索引] 内容
        return f"[{index+1}] {result.text}"
    
    def format_retrieval_results(self, results: List[RetrievalResult]) -> str:
        """
        格式化检索结果列表
        
        Args:
            results: 检索结果列表
            
        Returns:
            格式化后的文本
        """
        if not results:
            return ""
            
        formatted = []
        for i, result in enumerate(results):
            formatted.append(self.format_retrieval_result(result, i))
            
        return "\n\n".join(formatted)
    
    def format_system_prompt(self, retrieval_text: str, query: str) -> str:
        """
        格式化系统提示
        
        Args:
            retrieval_text: 检索结果文本
            query: 用户查询
            
        Returns:
            格式化后的系统提示
        """
        return (
            f"使用以下参考信息回答用户问题。如果参考信息无法回答问题，请基于你的知识谨慎回答。\n\n"
            f"{retrieval_text}"
        )
    
    def format_user_prompt(self, query: str) -> str:
        """
        格式化用户提示
        
        Args:
            query: 用户查询
            
        Returns:
            格式化后的用户提示
        """
        return query
    
    def format_full_prompt(self, 
                          system_prompt: str, 
                          user_prompt: str) -> str:
        """
        格式化完整提示
        
        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            
        Returns:
            格式化后的完整提示
        """
        return f"{system_prompt}\n\n用户问题: {user_prompt}"
    
    def get_formatter_type(self) -> str:
        """
        获取格式化器类型
        
        Returns:
            格式化器类型
        """
        return "default_formatter"


class DefaultFormatter(ContextFormatter):
    """
    默认格式化器，使用简单文本格式
    """
    
    def format_retrieval_result(self, result: RetrievalResult, index: int) -> str:
        """
        格式化单个检索结果
        
        Args:
            result: 检索结果
            index: 结果索引
            
        Returns:
            格式化后的文本
        """
        # 基本格式：[索引] 内容
        formatted = f"[{index+1}] {result.text}"
        
        # 添加元数据（如果有）
        if result.metadata:
            metadata_items = []
            for k, v in result.metadata.items():
                if v is not None and k not in ["text", "embedding"]:
                    metadata_items.append(f"{k}: {v}")
            
            if metadata_items:
                metadata_str = ", ".join(metadata_items)
                formatted += f"\n来源: {metadata_str}"
                
        return formatted
    
    def get_formatter_type(self) -> str:
        """
        获取格式化器类型
        
        Returns:
            格式化器类型
        """
        return "default_formatter"


class MarkdownFormatter(ContextFormatter):
    """
    Markdown格式化器，使用Markdown格式
    """
    
    def format_retrieval_result(self, result: RetrievalResult, index: int) -> str:
        """
        格式化单个检索结果
        
        Args:
            result: 检索结果
            index: 结果索引
            
        Returns:
            格式化后的文本
        """
        # Markdown格式：引用块+元数据
        formatted = f"### 参考资料 {index+1}\n\n"
        formatted += f"> {result.text.replace('\n', '\n> ')}\n\n"
        
        # 添加元数据（如果有）
        if result.metadata:
            metadata_items = []
            for k, v in result.metadata.items():
                if v is not None and k not in ["text", "embedding"]:
                    metadata_items.append(f"**{k}**: {v}")
            
            if metadata_items:
                metadata_str = " | ".join(metadata_items)
                formatted += f"*来源*: {metadata_str}\n\n"
                
        return formatted
    
    def format_system_prompt(self, retrieval_text: str, query: str) -> str:
        """
        格式化系统提示
        
        Args:
            retrieval_text: 检索结果文本
            query: 用户查询
            
        Returns:
            格式化后的系统提示
        """
        return (
            f"# 回答指南\n\n"
            f"请基于以下参考资料回答用户的问题。如果参考资料不足以回答问题，请基于你的知识谨慎回答。\n\n"
            f"使用Markdown格式，确保你的回答结构清晰。引用相关的参考资料编号 [X]。\n\n"
            f"---\n\n"
            f"# 参考资料\n\n"
            f"{retrieval_text}\n\n"
            f"---\n\n"
        )
    
    def format_user_prompt(self, query: str) -> str:
        """
        格式化用户提示
        
        Args:
            query: 用户查询
            
        Returns:
            格式化后的用户提示
        """
        return f"# 用户问题\n\n{query}"
    
    def get_formatter_type(self) -> str:
        """
        获取格式化器类型
        
        Returns:
            格式化器类型
        """
        return "markdown_formatter"


class SchemaFormatter(ContextFormatter):
    """
    模式格式化器，使用XML标记格式化内容
    """
    
    def format_retrieval_result(self, result: RetrievalResult, index: int) -> str:
        """
        格式化单个检索结果
        
        Args:
            result: 检索结果
            index: 结果索引
            
        Returns:
            格式化后的文本
        """
        # XML格式
        formatted = f"<reference id=\"{index+1}\">\n"
        formatted += f"  <content>{self._escape_xml(result.text)}</content>\n"
        
        # 添加元数据（如果有）
        if result.metadata:
            formatted += "  <metadata>\n"
            for k, v in result.metadata.items():
                if v is not None and k not in ["text", "embedding"]:
                    formatted += f"    <{k}>{self._escape_xml(str(v))}</{k}>\n"
            formatted += "  </metadata>\n"
                
        formatted += "</reference>"
        return formatted
    
    def format_retrieval_results(self, results: List[RetrievalResult]) -> str:
        """
        格式化检索结果列表
        
        Args:
            results: 检索结果列表
            
        Returns:
            格式化后的文本
        """
        if not results:
            return "<references></references>"
            
        formatted = ["<references>"]
        for i, result in enumerate(results):
            formatted.append(self.format_retrieval_result(result, i))
        formatted.append("</references>")
            
        return "\n".join(formatted)
    
    def format_system_prompt(self, retrieval_text: str, query: str) -> str:
        """
        格式化系统提示
        
        Args:
            retrieval_text: 检索结果文本
            query: 用户查询
            
        Returns:
            格式化后的系统提示
        """
        return (
            f"<system>\n"
            f"  <instructions>\n"
            f"    使用提供的参考资料回答用户问题。如果参考资料不足以回答问题，请基于你的知识谨慎回答。\n"
            f"    在回答中引用相关参考资料的编号，格式为 [ID]。\n"
            f"    使用<answer>标签包裹你的回答。\n"
            f"  </instructions>\n\n"
            f"{retrieval_text}\n"
            f"</system>"
        )
    
    def format_user_prompt(self, query: str) -> str:
        """
        格式化用户提示
        
        Args:
            query: 用户查询
            
        Returns:
            格式化后的用户提示
        """
        return f"<query>{self._escape_xml(query)}</query>"
    
    def format_full_prompt(self, 
                          system_prompt: str, 
                          user_prompt: str) -> str:
        """
        格式化完整提示
        
        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            
        Returns:
            格式化后的完整提示
        """
        return f"{system_prompt}\n\n{user_prompt}\n\n<answer>\n"
    
    def get_formatter_type(self) -> str:
        """
        获取格式化器类型
        
        Returns:
            格式化器类型
        """
        return "schema_formatter"
    
    def _escape_xml(self, text: str) -> str:
        """
        转义XML特殊字符
        
        Args:
            text: 原始文本
            
        Returns:
            转义后的文本
        """
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;")
                .replace("'", "&apos;")
        )
