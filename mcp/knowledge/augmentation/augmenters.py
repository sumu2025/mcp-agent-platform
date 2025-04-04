"""
上下文增强器 - 实现各种上下文增强策略
"""

import os
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from datetime import datetime

# 导入依赖模块
from .base import (
    ContextAugmenter, 
    AugmentationConfig, 
    AugmentedContext,
    AugmentationMode
)
from .prompt_builder import (
    PromptBuilder,
    BasicPromptBuilder,
    StructuredPromptBuilder,
    TemplatePromptBuilder
)
from .context_formatter import (
    ContextFormatter,
    DefaultFormatter,
    MarkdownFormatter,
    SchemaFormatter
)
from .token_management import TokenManager
from ..retrieval import RetrievalResult

# 设置日志
logger = logging.getLogger(__name__)


class BasicAugmenter(ContextAugmenter):
    """
    基本上下文增强器，使用简单格式增强上下文
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        初始化基本增强器
        
        Args:
            config: 增强配置，如果为None则使用默认配置
        """
        super().__init__(config)
        
        self.prompt_builder = None
        self.formatter = None
        self.token_manager = None
    
    def initialize(self) -> None:
        """初始化增强器"""
        # 创建提示构建器
        self.prompt_builder = BasicPromptBuilder()
        
        # 创建格式化器
        self.formatter = DefaultFormatter()
        
        # 创建Token管理器
        if self.config.use_token_management:
            self.token_manager = TokenManager()
        
        self._initialized = True
        logger.info(f"基本上下文增强器初始化完成，模式: {self.config.mode.name}")
    
    def augment(self, 
               query: str, 
               retrieval_results: List[RetrievalResult]) -> AugmentedContext:
        """
        增强上下文
        
        Args:
            query: 用户查询
            retrieval_results: 检索结果列表
            
        Returns:
            增强后的上下文
        """
        self.ensure_initialized()
        
        # 格式化检索结果
        retrieval_text = self.format_retrieval_results(retrieval_results)
        
        # 构建系统提示
        system_prompt = self.build_system_prompt(retrieval_text, query)
        
        # 构建用户提示
        user_prompt = self.build_user_prompt(query)
        
        # 管理上下文长度
        system_prompt, retrieval_text, user_prompt = self.manage_context_length(
            system_prompt, retrieval_text, user_prompt
        )
        
        # 构建完整提示
        full_prompt = self.formatter.format_full_prompt(system_prompt, user_prompt)
        
        # 计算token
        token_count = {}
        if self.token_manager:
            token_count = {
                "system": self.token_manager.count_tokens(system_prompt),
                "retrieval": self.token_manager.count_tokens(retrieval_text),
                "query": self.token_manager.count_tokens(user_prompt),
                "total": self.token_manager.count_tokens(full_prompt)
            }
        
        # 创建增强上下文
        context = AugmentedContext(
            prompt=full_prompt,
            system=system_prompt,
            retrieval=retrieval_text,
            query=user_prompt,
            retrieval_results=retrieval_results,
            token_count=token_count
        )
        
        return context
    
    def format_retrieval_results(self, 
                                retrieval_results: List[RetrievalResult]) -> str:
        """
        格式化检索结果
        
        Args:
            retrieval_results: 检索结果列表
            
        Returns:
            格式化后的文本
        """
        return self.formatter.format_retrieval_results(retrieval_results)
    
    def build_system_prompt(self, 
                           retrieval_text: str,
                           query: str) -> str:
        """
        构建系统提示
        
        Args:
            retrieval_text: 格式化后的检索结果文本
            query: 用户查询
            
        Returns:
            系统提示文本
        """
        return self.formatter.format_system_prompt(retrieval_text, query)
    
    def build_user_prompt(self, query: str) -> str:
        """
        构建用户提示
        
        Args:
            query: 用户查询
            
        Returns:
            用户提示文本
        """
        return self.formatter.format_user_prompt(query)
    
    def manage_context_length(self, 
                             system: str,
                             retrieval: str,
                             query: str) -> Tuple[str, str, str]:
        """
        管理上下文长度，确保不超过最大长度
        
        Args:
            system: 系统指令部分
            retrieval: 检索结果部分
            query: 用户查询部分
            
        Returns:
            调整后的(系统指令, 检索结果, 用户查询)元组
        """
        if not self.token_manager or not self.config.use_token_management:
            return system, retrieval, query
            
        # 计算各部分的token数
        system_tokens = self.token_manager.count_tokens(system)
        retrieval_tokens = self.token_manager.count_tokens(retrieval)
        query_tokens = self.token_manager.count_tokens(query)
        total_tokens = system_tokens + retrieval_tokens + query_tokens
        
        # 检查是否超过最大长度
        if total_tokens <= self.config.max_context_length:
            return system, retrieval, query
            
        # 需要减少的token数
        excess_tokens = total_tokens - self.config.max_context_length
        
        # 根据优先级顺序调整各部分长度
        sections = {
            "system": system,
            "retrieval": retrieval,
            "query": query
        }
        
        priorities = self.config.priority_order
        
        # 分配token
        adjusted = self.token_manager.distribute_tokens(
            self.config.max_context_length,
            sections,
            priorities
        )
        
        return adjusted["system"], adjusted["retrieval"], adjusted["query"]
    
    def get_token_count(self, text: str) -> int:
        """
        获取文本的token数量
        
        Args:
            text: 文本
            
        Returns:
            token数量
        """
        if self.token_manager:
            return self.token_manager.count_tokens(text)
        else:
            # 简单估计: 单词数量的1.3倍
            words = text.split()
            return int(len(words) * 1.3)
    
    def get_augmenter_type(self) -> str:
        """
        获取增强器类型
        
        Returns:
            增强器类型
        """
        return "basic_augmenter"
    
    def get_augmenter_info(self) -> Dict[str, Any]:
        """
        获取增强器信息
        
        Returns:
            增强器信息字典
        """
        return {
            "type": self.get_augmenter_type(),
            "mode": self.config.mode.name,
            "max_context_length": self.config.max_context_length,
            "use_token_management": self.config.use_token_management,
            "formatter": self.formatter.get_formatter_type() if self.formatter else None
        }


class StructuredAugmenter(BasicAugmenter):
    """
    结构化上下文增强器，使用结构化格式增强上下文
    """
    
    def initialize(self) -> None:
        """初始化增强器"""
        # 创建结构化提示构建器
        self.prompt_builder = StructuredPromptBuilder()
        
        # 创建Markdown格式化器
        self.formatter = MarkdownFormatter()
        
        # 创建Token管理器
        if self.config.use_token_management:
            self.token_manager = TokenManager()
        
        self._initialized = True
        logger.info(f"结构化上下文增强器初始化完成，模式: {self.config.mode.name}")
    
    def build_system_prompt(self, 
                           retrieval_text: str,
                           query: str) -> str:
        """
        构建系统提示
        
        Args:
            retrieval_text: 格式化后的检索结果文本
            query: 用户查询
            
        Returns:
            系统提示文本
        """
        # 使用提示构建器的结构化模板
        return self.prompt_builder.build_system_prompt(
            retrieval_text, 
            query,
            template="structured_system"
        )
    
    def get_augmenter_type(self) -> str:
        """
        获取增强器类型
        
        Returns:
            增强器类型
        """
        return "structured_augmenter"


class TemplateAugmenter(BasicAugmenter):
    """
    模板上下文增强器，使用自定义模板增强上下文
    """
    
    def __init__(self, 
                config: Optional[AugmentationConfig] = None,
                template_name: Optional[str] = None,
                templates_dir: Optional[str] = None):
        """
        初始化模板增强器
        
        Args:
            config: 增强配置，如果为None则使用默认配置
            template_name: 模板名称
            templates_dir: 模板目录
        """
        super().__init__(config)
        
        self.template_name = template_name or "educational"
        self.templates_dir = templates_dir
    
    def initialize(self) -> None:
        """初始化增强器"""
        # 创建模板提示构建器
        self.prompt_builder = TemplatePromptBuilder(self.templates_dir)
        
        # 创建格式化器，根据模板选择
        if self.template_name in ["academic", "medical"]:
            self.formatter = SchemaFormatter()
        else:
            self.formatter = MarkdownFormatter()
        
        # 创建Token管理器
        if self.config.use_token_management:
            self.token_manager = TokenManager()
        
        self._initialized = True
        logger.info(f"模板上下文增强器初始化完成，模板: {self.template_name}")
    
    def build_system_prompt(self, 
                           retrieval_text: str,
                           query: str) -> str:
        """
        构建系统提示
        
        Args:
            retrieval_text: 格式化后的检索结果文本
            query: 用户查询
            
        Returns:
            系统提示文本
        """
        # 使用提示构建器的特定模板
        return self.prompt_builder.build_system_prompt(
            retrieval_text, 
            query,
            template=self.template_name
        )
    
    def get_augmenter_type(self) -> str:
        """
        获取增强器类型
        
        Returns:
            增强器类型
        """
        return "template_augmenter"
    
    def get_augmenter_info(self) -> Dict[str, Any]:
        """
        获取增强器信息
        
        Returns:
            增强器信息字典
        """
        info = super().get_augmenter_info()
        info.update({
            "template_name": self.template_name,
            "templates_dir": self.templates_dir
        })
        return info


def get_augmenter(config: Optional[AugmentationConfig] = None) -> ContextAugmenter:
    """
    获取上下文增强器
    
    Args:
        config: 增强配置，如果为None则使用默认配置
        
    Returns:
        上下文增强器
    """
    config = config or AugmentationConfig()
    
    if config.mode == AugmentationMode.BASIC:
        return BasicAugmenter(config)
    elif config.mode == AugmentationMode.STRUCTURED:
        return StructuredAugmenter(config)
    elif config.mode == AugmentationMode.TEMPLATE:
        template_name = config.custom_params.get("template_name", "educational")
        templates_dir = config.custom_params.get("templates_dir")
        return TemplateAugmenter(config, template_name, templates_dir)
    else:
        logger.warning(f"未知的增强模式: {config.mode}，使用结构化模式")
        return StructuredAugmenter(config)
