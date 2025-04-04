"""
上下文增强模块 - 负责构建和管理RAG的提示上下文
"""

from .base import ContextAugmenter, AugmentationConfig, AugmentedContext

# 按需导入具体实现类
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

__all__ = [
    'ContextAugmenter',
    'AugmentationConfig',
    'AugmentedContext',
    'PromptBuilder',
    'BasicPromptBuilder',
    'StructuredPromptBuilder',
    'TemplatePromptBuilder',
    'ContextFormatter',
    'DefaultFormatter',
    'MarkdownFormatter',
    'SchemaFormatter',
    'TokenManager'
]
