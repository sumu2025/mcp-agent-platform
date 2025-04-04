"""
提示构建器 - 负责构建RAG提示
"""

import os
import string
import json
import re
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from datetime import datetime
from jinja2 import Template, Environment, FileSystemLoader

# 导入依赖模块
from .base import AugmentationMode
from ..retrieval import RetrievalResult

# 设置日志
logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    提示构建器基类，负责构建RAG提示
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        初始化提示构建器
        
        Args:
            templates_dir: 模板目录路径，如果为None则使用默认模板
        """
        self.templates_dir = templates_dir
        self.templates = {}
        self.jinja_env = None
        
        # 初始化Jinja2环境（如果有模板目录）
        if templates_dir and os.path.exists(templates_dir):
            self.jinja_env = Environment(
                loader=FileSystemLoader(templates_dir),
                autoescape=False
            )
    
    def load_template(self, name: str) -> Optional[str]:
        """
        加载模板
        
        Args:
            name: 模板名称
            
        Returns:
            模板内容，如果不存在则为None
        """
        # 如果已缓存，直接返回
        if name in self.templates:
            return self.templates[name]
            
        # 如果有Jinja2环境，尝试加载模板
        if self.jinja_env:
            try:
                template = self.jinja_env.get_template(f"{name}.j2")
                self.templates[name] = template.render
                return template.render
            except Exception as e:
                logger.warning(f"加载模板 {name} 失败: {str(e)}")
                
        return None
    
    def register_template(self, name: str, template_content: str) -> None:
        """
        注册模板
        
        Args:
            name: 模板名称
            template_content: 模板内容
        """
        try:
            template = Template(template_content)
            self.templates[name] = template.render
            logger.info(f"已注册模板: {name}")
        except Exception as e:
            logger.error(f"注册模板 {name} 失败: {str(e)}")
    
    def build_system_prompt(self, 
                          retrieval_text: str, 
                          query: str,
                          template: Optional[str] = None) -> str:
        """
        构建系统提示
        
        Args:
            retrieval_text: 检索结果文本
            query: 用户查询
            template: 模板内容或名称，如果为None则使用默认模板
            
        Returns:
            系统提示文本
        """
        # 默认模板
        default_template = (
            "使用以下参考信息回答用户问题。如果参考信息无法回答问题，请基于你的知识谨慎回答。\n\n"
            "{retrieval_results}\n\n"
            "回答时，请引用相关信息来源。如果无法从参考信息中找到答案，请明确说明。"
        )
        
        # 准备模板
        template_render = None
        
        # 如果提供了模板名称，尝试加载
        if template and not template.startswith("{") and not "\n" in template:
            template_render = self.load_template(template)
            
        # 如果没有找到模板，使用提供的模板内容或默认模板
        if not template_render:
            template_content = template or default_template
            try:
                template_obj = Template(template_content)
                template_render = template_obj.render
            except Exception as e:
                logger.error(f"解析模板失败: {str(e)}，使用默认模板")
                template_obj = Template(default_template)
                template_render = template_obj.render
        
        # 渲染模板
        try:
            prompt = template_render(
                retrieval_results=retrieval_text,
                query=query
            )
            return prompt
        except Exception as e:
            logger.error(f"渲染模板失败: {str(e)}，使用简单拼接")
            # 简单拼接
            return f"{default_template.format(retrieval_results=retrieval_text)}"
    
    def build_user_prompt(self, 
                         query: str,
                         template: Optional[str] = None) -> str:
        """
        构建用户提示
        
        Args:
            query: 用户查询
            template: 模板内容或名称，如果为None则直接使用查询
            
        Returns:
            用户提示文本
        """
        # 如果没有模板，直接返回查询
        if not template:
            return query
            
        # 准备模板
        template_render = None
        
        # 如果提供了模板名称，尝试加载
        if not template.startswith("{") and not "\n" in template:
            template_render = self.load_template(template)
            
        # 如果没有找到模板，使用提供的模板内容
        if not template_render:
            try:
                template_obj = Template(template)
                template_render = template_obj.render
            except Exception as e:
                logger.error(f"解析模板失败: {str(e)}，直接使用查询")
                return query
        
        # 渲染模板
        try:
            prompt = template_render(query=query)
            return prompt
        except Exception as e:
            logger.error(f"渲染模板失败: {str(e)}，直接使用查询")
            return query
    
    def format_retrieval_result(self, 
                              result: RetrievalResult, 
                              template: Optional[str] = None,
                              include_metadata: bool = True,
                              metadata_fields: Optional[List[str]] = None) -> str:
        """
        格式化单个检索结果
        
        Args:
            result: 检索结果
            template: 模板内容或名称，如果为None则使用默认模板
            include_metadata: 是否包含元数据
            metadata_fields: 要包含的元数据字段，如果为None则包含所有字段
            
        Returns:
            格式化后的文本
        """
        # 默认模板
        default_template = "[{index}] {content}"
        if include_metadata:
            default_template = "[{index}] {content}\n来源: {metadata}"
            
        # 准备模板
        template_render = None
        
        # 如果提供了模板名称，尝试加载
        if template and not template.startswith("{") and not "\n" in template:
            template_render = self.load_template(template)
            
        # 如果没有找到模板，使用提供的模板内容或默认模板
        if not template_render:
            template_content = template or default_template
            try:
                template_obj = Template(template_content)
                template_render = template_obj.render
            except Exception as e:
                logger.error(f"解析模板失败: {str(e)}，使用默认模板")
                template_obj = Template(default_template)
                template_render = template_obj.render
        
        # 准备元数据
        metadata_str = ""
        if include_metadata and result.metadata:
            # 过滤元数据字段
            filtered_metadata = {}
            if metadata_fields:
                for field in metadata_fields:
                    if field in result.metadata:
                        filtered_metadata[field] = result.metadata[field]
            else:
                filtered_metadata = result.metadata
                
            # 格式化元数据
            try:
                metadata_items = []
                for k, v in filtered_metadata.items():
                    if v is not None:
                        metadata_items.append(f"{k}: {v}")
                metadata_str = ", ".join(metadata_items)
            except Exception as e:
                logger.warning(f"格式化元数据失败: {str(e)}")
                metadata_str = str(filtered_metadata)
        
        # 渲染模板
        try:
            formatted = template_render(
                content=result.text,
                metadata=metadata_str,
                score=result.score,
                id=result.id,
                source=result.source,
                index=0  # 将在外部更新
            )
            return formatted
        except Exception as e:
            logger.error(f"渲染模板失败: {str(e)}，使用简单格式")
            # 简单格式
            if include_metadata and metadata_str:
                return f"{result.text}\n来源: {metadata_str}"
            else:
                return result.text
    
    def format_retrieval_results(self, 
                                retrieval_results: List[RetrievalResult], 
                                template: Optional[str] = None,
                                include_metadata: bool = True,
                                metadata_fields: Optional[List[str]] = None) -> str:
        """
        格式化检索结果列表
        
        Args:
            retrieval_results: 检索结果列表
            template: 单个结果的模板
            include_metadata: 是否包含元数据
            metadata_fields: 要包含的元数据字段，如果为None则包含所有字段
            
        Returns:
            格式化后的文本
        """
        if not retrieval_results:
            return ""
            
        formatted_results = []
        for i, result in enumerate(retrieval_results):
            # 格式化单个结果
            formatted = self.format_retrieval_result(
                result, 
                template=template, 
                include_metadata=include_metadata,
                metadata_fields=metadata_fields
            )
            
            # 替换索引
            formatted = formatted.replace("{index}", str(i+1))
            formatted = re.sub(r'\[0\]', f'[{i+1}]', formatted)
            
            formatted_results.append(formatted)
            
        # 合并结果
        return "\n\n".join(formatted_results)


class BasicPromptBuilder(PromptBuilder):
    """
    基本提示构建器，使用简单格式构建提示
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        初始化基本提示构建器
        
        Args:
            templates_dir: 模板目录路径，如果为None则使用默认模板
        """
        super().__init__(templates_dir)
        
        # 注册默认模板
        self.register_template("basic_system", 
            "使用以下参考信息回答用户问题：\n\n"
            "{retrieval_results}\n\n"
            "如果参考信息不足以回答问题，请基于你的知识谨慎回答。"
        )
        
        self.register_template("basic_result", 
            "[{index}] {content}\n"
            "来源: {metadata}"
        )


class StructuredPromptBuilder(PromptBuilder):
    """
    结构化提示构建器，使用结构化格式构建提示
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        初始化结构化提示构建器
        
        Args:
            templates_dir: 模板目录路径，如果为None则使用默认模板
        """
        super().__init__(templates_dir)
        
        # 注册默认模板
        self.register_template("structured_system", 
            "你是一个知识助手，请根据以下参考资料回答用户的问题。\n\n"
            "参考资料:\n{retrieval_results}\n\n"
            "回答要求:\n"
            "1. 基于参考资料提供准确信息\n"
            "2. 引用相关的参考编号 [X]\n"
            "3. 如果参考资料不足，明确说明并谨慎地给出你自己的回答\n"
            "4. 保持回答简洁、专业且条理清晰\n"
        )
        
        self.register_template("structured_result", 
            "[{index}] {content}\n\n"
            "元数据: {metadata}"
        )


class TemplatePromptBuilder(PromptBuilder):
    """
    模板提示构建器，使用自定义模板构建提示
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        初始化模板提示构建器
        
        Args:
            templates_dir: 模板目录路径，如果为None则使用默认模板
        """
        super().__init__(templates_dir)
        
        # 加载内置模板
        self._load_builtin_templates()
    
    def _load_builtin_templates(self) -> None:
        """加载内置模板"""
        # 学术风格模板
        self.register_template("academic", 
            "你是一位学术助手，请根据所提供的参考资料，以学术和专业的方式回答用户的问题。\n\n"
            "参考资料:\n"
            "{retrieval_results}\n\n"
            "注意事项:\n"
            "- 使用正式的学术语言和术语\n"
            "- 提供基于证据的回答，引用相关参考资料 [X]\n"
            "- 当存在不确定性时，请明确指出\n"
            "- 保持客观中立，避免主观判断\n"
            "- 如需使用学术观点，请明确归因\n"
        )
        
        # 商业风格模板
        self.register_template("business", 
            "你是一位商业顾问，请根据提供的参考资料，简明扼要地回答用户的商业相关问题。\n\n"
            "参考资料:\n"
            "{retrieval_results}\n\n"
            "回答指南:\n"
            "- 聚焦实用性和可行性\n"
            "- 使用清晰的商业语言\n"
            "- 尽可能提供具体的观点或建议\n"
            "- 引用相关参考资料 [X]\n"
            "- 在适当的情况下，提供简短的行动步骤\n"
        )
        
        # 教学风格模板
        self.register_template("educational", 
            "你是一位教育助手，请根据提供的参考资料，以易于理解的方式回答用户的问题。\n\n"
            "参考材料:\n"
            "{retrieval_results}\n\n"
            "教学指南:\n"
            "- 使用简明易懂的语言\n"
            "- 分步骤解释复杂概念\n"
            "- 提供具体示例辅助理解\n"
            "- 在引用参考资料时，使用 [X] 标记\n"
            "- 当信息不完整时，明确指出并提供补充解释\n"
        )
        
        # 代码助手模板
        self.register_template("code_assistant", 
            "你是一位编程助手，请根据提供的参考资料，回答用户的编程相关问题。\n\n"
            "参考资料:\n"
            "{retrieval_results}\n\n"
            "回答准则:\n"
            "- 提供清晰、可运行的代码示例\n"
            "- 解释代码的关键部分和工作原理\n"
            "- 考虑代码的效率和最佳实践\n"
            "- 引用参考资料 [X] 中的相关部分\n"
            "- 如果有多种解决方案，请简要说明优缺点\n"
        )
        
        # 医疗咨询模板
        self.register_template("medical", 
            "你是一位医学信息助手，请根据提供的参考资料，回答用户的医疗相关问题。\n\n"
            "参考资料:\n"
            "{retrieval_results}\n\n"
            "重要提示:\n"
            "- 你提供的是医学信息，不是医疗建议\n"
            "- 始终建议用户咨询合格的医疗专业人员\n"
            "- 基于参考资料 [X] 提供准确信息\n"
            "- 使用清晰、专业但易于理解的语言\n"
            "- 避免做出诊断或治疗建议\n"
            "- 当信息不足或不确定时，明确说明\n"
        )
