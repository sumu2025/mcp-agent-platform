"""
提示构建模块 - 负责构建结构化提示和管理令牌数量
"""

import re
import os
import json
import yaml
from string import Template
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from datetime import datetime

# 导入依赖模块
from ..retrieval import RetrievalResult

# 设置日志
logger = logging.getLogger(__name__)


class TokenManager:
    """
    令牌管理器，负责令牌计数和内容优化
    """
    
    def __init__(self, 
                model_name: str = "default",
                conservative: bool = True):
        """
        初始化令牌管理器
        
        Args:
            model_name: 模型名称，用于选择计数方法
            conservative: 是否使用保守计数（高估令牌数）
        """
        self.model_name = model_name
        self.conservative = conservative
        
        # 加载模型特定的令牌化器（如果可用）
        self.tokenizer = self._load_tokenizer(model_name)
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的令牌数量
        
        Args:
            text: 文本内容
            
        Returns:
            令牌数量
        """
        if self.tokenizer:
            # 使用特定模型的令牌化器
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"使用特定令牌化器失败: {str(e)}，回退到估算方法")
        
        # 回退到估算方法
        return self._estimate_tokens(text)
    
    def truncate_to_token_limit(self, 
                              text: str, 
                              max_tokens: int) -> str:
        """
        截断文本以符合令牌限制
        
        Args:
            text: 文本内容
            max_tokens: 最大令牌数
            
        Returns:
            截断后的文本
        """
        if self.count_tokens(text) <= max_tokens:
            return text
            
        if self.tokenizer:
            # 使用特定模型的令牌化器截断
            try:
                tokens = self.tokenizer.encode(text)
                truncated_tokens = tokens[:max_tokens]
                return self.tokenizer.decode(truncated_tokens)
            except Exception as e:
                logger.warning(f"使用特定令牌化器截断失败: {str(e)}，回退到估算方法")
        
        # 回退到估算方法
        return self._truncate_by_estimate(text, max_tokens)
    
    def distribute_tokens(self, 
                         parts: Dict[str, str], 
                         weights: Dict[str, float],
                         max_tokens: int) -> Dict[str, str]:
        """
        按权重分配令牌数量
        
        Args:
            parts: 文本部分字典
            weights: 各部分的权重
            max_tokens: 最大总令牌数
            
        Returns:
            调整后的文本部分字典
        """
        # 计算每个部分的当前令牌数
        current_tokens = {k: self.count_tokens(v) for k, v in parts.items()}
        total_current = sum(current_tokens.values())
        
        # 如果总数已经在限制内，直接返回
        if total_current <= max_tokens:
            return parts
        
        # 标准化权重
        total_weight = sum(weights.values())
        normalized_weights = {k: w / total_weight for k, w in weights.items()}
        
        # 计算每个部分的目标令牌数
        target_tokens = {k: int(max_tokens * normalized_weights.get(k, 0.1)) for k in parts}
        
        # 确保每个部分至少有一些令牌
        min_tokens = 50
        for k in target_tokens:
            if target_tokens[k] < min_tokens:
                target_tokens[k] = min_tokens
        
        # 调整目标令牌数，确保总和不超过最大值
        total_target = sum(target_tokens.values())
        if total_target > max_tokens:
            scale_factor = max_tokens / total_target
            target_tokens = {k: int(t * scale_factor) for k, t in target_tokens.items()}
        
        # 截断每个部分
        adjusted_parts = {}
        for k, v in parts.items():
            if k in target_tokens:
                adjusted_parts[k] = self.truncate_to_token_limit(v, target_tokens[k])
            else:
                # 对于没有指定权重的部分，分配少量令牌
                adjusted_parts[k] = self.truncate_to_token_limit(v, min_tokens)
        
        return adjusted_parts
    
    def optimize_context(self, 
                        context: str, 
                        max_tokens: int,
                        prioritize_start: bool = True) -> str:
        """
        优化上下文以符合令牌限制，保留最重要的部分
        
        Args:
            context: 上下文文本
            max_tokens: 最大令牌数
            prioritize_start: 是否优先保留开头部分
            
        Returns:
            优化后的上下文
        """
        if self.count_tokens(context) <= max_tokens:
            return context
            
        # 将上下文分成段落
        paragraphs = re.split(r'\n\s*\n', context)
        
        # 如果优先保留开头，反转段落列表
        if not prioritize_start:
            paragraphs.reverse()
        
        # 按顺序添加段落，直到达到令牌限制
        optimized = []
        total_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # 如果添加此段落会超出限制，跳过
            if total_tokens + para_tokens > max_tokens:
                continue
                
            optimized.append(para)
            total_tokens += para_tokens
        
        # 如果优先保留开头，反转回来
        if not prioritize_start:
            optimized.reverse()
        
        # 连接段落
        result = "\n\n".join(optimized)
        
        # 如果仍然超出限制，截断
        if self.count_tokens(result) > max_tokens:
            result = self.truncate_to_token_limit(result, max_tokens)
            
        return result
    
    def _load_tokenizer(self, model_name: str):
        """
        加载模型特定的令牌化器
        
        Args:
            model_name: 模型名称
            
        Returns:
            令牌化器对象，如果不可用则为None
        """
        # 尝试加载对应模型的令牌化器
        try:
            if "gpt" in model_name.lower():
                import tiktoken
                return tiktoken.encoding_for_model(model_name)
            elif "claude" in model_name.lower():
                # Claude没有官方令牌化器，使用估算
                return None
            elif "llama" in model_name.lower() or "mistral" in model_name.lower():
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(model_name)
            else:
                # 未知模型，使用估算
                return None
        except Exception as e:
            logger.warning(f"加载令牌化器失败: {str(e)}")
            return None
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的令牌数量
        
        Args:
            text: 文本内容
            
        Returns:
            估计的令牌数量
        """
        # 使用简单的启发式方法估算令牌数
        # 对于英文，约4个字符≈1个令牌
        # 对于中文，约1.5个字符≈1个令牌
        
        # 计算中文字符数
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        
        # 计算非中文字符数
        non_chinese_chars = len(text) - chinese_chars
        
        # 估算令牌数
        estimated_tokens = chinese_chars / 1.5 + non_chinese_chars / 4
        
        # 如果是保守计算，增加10%
        if self.conservative:
            estimated_tokens *= 1.1
            
        return int(estimated_tokens)
    
    def _truncate_by_estimate(self, 
                            text: str, 
                            max_tokens: int) -> str:
        """
        使用估算方法截断文本
        
        Args:
            text: 文本内容
            max_tokens: 最大令牌数
            
        Returns:
            截断后的文本
        """
        # 计算当前估计令牌数
        current_tokens = self._estimate_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
            
        # 计算截断比例
        ratio = max_tokens / current_tokens
        
        # 确定截断位置
        # 保守估计，截断稍多一些
        cut_index = int(len(text) * ratio * 0.95)
        
        # 确保截断在字符边界
        while cut_index > 0 and '\u4e00' <= text[cut_index - 1] <= '\u9fff':
            cut_index -= 1
            
        truncated = text[:cut_index] + "..."
        
        # 验证截断后的令牌数
        if self._estimate_tokens(truncated) > max_tokens:
            # 如果仍然超出，进一步截断
            return self._truncate_by_estimate(truncated, max_tokens)
            
        return truncated


class PromptTemplate:
    """
    提示模板，用于构建结构化提示
    """
    
    def __init__(self, template: str):
        """
        初始化提示模板
        
        Args:
            template: 模板字符串，使用 ${变量名} 语法
        """
        self.template = template
        self._template = Template(template)
    
    def render(self, variables: Dict[str, str]) -> str:
        """
        渲染模板
        
        Args:
            variables: 变量字典
            
        Returns:
            渲染后的文本
        """
        try:
            return self._template.substitute(variables)
        except KeyError as e:
            # 处理缺少的变量
            logger.warning(f"渲染模板时缺少变量: {str(e)}")
            # 尝试用空字符串替换缺少的变量
            result = self.template
            for key, value in variables.items():
                result = result.replace(f"${{{key}}}", value)
            return result
    
    @classmethod
    def from_file(cls, file_path: str) -> 'PromptTemplate':
        """
        从文件加载模板
        
        Args:
            file_path: 模板文件路径
            
        Returns:
            提示模板对象
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            template = f.read()
            
        return cls(template)
    
    @classmethod
    def get_default_templates(cls) -> Dict[str, 'PromptTemplate']:
        """
        获取默认模板集合
        
        Returns:
            模板字典
        """
        templates = {
            "basic": cls("""
${system_message}

${context}

${query_prefix}
${query}
${query_suffix}
"""),
            "structured": cls("""
${system_message}

下面是一些相关的上下文信息:

${context}

${query_prefix}
${query}
${query_suffix}
"""),
            "qa": cls("""
${system_message}

问题: ${query}

相关信息:
${context}

请根据上述信息回答问题。${query_suffix}
"""),
            "chat": cls("""
${system_message}

用户: ${query}

相关上下文:
${context}

助手:
"""),
            "instruction": cls("""
${system_message}

指令: 根据提供的上下文信息，${query}

上下文信息:
${context}

${query_suffix}
""")
        }
        
        return templates


class PromptBuilder:
    """
    提示构建器，用于构建结构化提示
    """
    
    def __init__(self, 
                template: Optional[Union[str, PromptTemplate]] = None,
                token_manager: Optional[TokenManager] = None):
        """
        初始化提示构建器
        
        Args:
            template: 提示模板或模板字符串，如果为None则使用默认模板
            token_manager: 令牌管理器，如果为None则创建默认管理器
        """
        # 设置模板
        if template is None:
            # 使用默认模板
            self.templates = PromptTemplate.get_default_templates()
            self.template = self.templates["structured"]
        elif isinstance(template, str):
            self.template = PromptTemplate(template)
        else:
            self.template = template
            
        # 设置令牌管理器
        self.token_manager = token_manager or TokenManager()
    
    def build_prompt(self, 
                    query: str, 
                    context: str,
                    system_message: str = "",
                    query_prefix: str = "",
                    query_suffix: str = "",
                    max_tokens: Optional[int] = None) -> str:
        """
        构建完整提示
        
        Args:
            query: 用户查询
            context: 上下文内容
            system_message: 系统消息
            query_prefix: 查询前缀
            query_suffix: 查询后缀
            max_tokens: 最大令牌数
            
        Returns:
            完整提示
        """
        # 准备变量
        variables = {
            "query": query,
            "context": context,
            "system_message": system_message,
            "query_prefix": query_prefix,
            "query_suffix": query_suffix
        }
        
        # 渲染模板
        prompt = self.template.render(variables)
        
        # 如果有令牌限制，截断提示
        if max_tokens and self.token_manager:
            # 计算每个部分的权重
            weights = {
                "context": 0.7,  # 上下文占70%
                "query": 0.2,    # 查询占20%
                "system_message": 0.1  # 系统消息占10%
            }
            
            # 分配令牌数量
            adjusted_vars = self.token_manager.distribute_tokens(
                {k: v for k, v in variables.items() if k in weights},
                weights,
                max_tokens
            )
            
            # 更新变量
            variables.update(adjusted_vars)
            
            # 重新渲染模板
            prompt = self.template.render(variables)
        
        return prompt
    
    def format_documents(self, 
                        retrieval_results: List[RetrievalResult],
                        document_prefix: str = "",
                        document_separator: str = "\n\n",
                        include_metadata: bool = True,
                        indexing_style: str = "numbered",
                        max_documents: Optional[int] = None) -> str:
        """
        格式化文档列表
        
        Args:
            retrieval_results: 检索结果列表
            document_prefix: 文档前缀
            document_separator: 文档分隔符
            include_metadata: 是否包含元数据
            indexing_style: 索引方式 (none, numbered, lettered)
            max_documents: 最大文档数量
            
        Returns:
            格式化后的文档文本
        """
        # 限制文档数量
        if max_documents:
            results = retrieval_results[:max_documents]
        else:
            results = retrieval_results
            
        # 格式化每个文档
        formatted_docs = []
        
        for i, result in enumerate(results):
            # 创建索引
            if indexing_style == "none":
                index = ""
            elif indexing_style == "lettered":
                index = f"[{chr(65 + i)}] "
            else:  # numbered
                index = f"[{i + 1}] "
                
            # 格式化文档
            doc_text = f"{index}{document_prefix}{result.text}"
            
            # 添加元数据
            if include_metadata and result.metadata:
                metadata_text = self.format_metadata(result.metadata)
                doc_text += f"\n{metadata_text}"
                
            formatted_docs.append(doc_text)
            
        # 连接文档
        return document_separator.join(formatted_docs)
    
    def format_metadata(self, 
                       metadata: Dict[str, Any],
                       format_type: str = "yaml") -> str:
        """
        格式化元数据
        
        Args:
            metadata: 元数据字典
            format_type: 格式类型 (plain, yaml, json)
            
        Returns:
            格式化后的元数据文本
        """
        if not metadata:
            return ""
            
        if format_type == "json":
            # JSON格式
            try:
                return json.dumps(metadata, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"格式化JSON元数据失败: {str(e)}")
                format_type = "plain"  # 回退到纯文本格式
                
        if format_type == "yaml":
            # YAML格式
            try:
                return yaml.dump(metadata, allow_unicode=True, sort_keys=False)
            except Exception as e:
                logger.warning(f"格式化YAML元数据失败: {str(e)}")
                format_type = "plain"  # 回退到纯文本格式
                
        # 纯文本格式
        lines = []
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                # 处理嵌套结构
                value = str(value)
            lines.append(f"{key}: {value}")
            
        return "\n".join(lines)
    
    def change_template(self, 
                      template: Union[str, PromptTemplate],
                      template_type: Optional[str] = None) -> None:
        """
        更改提示模板
        
        Args:
            template: 新模板或模板类型
            template_type: 模板类型，如果template是字符串且不是模板类型
        """
        if isinstance(template, str):
            if template in self.templates:
                # 使用预定义模板
                self.template = self.templates[template]
            elif template_type:
                # 创建新模板
                self.template = PromptTemplate(template)
        else:
            # 直接使用PromptTemplate对象
            self.template = template
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的令牌数量
        
        Args:
            text: 文本内容
            
        Returns:
            令牌数量
        """
        if self.token_manager:
            return self.token_manager.count_tokens(text)
        else:
            # 简单估算
            words = text.split()
            return int(len(words) / 1.3)
