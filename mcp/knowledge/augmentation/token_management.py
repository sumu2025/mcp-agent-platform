"""
Token管理 - 负责管理LLM Token使用
"""

import re
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
import tiktoken

# 设置日志
logger = logging.getLogger(__name__)


class TokenManager:
    """
    Token管理器，负责计数和管理LLM Token使用
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        初始化Token管理器
        
        Args:
            model_name: 模型名称，用于选择正确的分词器
        """
        self.model_name = model_name
        self.encoder = None
        
        # 不同模型的token计数映射
        self.model_map = {
            # OpenAI模型
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-4": "cl100k_base",
            "text-embedding-ada-002": "cl100k_base",
            
            # Claude模型 (使用OpenAI的分词器估计)
            "claude-2": "cl100k_base",
            "claude-instant-1": "cl100k_base",
            
            # 本地模型
            "llama": "cl100k_base",
            "mistral": "cl100k_base"
        }
        
        # 初始化分词器
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self) -> None:
        """初始化分词器"""
        try:
            # 确定正确的编码器名称
            encoding_name = self.model_map.get(self.model_name, "cl100k_base")
            
            # 获取编码器
            self.encoder = tiktoken.get_encoding(encoding_name)
            logger.info(f"成功初始化tokenizer: {encoding_name}")
            
        except Exception as e:
            logger.warning(f"初始化tokenizer失败: {str(e)}，将使用简单估计")
            self.encoder = None
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            token数量
        """
        if not text:
            return 0
            
        # 使用tiktoken（如果可用）
        if self.encoder:
            try:
                tokens = self.encoder.encode(text)
                return len(tokens)
            except Exception as e:
                logger.warning(f"使用tiktoken计算token失败: {str(e)}，使用简单估计")
                
        # 简单估计: 单词数量的1.3倍
        return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        简单估计token数量（如果tiktoken不可用）
        
        Args:
            text: 输入文本
            
        Returns:
            估计的token数量
        """
        # 拆分为单词
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        # 估计标点符号和空格
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        
        # 估计token数量: 单词数量的1.3倍 + 标点符号数量
        estimated_tokens = int(word_count * 1.3) + punctuation_count
        
        return estimated_tokens
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """
        截断文本至最大token数
        
        Args:
            text: 输入文本
            max_tokens: 最大token数
            
        Returns:
            截断后的文本
        """
        if not text:
            return ""
            
        # 检查是否需要截断
        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text
            
        # 使用tiktoken精确截断（如果可用）
        if self.encoder:
            try:
                tokens = self.encoder.encode(text)
                
                # 截断token并解码
                truncated_tokens = tokens[:max_tokens]
                truncated_text = self.encoder.decode(truncated_tokens)
                
                return truncated_text
            except Exception as e:
                logger.warning(f"使用tiktoken截断文本失败: {str(e)}，使用简单截断")
                
        # 简单截断: 按比例估计
        ratio = max_tokens / current_tokens
        char_limit = int(len(text) * ratio)
        
        # 尝试在句子边界截断
        sentences = re.split(r'(?<=[.!?])\s+', text[:char_limit])
        if sentences:
            # 去掉最后一个可能不完整的句子
            return ' '.join(sentences[:-1]) + '...'
        
        # 如果无法按句子截断，直接截断
        return text[:char_limit] + '...'
    
    def distribute_tokens(self, 
                         total_tokens: int,
                         sections: Dict[str, str],
                         priorities: List[str]) -> Dict[str, str]:
        """
        根据优先级分配token给不同部分
        
        Args:
            total_tokens: 总可用token数
            sections: 部分名称到文本的映射
            priorities: 优先级顺序（高到低）
            
        Returns:
            调整后的部分
        """
        if not sections:
            return {}
            
        # 计算每个部分的token数
        token_counts = {
            name: self.count_tokens(text)
            for name, text in sections.items()
        }
        
        total_current = sum(token_counts.values())
        
        # 如果总数已经在限制之内，无需调整
        if total_current <= total_tokens:
            return sections.copy()
            
        # 需要减少的token数
        to_reduce = total_current - total_tokens
        
        # 调整后的部分
        adjusted = sections.copy()
        
        # 按低优先级到高优先级的顺序尝试减少
        reversed_priorities = priorities.copy()
        reversed_priorities.reverse()
        
        for section_name in reversed_priorities:
            if section_name not in adjusted:
                continue
                
            section_text = adjusted[section_name]
            section_tokens = token_counts[section_name]
            
            if to_reduce >= section_tokens:
                # 完全移除此部分
                to_reduce -= section_tokens
                adjusted[section_name] = ""
            else:
                # 部分截断此部分
                max_tokens_for_section = section_tokens - to_reduce
                adjusted[section_name] = self.truncate_text(
                    section_text, max_tokens_for_section
                )
                to_reduce = 0
                
            # 如果已经减少足够的token，退出
            if to_reduce <= 0:
                break
                
        return adjusted
    
    def trim_retrieval_results(self, 
                              texts: List[str],
                              max_tokens: int,
                              preserve_count: Optional[int] = None) -> List[str]:
        """
        裁剪检索结果，确保总token不超过限制
        
        Args:
            texts: 检索结果文本列表
            max_tokens: 最大token数
            preserve_count: 保证保留的结果数量，如果为None则尽量保留所有
            
        Returns:
            裁剪后的文本列表
        """
        if not texts:
            return []
            
        # 计算token数
        token_counts = [self.count_tokens(text) for text in texts]
        total_tokens = sum(token_counts)
        
        # 如果总数已经在限制之内，无需调整
        if total_tokens <= max_tokens:
            return texts.copy()
            
        # 设置保留数量
        if preserve_count is None or preserve_count > len(texts):
            preserve_count = len(texts)
            
        # 计算每个结果的平均可用token
        tokens_per_result = max_tokens // preserve_count
        
        # 裁剪结果
        trimmed = []
        remaining_tokens = max_tokens
        
        for i, text in enumerate(texts):
            # 如果已达到保留数量，退出
            if i >= preserve_count:
                break
                
            current_tokens = token_counts[i]
            
            if current_tokens <= tokens_per_result:
                # 无需裁剪
                trimmed.append(text)
                remaining_tokens -= current_tokens
            else:
                # 需要裁剪
                max_tokens_for_text = min(tokens_per_result, remaining_tokens)
                trimmed_text = self.truncate_text(text, max_tokens_for_text)
                trimmed.append(trimmed_text)
                
                # 更新剩余token
                used_tokens = self.count_tokens(trimmed_text)
                remaining_tokens -= used_tokens
            
            # 重新计算每个结果的平均可用token
            if i < preserve_count - 1:  # 不是最后一个结果
                remaining_results = preserve_count - (i + 1)
                if remaining_results > 0:
                    tokens_per_result = remaining_tokens // remaining_results
                
        return trimmed
