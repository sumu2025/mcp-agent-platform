"""
关键词任务分析器 - 基于关键词匹配识别任务类型
"""

import re
from typing import Dict, List, Set, Tuple, Optional, Any
import logging

from .base import TaskAnalyzer, AnalysisResult
from .task_types import TaskType

# 获取日志记录器
logger = logging.getLogger(__name__)


class KeywordTaskAnalyzer(TaskAnalyzer):
    """
    基于关键词的任务分析器，通过匹配输入文本中的关键词来识别任务类型
    """
    
    def __init__(self, min_confidence: float = 0.3):
        """
        初始化关键词任务分析器
        
        Args:
            min_confidence: 最小置信度阈值，低于此值的匹配将被忽略
        """
        self.min_confidence = min_confidence
    
    def get_name(self) -> str:
        """
        获取分析器名称
        
        Returns:
            分析器名称
        """
        return "keyword_analyzer"
    
    def get_description(self) -> str:
        """
        获取分析器描述
        
        Returns:
            分析器描述
        """
        return "基于关键词匹配的任务分析器，通过识别文本中的关键词来确定任务类型"
    
    def get_priority(self) -> int:
        """
        获取分析器优先级（数值越小优先级越高）
        
        Returns:
            分析器优先级
        """
        return 100  # 关键词分析是基础分析，优先级较低
    
    def analyze(self, text: str) -> AnalysisResult:
        """
        分析输入文本，识别任务类型
        
        Args:
            text: 用户输入文本
            
        Returns:
            任务分析结果
        """
        # 文本预处理
        normalized_text = self._preprocess_text(text)
        
        # 分析每种任务类型的匹配度
        task_scores = {}
        match_details = {}
        
        # 对每种任务类型进行匹配分析
        for task_type in TaskType:
            keywords = TaskType.get_keywords(task_type)
            score, matches = self._calculate_match_score(normalized_text, keywords)
            
            if score > self.min_confidence:
                task_scores[task_type] = score
                match_details[task_type.name] = {
                    "matched_keywords": list(matches),
                    "score": score
                }
        
        # 如果没有足够的匹配，默认为一般聊天
        if not task_scores:
            logger.info("No strong task type match found, defaulting to GENERAL_CHAT")
            return AnalysisResult(
                TaskType.GENERAL_CHAT, 
                0.5,
                {"reason": "No specific task type identified, defaulting to general chat"}
            )
        
        # 找出最佳匹配
        best_task_type = max(task_scores.items(), key=lambda x: x[1])[0]
        confidence = task_scores[best_task_type]
        
        logger.info(f"Identified task type: {best_task_type.name} with confidence: {confidence:.2f}")
        
        return AnalysisResult(
            best_task_type,
            confidence,
            {
                "match_details": match_details,
                "analyzed_text": normalized_text[:100] + "..." if len(normalized_text) > 100 else normalized_text
            }
        )
    
    def _preprocess_text(self, text: str) -> str:
        """
        预处理输入文本
        
        Args:
            text: 原始输入文本
            
        Returns:
            预处理后的文本
        """
        # 转为小写
        text = text.lower()
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_match_score(self, text: str, keywords: Set[str]) -> Tuple[float, Set[str]]:
        """
        计算文本与关键词集合的匹配分数
        
        Args:
            text: 预处理后的文本
            keywords: 关键词集合
            
        Returns:
            匹配分数（0.0-1.0）和匹配到的关键词集合
        """
        if not keywords:
            return 0.0, set()
        
        # 找出匹配的关键词
        matched_keywords = set()
        for keyword in keywords:
            # 使用单词边界匹配以提高准确性
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text, re.IGNORECASE) or keyword in text:
                matched_keywords.add(keyword)
        
        # 计算分数
        # 分数基于匹配关键词的数量和比例
        if not matched_keywords:
            return 0.0, set()
        
        # 计算基本匹配率
        match_ratio = len(matched_keywords) / len(keywords)
        
        # 应用加权计算 - 匹配越多关键词分数越高，但有上限
        score = min(0.3 + match_ratio * 0.7, 1.0)
        
        return score, matched_keywords
