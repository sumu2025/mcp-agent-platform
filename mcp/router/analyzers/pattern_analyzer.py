"""
模式匹配任务分析器 - 基于正则表达式和模式匹配识别任务类型
"""

import re
from typing import Dict, List, Tuple, Set, Optional, Any
import logging

from .base import TaskAnalyzer, AnalysisResult
from .task_types import TaskType

# 获取日志记录器
logger = logging.getLogger(__name__)


class PatternTaskAnalyzer(TaskAnalyzer):
    """
    基于模式匹配的任务分析器，使用正则表达式和语法模式识别任务类型
    """
    
    def __init__(self, min_confidence: float = 0.5):
        """
        初始化模式匹配任务分析器
        
        Args:
            min_confidence: 最小置信度阈值，低于此值的匹配将被忽略
        """
        self.min_confidence = min_confidence
        self._init_patterns()
    
    def get_name(self) -> str:
        """
        获取分析器名称
        
        Returns:
            分析器名称
        """
        return "pattern_analyzer"
    
    def get_description(self) -> str:
        """
        获取分析器描述
        
        Returns:
            分析器描述
        """
        return "基于正则表达式和语法模式的任务分析器，通过识别特定结构来确定任务类型"
    
    def get_priority(self) -> int:
        """
        获取分析器优先级（数值越小优先级越高）
        
        Returns:
            分析器优先级
        """
        return 80  # 模式分析比关键词分析更精确，优先级更高
    
    def _init_patterns(self):
        """
        初始化任务类型模式
        """
        # 为不同任务类型定义特定的正则表达式模式
        self.task_patterns = {
            TaskType.CREATIVE_WRITING: [
                # 写作请求模式
                r'(?:写|写一[篇个]|创作|撰写|帮我写).*?(?:故事|小说|文章|诗歌|剧本|散文|日记|信|邮件)',
                r'(?:write|compose|create)\s+(?:a|an)\s+(?:story|novel|article|poem|essay|script|email|letter)'
            ],
            
            TaskType.CODE_GENERATION: [
                # 代码生成模式
                r'(?:写|创建|生成|实现|编写).*?(?:代码|函数|程序|脚本|类|模块)',
                r'(?:write|create|generate|implement|code)\s+(?:a|an|the)?\s*(?:function|program|script|class|module)'
            ],
            
            TaskType.CODE_EXPLANATION: [
                # 代码解释模式
                r'(?:解释|分析|解读|理解|注释).*?(?:代码|函数|程序|脚本|类|模块)',
                r'(?:explain|analyze|understand|comment)\s+(?:this|the|following|my)?\s*(?:code|function|program|script|class|module)'
            ],
            
            TaskType.SUMMARIZATION: [
                # 摘要请求模式
                r'(?:总结|概括|摘要|提炼).*?(?:内容|要点|文章|信息)',
                r'(?:summarize|summarise|sum\s*up|give\s*(?:a|me|the)\s*summary\s*of)'
            ],
            
            TaskType.TRANSLATION: [
                # 翻译请求模式
                r'(?:翻译|中译英|英译中|翻成|转成).*?(?:中文|英文|日文|法文|德文|翻译)',
                r'(?:translate|translation|転?の日本語を英語に|英語を日本語に)'
            ],
            
            TaskType.MATH_PROBLEM: [
                # 数学问题模式
                r'(?:计算|算|求|解).*?(?:方程|数学|概率|统计|积分|微分|导数)',
                r'(?:calculate|compute|solve|find)\s+(?:the|this|following)?\s*(?:equation|math|probability|statistics|integral|derivative)'
            ],
            
            TaskType.STRUCTURED_OUTPUT: [
                # 结构化输出请求模式
                r'(?:生成|创建|制作).*?(?:表格|列表|JSON|XML|CSV|markdown|格式化)',
                r'(?:generate|create|make|produce)\s+(?:a|an|the)?\s*(?:table|list|JSON|XML|CSV|markdown|formatted)'
            ]
        }
    
    def analyze(self, text: str) -> AnalysisResult:
        """
        分析输入文本，识别任务类型
        
        Args:
            text: 用户输入文本
            
        Returns:
            任务分析结果
        """
        # 分析每种任务类型的匹配度
        task_scores = {}
        match_details = {}
        
        # 对每种任务类型进行模式匹配
        for task_type, patterns in self.task_patterns.items():
            score, matches = self._calculate_pattern_score(text, patterns)
            
            if score > self.min_confidence:
                task_scores[task_type] = score
                match_details[task_type.name] = {
                    "matched_patterns": matches,
                    "score": score
                }
        
        # 如果没有足够的匹配，返回None表示无法确定
        if not task_scores:
            logger.info("No strong pattern match found")
            return AnalysisResult(
                TaskType.GENERAL_CHAT,  # 默认为一般对话
                0.3,  # 低置信度
                {"reason": "No specific pattern identified"}
            )
        
        # 找出最佳匹配
        best_task_type = max(task_scores.items(), key=lambda x: x[1])[0]
        confidence = task_scores[best_task_type]
        
        logger.info(f"Pattern analyzer identified task type: {best_task_type.name} with confidence: {confidence:.2f}")
        
        return AnalysisResult(
            best_task_type,
            confidence,
            {
                "match_details": match_details,
                "analyzer": "pattern_analyzer"
            }
        )
    
    def _calculate_pattern_score(self, text: str, patterns: List[str]) -> Tuple[float, List[str]]:
        """
        计算文本与模式的匹配分数
        
        Args:
            text: 输入文本
            patterns: 模式列表
            
        Returns:
            匹配分数（0.0-1.0）和匹配到的模式列表
        """
        if not patterns:
            return 0.0, []
        
        # 找出匹配的模式
        matched_patterns = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched_patterns.append(pattern)
        
        # 计算分数
        if not matched_patterns:
            return 0.0, []
        
        # 模式匹配是强匹配，单个匹配就给较高分数
        match_ratio = len(matched_patterns) / len(patterns)
        
        # 应用加权计算 - 一个匹配就有较高的基础分
        score = min(0.7 + match_ratio * 0.3, 1.0)
        
        return score, matched_patterns
