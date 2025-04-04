"""
任务类型定义 - 枚举不同类型的任务及其特点
"""

from enum import Enum, auto
from typing import Dict, List, Set, Optional


class TaskType(Enum):
    """
    任务类型枚举，表示不同类型的用户请求
    
    每种任务类型对应不同的处理策略和最适合的模型
    """
    GENERAL_CHAT = auto()           # 一般聊天对话
    CREATIVE_WRITING = auto()       # 创意写作
    CODE_GENERATION = auto()        # 代码生成
    CODE_EXPLANATION = auto()       # 代码解释
    DATA_ANALYSIS = auto()          # 数据分析
    SUMMARIZATION = auto()          # 内容摘要
    TRANSLATION = auto()            # 语言翻译
    MATH_PROBLEM = auto()           # 数学问题
    KNOWLEDGE_QA = auto()           # 知识问答
    REASONING = auto()              # 逻辑推理
    STRUCTURED_OUTPUT = auto()      # 结构化输出
    
    @classmethod
    def get_keywords(cls, task_type) -> Set[str]:
        """
        获取与特定任务类型相关的关键词
        
        Args:
            task_type: TaskType枚举值
            
        Returns:
            与该任务类型相关的关键词集合
        """
        keyword_map = {
            cls.GENERAL_CHAT: {
                "聊天", "对话", "交谈", "闲聊", "你好", "你是谁", "聊一聊",
                "chat", "talk", "hello", "hi", "hey"
            },
            cls.CREATIVE_WRITING: {
                "写一篇", "创作", "写作", "故事", "小说", "诗歌", "文章", "剧本",
                "write", "story", "novel", "poem", "article", "creative", "script"
            },
            cls.CODE_GENERATION: {
                "写代码", "编程", "实现", "函数", "类", "开发", "算法",
                "code", "program", "implement", "function", "class", "develop", "algorithm"
            },
            cls.CODE_EXPLANATION: {
                "解释代码", "分析代码", "代码什么意思", "代码如何工作", "解读", 
                "explain code", "analyze code", "how does this code work", "what does this code do"
            },
            cls.DATA_ANALYSIS: {
                "分析数据", "数据分析", "统计", "可视化", "图表", "趋势",
                "analyze data", "data analysis", "statistics", "visualization", "chart", "trend"
            },
            cls.SUMMARIZATION: {
                "总结", "摘要", "概括", "提炼", "要点", 
                "summarize", "summary", "key points", "tldr", "gist"
            },
            cls.TRANSLATION: {
                "翻译", "转换", "中译英", "英译中", "日语", "法语",
                "translate", "translation", "chinese to english", "english to chinese"
            },
            cls.MATH_PROBLEM: {
                "计算", "数学", "方程", "解方程", "概率", "统计", "公式",
                "calculate", "math", "equation", "solve", "probability", "formula"
            },
            cls.KNOWLEDGE_QA: {
                "什么是", "解释", "定义", "介绍", "为什么", "如何", "谁", "何时", "何地",
                "what is", "explain", "define", "introduce", "why", "how", "who", "when", "where"
            },
            cls.REASONING: {
                "推理", "逻辑", "假设", "推断", "如果", "那么", "分析", "判断",
                "reason", "logic", "assume", "infer", "if", "then", "analyze", "deduce"
            },
            cls.STRUCTURED_OUTPUT: {
                "表格", "列表", "json", "xml", "格式化", "结构化", "分类", 
                "table", "list", "format", "structured", "categorize", "organize"
            }
        }
        
        return keyword_map.get(task_type, set())
    
    @classmethod
    def get_description(cls, task_type) -> str:
        """
        获取任务类型的描述
        
        Args:
            task_type: TaskType枚举值
            
        Returns:
            任务类型描述
        """
        description_map = {
            cls.GENERAL_CHAT: "一般聊天对话",
            cls.CREATIVE_WRITING: "创意写作内容生成",
            cls.CODE_GENERATION: "代码生成和编程",
            cls.CODE_EXPLANATION: "代码解释和分析",
            cls.DATA_ANALYSIS: "数据分析和处理",
            cls.SUMMARIZATION: "内容摘要和要点提取",
            cls.TRANSLATION: "语言翻译",
            cls.MATH_PROBLEM: "数学问题求解",
            cls.KNOWLEDGE_QA: "知识问答",
            cls.REASONING: "逻辑推理和分析",
            cls.STRUCTURED_OUTPUT: "生成结构化输出"
        }
        
        return description_map.get(task_type, "未知任务类型")
    
    @classmethod
    def get_model_affinity(cls, task_type) -> Dict[str, float]:
        """
        获取任务类型与不同模型的亲和度
        
        Args:
            task_type: TaskType枚举值
            
        Returns:
            模型亲和度字典，键为模型名称，值为0-1的亲和度分数
        """
        # 这里的值是初始预设，将通过性能跟踪系统动态调整
        affinity_map = {
            cls.GENERAL_CHAT: {
                "claude": 0.8,
                "deepseek": 0.8,
                "mock": 0.5
            },
            cls.CREATIVE_WRITING: {
                "claude": 0.9,
                "deepseek": 0.7,
                "mock": 0.3
            },
            cls.CODE_GENERATION: {
                "claude": 0.7,
                "deepseek": 0.9,
                "mock": 0.3
            },
            cls.CODE_EXPLANATION: {
                "claude": 0.8,
                "deepseek": 0.8,
                "mock": 0.3
            },
            cls.DATA_ANALYSIS: {
                "claude": 0.8,
                "deepseek": 0.7,
                "mock": 0.2
            },
            cls.SUMMARIZATION: {
                "claude": 0.9,
                "deepseek": 0.8,
                "mock": 0.4
            },
            cls.TRANSLATION: {
                "claude": 0.8,
                "deepseek": 0.8,
                "mock": 0.2
            },
            cls.MATH_PROBLEM: {
                "claude": 0.7,
                "deepseek": 0.9,
                "mock": 0.2
            },
            cls.KNOWLEDGE_QA: {
                "claude": 0.9,
                "deepseek": 0.8,
                "mock": 0.4
            },
            cls.REASONING: {
                "claude": 0.9,
                "deepseek": 0.8,
                "mock": 0.3
            },
            cls.STRUCTURED_OUTPUT: {
                "claude": 0.8,
                "deepseek": 0.8,
                "mock": 0.5
            }
        }
        
        return affinity_map.get(task_type, {"claude": 0.5, "deepseek": 0.5, "mock": 0.5})
    
    @classmethod
    def get_parameter_adjustments(cls, task_type) -> Dict[str, Dict]:
        """
        获取任务类型对应的模型参数调整建议
        
        Args:
            task_type: TaskType枚举值
            
        Returns:
            参数调整字典，包含针对不同模型的参数调整建议
        """
        # 根据任务类型提供的模型参数调整建议
        adjustments_map = {
            cls.GENERAL_CHAT: {
                "temperature": 0.7,
                "top_p": 0.9,
                "system_prompt": "你是一个友好的智能助手，可以进行日常对话。"
            },
            cls.CREATIVE_WRITING: {
                "temperature": 0.8,
                "top_p": 0.95,
                "system_prompt": "你是一个创意写作助手，擅长创作各类文学内容。"
            },
            cls.CODE_GENERATION: {
                "temperature": 0.2,
                "top_p": 0.8,
                "system_prompt": "你是一个编程助手，专注于生成高质量、可运行的代码。请确保代码符合最佳实践，并提供必要的注释。"
            },
            cls.CODE_EXPLANATION: {
                "temperature": 0.3,
                "top_p": 0.8,
                "system_prompt": "你是一个代码解释专家，擅长分析代码结构和解释代码功能。请详细解释代码的工作原理，使用简明的语言。"
            },
            cls.DATA_ANALYSIS: {
                "temperature": 0.3,
                "top_p": 0.8,
                "system_prompt": "你是一个数据分析助手，专注于帮助用户理解和分析数据。"
            },
            cls.SUMMARIZATION: {
                "temperature": 0.3,
                "top_p": 0.8,
                "system_prompt": "你是一个摘要生成专家，擅长提取文本的关键信息并进行简明扼要的总结。"
            },
            cls.TRANSLATION: {
                "temperature": 0.3,
                "top_p": 0.8,
                "system_prompt": "你是一个翻译助手，专注于准确翻译不同语言之间的内容。请保持原文的意思和风格。"
            },
            cls.MATH_PROBLEM: {
                "temperature": 0.1,
                "top_p": 0.7,
                "system_prompt": "你是一个数学问题解答专家，擅长通过逐步分析来解决数学问题。请展示完整的解题过程。"
            },
            cls.KNOWLEDGE_QA: {
                "temperature": 0.4,
                "top_p": 0.8,
                "system_prompt": "你是一个知识问答助手，擅长回答各种知识性问题。请提供准确、全面的信息。"
            },
            cls.REASONING: {
                "temperature": 0.2,
                "top_p": 0.8,
                "system_prompt": "你是一个逻辑推理助手，擅长分析复杂问题和进行逻辑推导。请清晰地展示你的思考过程。"
            },
            cls.STRUCTURED_OUTPUT: {
                "temperature": 0.2,
                "top_p": 0.7,
                "system_prompt": "你是一个结构化数据生成助手，擅长创建格式化的内容。请严格按照要求的格式输出。"
            }
        }
        
        return adjustments_map.get(task_type, {"temperature": 0.7, "top_p": 0.9, "system_prompt": ""})
