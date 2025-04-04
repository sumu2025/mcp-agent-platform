"""
任务分析器测试脚本

本脚本测试任务分析器功能，包括关键词分析器、模式分析器和分析器管理器。
"""

import sys
import os
import json
from pprint import pprint

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcp.router.analyzers import TaskAnalyzerManager, KeywordTaskAnalyzer, PatternTaskAnalyzer, TaskType


def test_with_examples():
    """测试不同类型的输入示例"""
    
    # 创建分析器管理器
    analyzer_manager = TaskAnalyzerManager()
    
    # 测试用例
    test_cases = [
        # 一般聊天
        "你好，今天天气怎么样？",
        "你能告诉我你是谁吗？",
        
        # 创意写作
        "帮我写一篇关于春天的散文",
        "写一个科幻故事，主题是时间旅行",
        
        # 代码生成
        "用Python实现一个快速排序算法",
        "写一个计算斐波那契数列的JavaScript函数",
        
        # 代码解释
        "解释下面这段Python代码是做什么的",
        "这个React组件是如何工作的？",
        
        # 数据分析
        "分析这组数据的趋势",
        "帮我统计这些数字的平均值和标准差",
        
        # 摘要生成
        "总结这篇文章的要点",
        "把这个报告的主要内容提炼出来",
        
        # 翻译
        "把这段话翻译成英文",
        "将下面的英语句子翻译为中文",
        
        # 数学问题
        "求解方程 x^2 + 5x + 6 = 0",
        "计算积分 ∫x^2 dx",
        
        # 知识问答
        "什么是量子计算？",
        "解释一下相对论的基本原理",
        
        # 逻辑推理
        "如果所有的A都是B，所有的B都是C，那么所有的A都是C吗？",
        "分析这个问题的逻辑错误",
        
        # 结构化输出
        "生成一个表格，显示不同国家的GDP数据",
        "创建一个JSON格式的用户信息"
    ]
    
    # 测试每个用例
    print("====== 任务分析器测试 ======\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"测试用例 {i}: \"{text}\"")
        
        result = analyzer_manager.analyze(text)
        result_dict = result.to_dict()
        
        print(f"分析结果: {TaskType.get_description(result.task_type)}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"建议模型: {max(result_dict['model_affinity'].items(), key=lambda x: x[1])[0]}")
        print(f"参数调整: 温度={result_dict['parameter_adjustments']['temperature']}")
        print("详细信息:")
        if 'match_details' in result.analysis_details:
            for task_type, details in result.analysis_details['match_details'].items():
                print(f"  - {task_type}: 分数={details['score']:.2f}")
        print("\n" + "-" * 50 + "\n")


def test_individual_analyzers():
    """测试单个分析器的功能"""
    
    print("====== 单个分析器测试 ======\n")
    
    # 测试输入
    test_input = "写一个Python函数来计算斐波那契数列"
    
    # 测试关键词分析器
    keyword_analyzer = KeywordTaskAnalyzer()
    keyword_result = keyword_analyzer.analyze(test_input)
    
    print("关键词分析器结果:")
    print(f"任务类型: {TaskType.get_description(keyword_result.task_type)}")
    print(f"置信度: {keyword_result.confidence:.2f}")
    print("详细信息:")
    pprint(keyword_result.analysis_details)
    print()
    
    # 测试模式分析器
    pattern_analyzer = PatternTaskAnalyzer()
    pattern_result = pattern_analyzer.analyze(test_input)
    
    print("模式分析器结果:")
    print(f"任务类型: {TaskType.get_description(pattern_result.task_type)}")
    print(f"置信度: {pattern_result.confidence:.2f}")
    print("详细信息:")
    pprint(pattern_result.analysis_details)
    print()


def test_custom_input():
    """测试用户自定义输入"""
    
    analyzer_manager = TaskAnalyzerManager()
    
    print("====== 自定义输入测试 ======\n")
    print("输入'exit'退出测试")
    
    while True:
        user_input = input("\n请输入测试文本: ")
        if user_input.lower() == 'exit':
            break
        
        result = analyzer_manager.analyze(user_input)
        result_dict = result.to_dict()
        
        print(f"\n分析结果: {TaskType.get_description(result.task_type)}")
        print(f"置信度: {result.confidence:.2f}")
        print("模型亲和度:")
        for model, score in result_dict['model_affinity'].items():
            print(f"  - {model}: {score:.2f}")
        
        print("参数建议:")
        for param, value in result_dict['parameter_adjustments'].items():
            print(f"  - {param}: {value}")
        
        print("\n详细信息:")
        if 'match_details' in result.analysis_details:
            for task_type, details in result.analysis_details['match_details'].items():
                print(f"  - {task_type}:")
                if 'matched_keywords' in details:
                    print(f"    匹配关键词: {', '.join(details['matched_keywords'])}")
                if 'matched_patterns' in details:
                    print(f"    匹配模式数: {len(details['matched_patterns'])}")
                print(f"    分数: {details['score']:.2f}")


if __name__ == "__main__":
    # 运行所有测试
    test_with_examples()
    test_individual_analyzers()
    test_custom_input()
