"""
智能路由器测试脚本

本脚本测试智能路由器，整合任务分析和模型选择功能。
"""

import sys
import os
import json
from pprint import pprint

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcp.router.router import Router


def test_with_examples():
    """测试不同类型的输入示例"""
    
    # 创建路由器
    router = Router()
    
    # 测试用例
    test_cases = [
        # 一般聊天
        "你好，今天天气怎么样？",
        
        # 创意写作
        "帮我写一篇关于春天的散文",
        
        # 代码生成
        "用Python实现一个快速排序算法",
        
        # 数据分析
        "分析这组数据的趋势：1,5,3,8,12,9,6",
        
        # 翻译
        "把这段话翻译成英文：人工智能正在改变世界",
        
        # 数学问题
        "求解方程 x^2 + 5x + 6 = 0",
    ]
    
    # 测试每个用例
    print("====== 智能路由器测试 ======\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"测试用例 {i}: \"{text}\"")
        
        result = router.route(text)
        
        # 提取关键信息
        task_type = result["task_analysis"]["description"]
        task_confidence = result["task_analysis"]["confidence"]
        
        provider = result["model_selection"]["provider"]
        model = result["model_selection"]["model"]
        selection_confidence = result["model_selection"]["confidence"]
        
        params = result["model_selection"]["parameters"]
        temperature = params["temperature"]
        system_prompt = params.get("system_prompt", "")[:50] + "..." if len(params.get("system_prompt", "")) > 50 else params.get("system_prompt", "")
        
        # 显示结果
        print(f"任务类型: {task_type} (置信度: {task_confidence:.2f})")
        print(f"选择模型: {provider}/{model} (置信度: {selection_confidence:.2f})")
        print(f"参数调整: 温度={temperature}, 系统提示=\"{system_prompt}\"")
        
        # 模拟性能反馈
        router.update_performance(
            result["task_analysis"]["task_type"],
            provider,
            model,
            0.75 + 0.2 * (i % 2),  # 模拟不同的性能分数
            0.5 + 0.3 * i  # 模拟不同的响应时间
        )
        
        print("\n" + "-" * 50 + "\n")
    
    # 显示性能历史
    print("====== 性能历史 ======\n")
    context = router.get_context()
    
    if "performance_history" in context:
        for task_type, providers in context["performance_history"].items():
            print(f"任务类型: {task_type}")
            for provider, history in providers.items():
                samples = history["samples"]
                avg_score = sum(s["score"] for s in samples) / len(samples)
                avg_time = sum(s["response_time"] for s in samples) / len(samples)
                print(f"  • {provider}: 样本数={len(samples)}, 平均分数={avg_score:.2f}, 平均时间={avg_time:.2f}s")
            print()
    
    if "response_times" in context:
        print("响应时间:")
        for provider, time in context["response_times"].items():
            print(f"  • {provider}: {time:.2f}s")


def test_custom_input():
    """测试用户自定义输入"""
    
    router = Router()
    
    print("\n====== 自定义输入测试 ======\n")
    print("输入'exit'退出测试")
    
    while True:
        user_input = input("\n请输入测试文本: ")
        if user_input.lower() == 'exit':
            break
        
        result = router.route(user_input)
        
        # 显示结果
        print("\n路由结果:")
        print(f"任务类型: {result['task_analysis']['description']}")
        print(f"置信度: {result['task_analysis']['confidence']:.2f}")
        print(f"选择模型: {result['model_selection']['provider']}/{result['model_selection']['model']}")
        
        print("\n参数建议:")
        for param, value in result['model_selection']['parameters'].items():
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            print(f"  • {param}: {value}")
        
        print("\n备选模型:")
        for alt in result['model_selection']['alternative_models']:
            print(f"  • {alt['provider']}/{alt['model']} (分数: {alt['score']:.2f})")
        
        # 询问性能反馈
        feedback = input("\n请为这个选择评分(0.0-1.0): ")
        try:
            score = float(feedback)
            if 0.0 <= score <= 1.0:
                router.update_performance(
                    result["task_analysis"]["task_type"],
                    result["model_selection"]["provider"],
                    result["model_selection"]["model"],
                    score,
                    1.0  # 模拟响应时间
                )
                print(f"已更新性能记录，评分: {score:.2f}")
            else:
                print("评分无效，请输入0.0到1.0之间的数字")
        except ValueError:
            print("跳过性能更新")


if __name__ == "__main__":
    # 运行示例测试
    test_with_examples()
    
    # 运行自定义输入测试
    test_custom_input()
