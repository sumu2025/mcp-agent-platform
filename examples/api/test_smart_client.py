"""
智能客户端测试脚本

本脚本测试智能API客户端，展示智能路由功能。
"""

import sys
import os
import time
from pprint import pprint

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcp.api.client.smart_client import SmartMCPClient
from mcp.api.base import GenerationParameters


def test_smart_client():
    """测试智能客户端的基本功能"""
    
    # 创建智能客户端
    client = SmartMCPClient()
    
    # 测试用例
    test_cases = [
        # 一般聊天
        "你好，今天天气怎么样？",
        
        # 代码生成
        "用Python实现一个快速排序算法",
        
        # 数据分析
        "分析这组数据的趋势：1,5,3,8,12,9,6",
    ]
    
    # 测试每个用例
    print("====== 智能客户端测试 ======\n")
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"测试用例 {i}: \"{prompt}\"")
        
        # 记录开始时间
        start_time = time.time()
        
        # 生成内容
        params = GenerationParameters(prompt=prompt)
        
        try:
            response = client.generate(params)
            
            # 记录结束时间
            elapsed_time = time.time() - start_time
            
            # 显示结果
            print(f"生成内容: {response.text[:100]}...")
            print(f"生成时间: {elapsed_time:.2f}秒")
            print(f"总tokens: {response.total_tokens}")
            
            # 显示路由信息（如果有）
            if hasattr(response, "routing_info"):
                routing_info = response.routing_info
                print("\n路由信息:")
                print(f"任务类型: {routing_info['task_analysis']['description']}")
                print(f"选择模型: {routing_info['model_selection']['provider']}/{routing_info['model_selection']['model']}")
            
        except Exception as e:
            print(f"生成失败: {str(e)}")
        
        print("\n" + "-" * 50 + "\n")


def test_with_routing_disabled():
    """测试禁用智能路由的情况"""
    
    # 创建智能客户端
    client = SmartMCPClient()
    
    # 禁用智能路由
    client.set_routing_enabled(False)
    
    print("====== 禁用智能路由测试 ======\n")
    
    # 生成内容
    prompt = "用Python实现一个二分查找算法"
    params = GenerationParameters(prompt=prompt)
    
    try:
        response = client.generate(params)
        
        # 显示结果
        print(f"生成内容: {response.text[:100]}...")
        print(f"模型: {response.model}")
        
        # 确认没有路由信息
        if hasattr(response, "routing_info"):
            print("警告：禁用路由但仍有路由信息")
        else:
            print("正确：禁用路由后没有路由信息")
        
    except Exception as e:
        print(f"生成失败: {str(e)}")


def test_explicit_provider():
    """测试明确指定提供商的情况"""
    
    # 创建智能客户端
    client = SmartMCPClient()
    
    print("====== 明确指定提供商测试 ======\n")
    
    # 生成内容
    prompt = "解释什么是机器学习"
    params = GenerationParameters(prompt=prompt, provider="mock")
    
    try:
        response = client.generate(params)
        
        # 显示结果
        print(f"生成内容: {response.text[:100]}...")
        print(f"指定提供商: mock, 实际使用: {response.model}")
        
    except Exception as e:
        print(f"生成失败: {str(e)}")


if __name__ == "__main__":
    # 运行测试
    test_smart_client()
    test_with_routing_disabled()
    test_explicit_provider()
