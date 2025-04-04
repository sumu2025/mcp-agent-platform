#!/usr/bin/env python
"""
DeepSeek API快速测试脚本 - 不依赖于包安装
"""

import os
import json
import time
import logging
import requests
from dotenv import load_dotenv

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='[%(asctime)s] [%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("quick_test")

# 加载环境变量
load_dotenv()

def quick_test_deepseek():
    """快速测试DeepSeek API连接"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
    
    if not api_key:
        logger.error("DeepSeek API密钥未设置，请在.env文件中设置DEEPSEEK_API_KEY")
        return False
    
    # 构建请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # 构建请求体
    request_data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的编程教师，善于用简洁清晰的语言解释技术概念。"},
            {"role": "user", "content": "请用简洁的语言介绍一下Python的特点和优势"}
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }
    
    try:
        logger.info("发送请求到DeepSeek API...")
        
        response = requests.post(
            api_url,
            headers=headers,
            json=request_data,
            timeout=60
        )
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            logger.info("请求成功!")
            print("\n" + "=" * 50 + "\n")
            print(content)
            print("\n" + "=" * 50)
            
            return True
        else:
            logger.error(f"请求失败，状态码: {response.status_code}")
            logger.error(f"响应内容: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始快速测试DeepSeek API...")
    
    try:
        success = quick_test_deepseek()
        if success:
            print("\n测试成功完成!")
        else:
            print("\n测试失败，请检查错误日志。")
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生未预期错误: {str(e)}")
