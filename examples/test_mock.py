"""
模拟API客户端测试示例
"""

import os
import sys

# 添加项目根目录到路径，以便导入mcp包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.api.client import MCPClient
from mcp.utils.logger import setup_logger

# 设置日志
logger = setup_logger("test_mock", log_level="INFO")


def test_mock_generate():
    """测试模拟API客户端生成功能"""
    try:
        # 初始化客户端
        client = MCPClient(default_model="mock")
        
        # 测试生成文本
        prompt = "请用简洁的语言介绍一下Python的特点和优势"
        logger.info(f"发送提示词: {prompt}")
        
        # 调用API
        response = client.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            system_prompt="你是一个专业的编程教师，善于用简洁清晰的语言解释技术概念。"
        )
        
        # 输出结果
        logger.info("生成成功!")
        print("\n" + "=" * 50 + "\n")
        print(response)
        print("\n" + "=" * 50)
        
        return True
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    print("开始测试模拟API客户端...")
    result = test_mock_generate()
    
    if result:
        print("\n测试成功完成!")
    else:
        print("\n测试失败，请检查错误日志。")
