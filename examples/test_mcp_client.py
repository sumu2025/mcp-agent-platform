"""
MCP统一客户端测试示例
"""

import os
import sys

# 添加项目根目录到路径，以便导入mcp包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.api.client import MCPClient
from mcp.utils.logger import setup_logger

# 设置日志
logger = setup_logger("test_mcp_client", log_level="DEBUG")


def test_mcp_generate():
    """测试MCP统一客户端生成功能"""
    try:
        # 初始化客户端
        client = MCPClient(default_model="deepseek")
        
        # 测试生成文本
        prompt = "请简要解释人工智能和机器学习的区别"
        logger.info(f"发送提示词: {prompt}")
        
        # 调用API
        response = client.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            system_prompt="你是一个AI领域的专家，擅长用通俗易懂的语言解释复杂概念。"
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
    print("开始测试MCP统一客户端...")
    result = test_mcp_generate()
    
    if result:
        print("\n测试成功完成!")
    else:
        print("\n测试失败，请检查错误日志。")
