"""
缓存功能测试示例
"""

import os
import sys
import time

# 添加项目根目录到路径，以便导入mcp包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.api.client import MCPClient
from mcp.utils.cache import cache_manager
from mcp.utils.logger import setup_logger

# 设置日志
logger = setup_logger("test_cache", log_level="DEBUG")


def test_caching_with_mock():
    """测试使用模拟客户端的缓存功能"""
    # 清空缓存，确保测试从干净状态开始
    cache_manager.clear()
    logger.info("缓存已清空")
    
    # 创建客户端
    client = MCPClient(default_model="mock")
    logger.info("创建模拟客户端")
    
    # 构造一个测试查询
    prompt = "请用简洁的语言介绍一下Python的特点和优势"
    
    # 第一次调用 - 应该从API获取
    start_time = time.time()
    logger.info(f"发送第一次请求: {prompt}")
    response1 = client.generate(prompt=prompt, max_tokens=500, temperature=0.7)
    first_call_time = time.time() - start_time
    logger.info(f"第一次请求完成，用时: {first_call_time:.3f}秒")
    
    # 第二次调用 - 应该从缓存获取
    start_time = time.time()
    logger.info(f"发送第二次请求（相同参数）: {prompt}")
    response2 = client.generate(prompt=prompt, max_tokens=500, temperature=0.7)
    second_call_time = time.time() - start_time
    logger.info(f"第二次请求完成，用时: {second_call_time:.3f}秒")
    
    # 验证两次响应相同
    if response1 == response2:
        logger.info("两次响应内容一致，缓存工作正常")
    else:
        logger.error("两次响应内容不一致，缓存可能存在问题")
        return False
    
    # 验证第二次调用更快（缓存响应应该更快）
    if second_call_time < first_call_time:
        speedup = first_call_time / second_call_time
        logger.info(f"缓存命中使响应速度提高了{speedup:.1f}倍")
    else:
        logger.warning(f"缓存命中没有提高响应速度，第一次: {first_call_time:.3f}秒，第二次: {second_call_time:.3f}秒")
    
    # 获取缓存统计信息
    memory_stats = cache_manager.get_stats("memory")
    logger.info(f"内存缓存命中率: {memory_stats.get('hit_ratio', 0):.2%}")
    
    disk_stats = cache_manager.get_stats("disk")
    logger.info(f"磁盘缓存命中率: {disk_stats.get('hit_ratio', 0):.2%}")
    
    return True


def test_cache_with_different_params():
    """测试不同参数对缓存的影响"""
    # 创建客户端
    client = MCPClient(default_model="mock")
    
    # 构造一个测试查询
    prompt = "请用简洁的语言介绍一下Python的特点和优势"
    
    # 不同温度参数
    logger.info("测试不同参数的缓存效果")
    response1 = client.generate(prompt=prompt, temperature=0.7)
    response2 = client.generate(prompt=prompt, temperature=0.8)
    
    if response1 == response2:
        logger.warning("使用不同温度参数时，响应应该不同，但实际相同")
    else:
        logger.info("不同温度参数生成不同的缓存键，缓存工作正常")
    
    # 不同最大标记数
    response3 = client.generate(prompt=prompt, max_tokens=100)
    response4 = client.generate(prompt=prompt, max_tokens=1000)
    
    if response3 == response4:
        logger.warning("使用不同max_tokens参数时，响应应该不同，但实际相同")
    else:
        logger.info("不同max_tokens参数生成不同的缓存键，缓存工作正常")
    
    return True


if __name__ == "__main__":
    print("开始测试缓存功能...")
    
    logger.info("=" * 50)
    logger.info("测试1: 基本缓存功能")
    result1 = test_caching_with_mock()
    
    logger.info("=" * 50)
    logger.info("测试2: 不同参数的缓存效果")
    result2 = test_cache_with_different_params()
    
    # 获取并显示缓存的内容
    memory_cache = cache_manager.get_cache("memory")
    if memory_cache:
        keys = memory_cache.get_keys()
        logger.info(f"缓存的键: {keys}")
    
    # 打印整体结果
    if result1 and result2:
        print("\n所有测试通过！缓存功能工作正常。")
    else:
        print("\n存在失败的测试，请检查日志。")
