"""
上下文增强系统测试
"""

import sys
import os
from pathlib import Path
import unittest
import tempfile
import shutil
import logging

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入上下文增强组件
from mcp.knowledge.retrieval import RetrievalResult
from mcp.knowledge.augmentation import (
    AugmentationConfig,
    AugmentationMode,
    AugmentedContext,
    BasicAugmenter,
    StructuredAugmenter,
    TemplateAugmenter
)
from mcp.knowledge.augmentation.token_management import TokenManager


class TestAugmenter(unittest.TestCase):
    """测试上下文增强系统"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试检索结果
        self.test_results = [
            RetrievalResult(
                text="人工智能是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。",
                score=0.95,
                id="doc1",
                metadata={"category": "AI", "level": "introductory"}
            ),
            RetrievalResult(
                text="机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统。",
                score=0.85,
                id="doc2",
                metadata={"category": "AI", "subcategory": "machine_learning"}
            ),
            RetrievalResult(
                text="深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。",
                score=0.75,
                id="doc3",
                metadata={"category": "AI", "subcategory": "deep_learning"}
            )
        ]
        
        # 创建测试查询
        self.test_query = "什么是深度学习？它与机器学习有什么区别？"
        
        # 创建增强配置
        self.basic_config = AugmentationConfig(
            mode=AugmentationMode.BASIC,
            max_context_length=2000,
            include_metadata=True
        )
        
        self.structured_config = AugmentationConfig(
            mode=AugmentationMode.STRUCTURED,
            max_context_length=2000,
            include_metadata=True
        )
        
        self.template_config = AugmentationConfig(
            mode=AugmentationMode.TEMPLATE,
            max_context_length=2000,
            include_metadata=True,
            custom_params={"template_name": "educational"}
        )
        
        # 创建增强器
        self.basic_augmenter = BasicAugmenter(config=self.basic_config)
        self.structured_augmenter = StructuredAugmenter(config=self.structured_config)
        self.template_augmenter = TemplateAugmenter(
            config=self.template_config,
            template_name="educational"
        )
        
        # 初始化增强器
        self.basic_augmenter.initialize()
        self.structured_augmenter.initialize()
        self.template_augmenter.initialize()
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_basic_augmenter(self):
        """测试基本增强器"""
        # 增强上下文
        context = self.basic_augmenter.augment(self.test_query, self.test_results)
        
        # 验证结果
        self.assertIsInstance(context, AugmentedContext, "结果应该是AugmentedContext对象")
        self.assertIn(self.test_query, context.query, "上下文应该包含查询")
        
        # 验证检索结果是否包含在上下文中
        for result in self.test_results:
            self.assertIn(result.text, context.retrieval, 
                        f"上下文应该包含检索结果: {result.text[:20]}...")
        
        # 验证提示是否非空
        self.assertIsNotNone(context.prompt, "提示不应该为空")
        self.assertGreater(len(context.prompt), 0, "提示长度应该大于0")
    
    def test_structured_augmenter(self):
        """测试结构化增强器"""
        # 增强上下文
        context = self.structured_augmenter.augment(self.test_query, self.test_results)
        
        # 验证结果
        self.assertIsInstance(context, AugmentedContext, "结果应该是AugmentedContext对象")
        
        # 验证系统提示包含指示
        self.assertIn("参考资料", context.system, "系统提示应该包含参考资料指示")
        
        # 验证元数据是否包含在上下文中
        metadata_included = False
        for result in self.test_results:
            if "category" in result.metadata:
                category = result.metadata["category"]
                if category in context.retrieval:
                    metadata_included = True
                    break
        
        self.assertTrue(metadata_included, "上下文应该包含元数据")
    
    def test_template_augmenter(self):
        """测试模板增强器"""
        # 增强上下文
        context = self.template_augmenter.augment(self.test_query, self.test_results)
        
        # 验证结果
        self.assertIsInstance(context, AugmentedContext, "结果应该是AugmentedContext对象")
        
        # 验证系统提示使用了教育模板
        self.assertIn("教学", context.system, "系统提示应该使用教育模板")
    
    def test_token_count(self):
        """测试Token计数"""
        # 增强上下文
        context = self.basic_augmenter.augment(self.test_query, self.test_results)
        
        # 验证Token计数
        self.assertIn("system", context.token_count, "应该有系统提示的Token计数")
        self.assertIn("retrieval", context.token_count, "应该有检索结果的Token计数")
        self.assertIn("query", context.token_count, "应该有查询的Token计数")
        self.assertIn("total", context.token_count, "应该有总计的Token计数")
        
        # 验证总计
        total = context.token_count["total"]
        parts_sum = (
            context.token_count["system"] + 
            context.token_count["retrieval"] + 
            context.token_count["query"]
        )
        
        # 总计可能包含额外的格式化Token，不一定严格等于各部分之和
        self.assertGreaterEqual(total, parts_sum * 0.9, "总计应该与各部分之和接近")
    
    def test_context_length_management(self):
        """测试上下文长度管理"""
        # 创建很长的检索结果
        long_text = "这是一个非常长的文本，" * 100
        long_results = [
            RetrievalResult(
                text=long_text,
                score=0.95,
                id="long_doc",
                metadata={"category": "test"}
            )
        ]
        
        # 设置非常小的上下文长度限制
        self.basic_augmenter.config.max_context_length = 200
        
        # 增强上下文
        context = self.basic_augmenter.augment(self.test_query, long_results)
        
        # 验证结果
        total_tokens = context.token_count["total"]
        self.assertLessEqual(total_tokens, 200 * 1.1,  # 允许一点误差
                          f"总Token数 ({total_tokens}) 应该不超过限制 (200)")
    
    def test_token_manager(self):
        """测试Token管理器"""
        # 创建Token管理器
        token_manager = TokenManager()
        
        # 测试文本
        test_text = "这是一个测试文本" * 10
        
        # 计算Token数量
        token_count = token_manager.count_tokens(test_text)
        self.assertGreater(token_count, 0, "Token数量应该大于0")
        
        # 测试截断
        truncated = token_manager.truncate_text(test_text, 10)
        truncated_count = token_manager.count_tokens(truncated)
        self.assertLessEqual(truncated_count, 10 * 1.1,  # 允许一点误差
                          f"截断后的Token数 ({truncated_count}) 应该不超过限制 (10)")
        
        # 测试Token分配
        sections = {
            "section1": "第一部分" * 20,
            "section2": "第二部分" * 10,
            "section3": "第三部分" * 5
        }
        
        priorities = ["section3", "section1", "section2"]
        
        # 计算原始Token数
        original_tokens = sum(token_manager.count_tokens(text) for text in sections.values())
        
        # 分配Token
        max_tokens = original_tokens // 2  # 限制为原始总数的一半
        distributed = token_manager.distribute_tokens(max_tokens, sections, priorities)
        
        # 计算分配后的Token数
        distributed_tokens = sum(token_manager.count_tokens(text) for text in distributed.values())
        
        # 验证结果
        self.assertLessEqual(distributed_tokens, max_tokens * 1.1,  # 允许一点误差
                          f"分配后的Token总数 ({distributed_tokens}) 应该不超过限制 ({max_tokens})")
        
        # 验证优先级
        # 由于section3优先级最高，它应该被完整保留
        self.assertEqual(distributed["section3"], sections["section3"],
                       "优先级最高的部分应该被完整保留")


if __name__ == "__main__":
    unittest.main()
