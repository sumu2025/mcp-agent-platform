"""
上下文增强系统测试
"""

import sys
import os
from pathlib import Path
import unittest

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入所需组件
from mcp.knowledge.retrieval import RetrievalResult
from mcp.knowledge.augmentation import (
    AugmentationConfig,
    AugmentationMode,
    BasicAugmenter,
    StructuredAugmenter,
    TemplateAugmenter,
    AugmentedContext
)
from mcp.knowledge.augmentation.token_management import TokenManager
from mcp.knowledge.augmentation.prompt_builder import (
    PromptBuilder,
    TemplatePromptBuilder
)
from mcp.knowledge.augmentation.context_formatter import (
    DefaultFormatter,
    MarkdownFormatter,
    SchemaFormatter
)


class TestContextAugmenter(unittest.TestCase):
    """测试上下文增强器功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试数据
        self.test_query = "什么是深度学习？它与机器学习有什么区别？"
        
        self.test_results = [
            RetrievalResult(
                text="深度学习是机器学习的一种方法，它使用多层神经网络来学习复杂模式。神经网络的灵感来自人类大脑的结构，由相互连接的节点（神经元）组成。深度学习在图像识别、自然语言处理和游戏中取得了显著成功。",
                score=0.92,
                id="result_1",
                metadata={"category": "AI", "subcategory": "deep_learning", "level": "intermediate"}
            ),
            RetrievalResult(
                text="机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统，而无需被明确编程。常见的机器学习方法包括监督学习、无监督学习和强化学习。",
                score=0.85,
                id="result_2",
                metadata={"category": "AI", "subcategory": "machine_learning", "level": "introductory"}
            ),
            RetrievalResult(
                text="与传统机器学习相比，深度学习的主要区别在于它能够自动发现数据中的特征，而无需人工特征工程。这使其特别适合处理大规模、高维度和非结构化数据，如图像、语音和文本。",
                score=0.78,
                id="result_3",
                metadata={"category": "AI", "subcategory": "deep_learning", "level": "intermediate"}
            )
        ]
    
    def test_basic_augmenter(self):
        """测试基本增强器"""
        # 创建基本增强器
        augmenter = BasicAugmenter(config=AugmentationConfig(
            mode=AugmentationMode.BASIC,
            max_context_length=2000,
            include_metadata=True
        ))
        augmenter.initialize()
        
        # 增强上下文
        context = augmenter.augment(self.test_query, self.test_results)
        
        # 验证结果
        self.assertIsInstance(context, AugmentedContext)
        self.assertIn(self.test_query, context.query)
        
        # 检查检索结果是否包含在提示中
        for result in self.test_results:
            self.assertIn(result.text[:50], context.retrieval)
            
        # 检查元数据是否包含
        self.assertIn("category", context.retrieval)
        
        # 验证提示长度
        if augmenter.token_manager:
            token_count = augmenter.token_manager.count_tokens(context.prompt)
            self.assertLessEqual(token_count, augmenter.config.max_context_length)
    
    def test_structured_augmenter(self):
        """测试结构化增强器"""
        # 创建结构化增强器
        augmenter = StructuredAugmenter(config=AugmentationConfig(
            mode=AugmentationMode.STRUCTURED,
            max_context_length=2000
        ))
        augmenter.initialize()
        
        # 增强上下文
        context = augmenter.augment(self.test_query, self.test_results)
        
        # 验证结果
        self.assertIsInstance(context, AugmentedContext)
        
        # 检查是否使用了Markdown格式
        self.assertIn("#", context.system)
        self.assertIn("---", context.system)
        
        # 检查检索结果是否包含在提示中
        for result in self.test_results:
            self.assertIn(result.text[:50], context.retrieval)
    
    def test_template_augmenter(self):
        """测试模板增强器"""
        # 创建模板增强器
        augmenter = TemplateAugmenter(
            config=AugmentationConfig(
                mode=AugmentationMode.TEMPLATE,
                max_context_length=2000
            ),
            template_name="educational"
        )
        augmenter.initialize()
        
        # 增强上下文
        context = augmenter.augment(self.test_query, self.test_results)
        
        # 验证结果
        self.assertIsInstance(context, AugmentedContext)
        
        # 检查是否使用了教育模板
        self.assertIn("教学", context.system)
    
    def test_token_manager(self):
        """测试Token管理器"""
        # 创建Token管理器
        token_manager = TokenManager()
        
        # 测试Token计数
        text = "这是一个测试文本，用于测试Token管理器的计数功能。" * 10
        token_count = token_manager.count_tokens(text)
        self.assertGreater(token_count, 0)
        
        # 测试截断
        max_tokens = 20
        truncated = token_manager.truncate_text(text, max_tokens)
        truncated_count = token_manager.count_tokens(truncated)
        self.assertLessEqual(truncated_count, max_tokens)
        
        # 测试Token分配
        sections = {
            "system": "这是系统指令部分。" * 10,
            "retrieval": "这是检索结果部分。" * 20,
            "query": "这是查询部分。" * 5
        }
        
        priorities = ["system", "query", "retrieval"]
        
        total_tokens = 50
        adjusted = token_manager.distribute_tokens(total_tokens, sections, priorities)
        
        # 验证总Token数不超过限制
        total_adjusted_tokens = 0
        for section_name, section_text in adjusted.items():
            section_tokens = token_manager.count_tokens(section_text)
            total_adjusted_tokens += section_tokens
            
        self.assertLessEqual(total_adjusted_tokens, total_tokens)
        
        # 验证高优先级部分被保留
        self.assertEqual(adjusted["system"], sections["system"])
    
    def test_prompt_builder(self):
        """测试提示构建器"""
        # 创建提示构建器
        builder = PromptBuilder()
        
        # 测试系统提示构建
        retrieval_text = "检索结果1\n\n检索结果2"
        query = "测试查询"
        
        system_prompt = builder.build_system_prompt(retrieval_text, query)
        
        # 验证系统提示
        self.assertIn(retrieval_text, system_prompt)
        
        # 注册自定义模板
        template_content = "使用以下资料回答问题：\n\n{retrieval_results}\n\n问题是：{query}"
        builder.register_template("custom", template_content)
        
        # 使用自定义模板
        custom_prompt = builder.build_system_prompt(
            retrieval_text, 
            query,
            template="custom"
        )
        
        # 验证自定义提示
        self.assertIn(retrieval_text, custom_prompt)
        self.assertIn(query, custom_prompt)
    
    def test_formatters(self):
        """测试格式化器"""
        # 创建格式化器
        default_formatter = DefaultFormatter()
        markdown_formatter = MarkdownFormatter()
        schema_formatter = SchemaFormatter()
        
        # 创建测试数据
        result = self.test_results[0]
        
        # 测试默认格式化
        default_text = default_formatter.format_retrieval_result(result, 0)
        self.assertIn("[1]", default_text)
        self.assertIn(result.text[:50], default_text)
        
        # 测试Markdown格式化
        markdown_text = markdown_formatter.format_retrieval_result(result, 0)
        self.assertIn("###", markdown_text)
        self.assertIn(">", markdown_text)
        self.assertIn(result.text[:50], markdown_text)
        
        # 测试Schema格式化
        schema_text = schema_formatter.format_retrieval_result(result, 0)
        self.assertIn("<reference", schema_text)
        self.assertIn("<content>", schema_text)
        self.assertIn(result.text[:50], schema_text)


if __name__ == "__main__":
    unittest.main()
