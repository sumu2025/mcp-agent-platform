"""
上下文增强系统测试
"""

import sys
import os
from pathlib import Path
import unittest
import tempfile
import shutil

# 设置路径以导入MCP模块
sys.path.append(str(Path(__file__).parent.parent))

# 导入所需组件
from mcp.knowledge.retrieval import RetrievalResult
from mcp.knowledge.augmentation import (
    AugmentationConfig,
    AugmentationMode,
    AugmentedContext,
    BasicAugmenter,
    StructuredAugmenter,
    TemplateAugmenter,
    get_augmenter
)
from mcp.knowledge.augmentation.prompt_builder import (
    PromptBuilder,
    BasicPromptBuilder,
    StructuredPromptBuilder,
    TemplatePromptBuilder
)
from mcp.knowledge.augmentation.context_formatter import (
    ContextFormatter,
    DefaultFormatter,
    MarkdownFormatter,
    SchemaFormatter
)
from mcp.knowledge.augmentation.token_management import TokenManager

class TestAugmentation(unittest.TestCase):
    """测试上下文增强功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试检索结果
        self.retrieval_results = [
            RetrievalResult(
                text="人工智能（AI）是计算机科学的一个分支，它关注创建能够执行通常需要人类智能的任务的系统。",
                score=0.95,
                id="doc1",
                metadata={"category": "AI", "level": "introductory"}
            ),
            RetrievalResult(
                text="机器学习是人工智能的一个子领域，它关注构建能够从数据中学习的系统，而无需被明确编程。",
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
        
        # 测试查询
        self.test_query = "什么是人工智能和机器学习？"
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_prompt_builder(self):
        """测试提示构建器"""
        # 创建基本提示构建器
        builder = BasicPromptBuilder()
        
        # 测试系统提示构建
        retrieval_text = "这是检索结果文本。"
        query = "这是查询文本。"
        
        system_prompt = builder.build_system_prompt(retrieval_text, query)
        self.assertIsNotNone(system_prompt)
        self.assertIn(retrieval_text, system_prompt)
        
        # 测试用户提示构建
        user_prompt = builder.build_user_prompt(query)
        self.assertEqual(user_prompt, query)
        
        # 测试模板提示构建器
        template_builder = TemplatePromptBuilder()
        
        # 注册自定义模板
        custom_template = "这是一个自定义模板，包含{retrieval_results}和{query}。"
        template_builder.register_template("custom", custom_template)
        
        # 测试自定义模板
        custom_prompt = template_builder.build_system_prompt(
            retrieval_text,
            query,
            template="custom"
        )
        self.assertIn(retrieval_text, custom_prompt)
        self.assertIn(query, custom_prompt)
        
        # 测试检索结果格式化
        formatted_result = builder.format_retrieval_result(
            self.retrieval_results[0],
            template="[{index}] {content}\n来源: {metadata}",
            include_metadata=True
        )
        self.assertIn(self.retrieval_results[0].text, formatted_result)
        self.assertIn("AI", formatted_result)
    
    def test_context_formatter(self):
        """测试上下文格式化器"""
        # 创建默认格式化器
        formatter = DefaultFormatter()
        
        # 测试单个结果格式化
        formatted = formatter.format_retrieval_result(self.retrieval_results[0], 0)
        self.assertIn("[1]", formatted)
        self.assertIn(self.retrieval_results[0].text, formatted)
        
        # 测试结果列表格式化
        formatted_list = formatter.format_retrieval_results(self.retrieval_results)
        self.assertIn("[1]", formatted_list)
        self.assertIn("[2]", formatted_list)
        self.assertIn("[3]", formatted_list)
        
        # 测试Markdown格式化器
        md_formatter = MarkdownFormatter()
        md_formatted = md_formatter.format_retrieval_results(self.retrieval_results)
        self.assertIn("###", md_formatted)
        self.assertIn(">", md_formatted)
        
        # 测试模式格式化器
        schema_formatter = SchemaFormatter()
        schema_formatted = schema_formatter.format_retrieval_results(self.retrieval_results)
        self.assertIn("<reference", schema_formatted)
        self.assertIn("<content>", schema_formatted)
        self.assertIn("</reference>", schema_formatted)
    
    def test_token_manager(self):
        """测试Token管理器"""
        # 创建Token管理器
        token_manager = TokenManager()
        
        # 测试Token计数
        text = "这是一个测试文本，用于测试Token管理器的计数功能。"
        count = token_manager.count_tokens(text)
        self.assertGreater(count, 0)
        
        # 测试文本截断
        long_text = "这是一个很长的文本。" * 50
        truncated = token_manager.truncate_text(long_text, 20)
        truncated_count = token_manager.count_tokens(truncated)
        self.assertLessEqual(truncated_count, 20)
        
        # 测试Token分配
        sections = {
            "system": "这是系统提示部分。" * 10,
            "retrieval": "这是检索结果部分。" * 20,
            "query": "这是查询部分。" * 5
        }
        
        priorities = ["system", "query", "retrieval"]
        
        # 计算原始Token
        original_tokens = sum(token_manager.count_tokens(text) for text in sections.values())
        
        # 分配Token（限制为原来的一半）
        distributed = token_manager.distribute_tokens(
            original_tokens // 2,
            sections,
            priorities
        )
        
        # 验证结果
        distributed_tokens = sum(token_manager.count_tokens(text) for text in distributed.values())
        self.assertLessEqual(distributed_tokens, original_tokens // 2 + 5)  # 允许有少量误差
        
        # 验证优先级（高优先级的部分应该保留完整）
        self.assertEqual(distributed["system"], sections["system"])
        self.assertEqual(distributed["query"], sections["query"])
        self.assertNotEqual(distributed["retrieval"], sections["retrieval"])
    
    def test_basic_augmenter(self):
        """测试基本增强器"""
        # 创建配置
        config = AugmentationConfig(
            mode=AugmentationMode.BASIC,
            max_context_length=2000,
            include_metadata=True
        )
        
        # 创建增强器
        augmenter = BasicAugmenter(config=config)
        augmenter.initialize()
        
        # 测试增强
        context = augmenter.augment(self.test_query, self.retrieval_results)
        
        # 验证结果
        self.assertIsInstance(context, AugmentedContext)
        self.assertIsNotNone(context.prompt)
        self.assertIn(self.test_query, context.query)
        
        # 验证检索结果包含
        for result in self.retrieval_results:
            self.assertIn(result.text, context.retrieval)
    
    def test_structured_augmenter(self):
        """测试结构化增强器"""
        # 创建配置
        config = AugmentationConfig(
            mode=AugmentationMode.STRUCTURED,
            max_context_length=2000,
            include_metadata=True
        )
        
        # 创建增强器
        augmenter = StructuredAugmenter(config=config)
        augmenter.initialize()
        
        # 测试增强
        context = augmenter.augment(self.test_query, self.retrieval_results)
        
        # 验证结果
        self.assertIsInstance(context, AugmentedContext)
        self.assertIn("参考资料", context.system)
        self.assertIn(self.test_query, context.query)
        
        # 验证Markdown格式
        self.assertIn("#", context.prompt)
        self.assertIn(">", context.retrieval)
    
    def test_template_augmenter(self):
        """测试模板增强器"""
        # 创建配置
        config = AugmentationConfig(
            mode=AugmentationMode.TEMPLATE,
            max_context_length=2000,
            include_metadata=True,
            custom_params={"template_name": "educational"}
        )
        
        # 创建增强器
        augmenter = TemplateAugmenter(
            config=config,
            template_name="educational"
        )
        augmenter.initialize()
        
        # 测试增强
        context = augmenter.augment(self.test_query, self.retrieval_results)
        
        # 验证结果
        self.assertIsInstance(context, AugmentedContext)
        self.assertIn("教学", context.system)
        self.assertIn(self.test_query, context.query)
    
    def test_factory_function(self):
        """测试工厂函数"""
        # 测试基本模式
        basic_config = AugmentationConfig(mode=AugmentationMode.BASIC)
        basic_augmenter = get_augmenter(basic_config)
        self.assertIsInstance(basic_augmenter, BasicAugmenter)
        
        # 测试结构化模式
        structured_config = AugmentationConfig(mode=AugmentationMode.STRUCTURED)
        structured_augmenter = get_augmenter(structured_config)
        self.assertIsInstance(structured_augmenter, StructuredAugmenter)
        
        # 测试模板模式
        template_config = AugmentationConfig(
            mode=AugmentationMode.TEMPLATE,
            custom_params={"template_name": "business"}
        )
        template_augmenter = get_augmenter(template_config)
        self.assertIsInstance(template_augmenter, TemplateAugmenter)
    
    def test_context_length_management(self):
        """测试上下文长度管理"""
        # 创建非常短的最大长度配置
        config = AugmentationConfig(
            mode=AugmentationMode.BASIC,
            max_context_length=100,  # 非常小的上下文长度
            use_token_management=True
        )
        
        # 创建增强器
        augmenter = BasicAugmenter(config=config)
        augmenter.initialize()
        
        # 创建长检索结果
        long_results = [
            RetrievalResult(
                text="这是一个非常长的检索结果，" * 20,
                score=0.9,
                id="long1",
                metadata={}
            ),
            RetrievalResult(
                text="这是另一个非常长的检索结果，" * 20,
                score=0.8,
                id="long2",
                metadata={}
            )
        ]
        
        # 测试增强
        context = augmenter.augment(self.test_query, long_results)
        
        # 验证结果长度被管理
        token_count = augmenter.token_manager.count_tokens(context.prompt)
        self.assertLessEqual(token_count, config.max_context_length + 50)  # 允许有少量误差
    
    def test_custom_template(self):
        """测试自定义模板"""
        # 创建自定义模板目录
        template_dir = os.path.join(self.temp_dir, "templates")
        os.makedirs(template_dir, exist_ok=True)
        
        # 写入自定义模板
        template_content = """
# 自定义模板

这是一个测试模板，它包含以下参考资料：

{retrieval_results}

请回答以下问题：{query}
"""
        
        template_path = os.path.join(template_dir, "custom.j2")
        with open(template_path, "w") as f:
            f.write(template_content)
        
        # 创建使用该模板的提示构建器
        builder = PromptBuilder(templates_dir=template_dir)
        
        # 测试加载模板
        template_func = builder.load_template("custom")
        self.assertIsNotNone(template_func)
        
        # 测试使用模板
        prompt = builder.build_system_prompt(
            "这是检索结果。",
            "这是查询。",
            template="custom"
        )
        
        self.assertIn("自定义模板", prompt)
        self.assertIn("这是检索结果。", prompt)
        self.assertIn("这是查询。", prompt)


if __name__ == "__main__":
    unittest.main()
