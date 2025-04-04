require("dotenv").config();

// 导入适配器和管理器
const { ClaudeAdapter } = require("../mcp/adapters/claude-adapter-fixed");
const { DeepSeekAdapter } = require("../mcp/adapters/deepseek-adapter-real");
const { EnhancedModelManager } = require("../mcp/core/enhanced-model-manager");

async function main() {
  console.log("===== 测试增强版模型管理器 =====");
  
  // 从环境变量获取API密钥
  const claudeApiKey = process.env.CLAUDE_API_KEY || "";
  const deepSeekApiKey = process.env.DEEPSEEK_API_KEY || "";
  
  // 验证API密钥
  if (!claudeApiKey || !deepSeekApiKey) {
    console.error("错误: 缺少API密钥，请在.env文件中设置CLAUDE_API_KEY和DEEPSEEK_API_KEY");
    process.exit(1);
  }
  
  // 创建适配器和管理器
  const claudeAdapter = new ClaudeAdapter(claudeApiKey);
  const deepSeekAdapter = new DeepSeekAdapter(deepSeekApiKey);
  const manager = new EnhancedModelManager(claudeAdapter, deepSeekAdapter, {
    autoRecovery: true,
    recoveryInterval: 60000  // 测试时使用1分钟间隔
  });
  
  // 步骤1：测试正常使用
  console.log("\n步骤1: 正常使用主模型");
  try {
    console.log(`当前使用模型: ${manager.getCurrentModelName()}`);
    const response1 = await manager.processPrompt(
      "请简要介绍量子计算的基本原理",
      { maxTokens: 200 }
    );
    console.log(`结果: ${response1.substring(0, 200)}...`);
  } catch (error) {
    console.error("步骤1失败:", error);
  }
  
  // 步骤2：测试模拟故障
  console.log("\n步骤2: 模拟主模型故障");
  // 如果有setMockFailure方法可用，则使用它
  if (claudeAdapter.setMockFailure) {
    claudeAdapter.setMockFailure(true);
    console.log("已设置Claude模拟故障");
  } else {
    console.log("警告: 适配器不支持模拟故障，请手动更改API密钥以测试");
  }
  
  // 步骤3：测试备用模型
  console.log("\n步骤3: 使用备用模型");
  try {
    const response2 = await manager.processPrompt(
      "请解释神经网络的工作原理",
      { maxTokens: 200 }
    );
    console.log(`当前使用模型: ${manager.getCurrentModelName()}`);
    console.log(`结果: ${response2.substring(0, 200)}...`);
  } catch (error) {
    console.error("步骤3失败:", error);
  }
  
  // 步骤4：重置模拟，恢复正常
  console.log("\n步骤4: 重置模拟状态");
  if (claudeAdapter.setMockFailure) {
    claudeAdapter.setMockFailure(false);
    console.log("已重置Claude模拟故障");
  }
  
  manager.resetToDefault();
  console.log(`当前使用模型: ${manager.getCurrentModelName()}`);
  
  console.log("\n===== 测试完成 =====");
}

main().catch(error => {
  console.error("测试失败:", error);
  process.exit(1);
});
