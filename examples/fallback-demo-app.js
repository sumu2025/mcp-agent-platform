const { ClaudeAdapter } = require("../mcp/adapters/claude-adapter-impl");
const { DeepSeekAdapter } = require("../mcp/adapters/deepseek-adapter-impl");
const { ModelManager } = require("../mcp/core/model-manager");

/**
 * 模型降级演示应用
 */
class FallbackDemoApp {
  /**
   * 创建演示应用实例
   * @param {string} claudeApiKey - Claude API密钥
   * @param {string} deepSeekApiKey - DeepSeek API密钥
   */
  constructor(claudeApiKey, deepSeekApiKey) {
    this.claudeAdapter = new ClaudeAdapter(claudeApiKey);
    this.deepSeekAdapter = new DeepSeekAdapter(deepSeekApiKey);
    this.modelManager = new ModelManager(this.claudeAdapter, this.deepSeekAdapter);
  }
  
  /**
   * 处理提示词
   * @param {string} prompt - 输入提示词
   * @returns {Promise<string>} 生成的响应
   */
  async processPrompt(prompt) {
    console.log(`当前使用模型: ${this.modelManager.getCurrentModelName()}`);
    return await this.modelManager.processPrompt(prompt);
  }
  
  /**
   * 模拟Claude服务不可用
   */
  simulateClaudeFailure() {
    this.claudeAdapter.setSimulateFailure(true);
    console.log("已设置Claude模拟故障");
  }
  
  /**
   * 重置模拟状态
   */
  resetSimulation() {
    this.claudeAdapter.setSimulateFailure(false);
    this.modelManager.resetToDefault();
    console.log("已重置模拟状态和默认模型");
  }
  
  /**
   * 运行完整演示
   * @returns {Promise<void>}
   */
  async runDemo() {
    console.log("===== MCP模型降级演示 =====");
    
    // 步骤1：正常使用Claude
    console.log("\n步骤1: 使用Claude处理请求");
    const response1 = await this.processPrompt("解释量子计算的基本原理");
    console.log(`结果: ${response1}`);
    
    // 步骤2：模拟Claude故障
    console.log("\n步骤2: 模拟Claude故障");
    this.simulateClaudeFailure();
    
    // 步骤3：自动降级到DeepSeek
    console.log("\n步骤3: a测试自动降级");
    const response2 = await this.processPrompt("什么是量子纠缠？");
    console.log(`结果: ${response2}`);
    
    // 步骤4：重置系统
    console.log("\n步骤4: 重置系统");
    this.resetSimulation();
    
    // 步骤5：验证恢复
    console.log("\n步骤5: 验证正常恢复");
    const response3 = await this.processPrompt("量子计算的应用领域");
    console.log(`结果: ${response3}`);
    
    console.log("\n===== 演示完成 =====");
  }
}

module.exports = { FallbackDemoApp };
