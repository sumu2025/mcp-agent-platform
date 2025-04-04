const { ClaudeAdapter } = require("../mcp/adapters/claude-adapter-real");
const { DeepSeekAdapter } = require("../mcp/adapters/deepseek-adapter-real");
const { EnhancedModelManager } = require("../mcp/core/enhanced-model-manager");

/**
 * 使用真实API的降级演示应用
 */
class RealAPIFallbackDemo {
  /**
   * 创建演示应用实例
   * @param {string} claudeApiKey - Claude API密钥
   * @param {string} deepSeekApiKey - DeepSeek API密钥
   * @param {Object} options - 配置选项
   */
  constructor(claudeApiKey, deepSeekApiKey, options = {}) {
    // 创建适配器
    this.claudeAdapter = new ClaudeAdapter(claudeApiKey, {
      modelVersion: options.claudeModel || "claude-3-sonnet-20240229"
    });
    
    this.deepSeekAdapter = new DeepSeekAdapter(deepSeekApiKey, {
      model: options.deepSeekModel || "deepseek-chat"
    });
    
    // 创建模型管理器
    this.modelManager = new EnhancedModelManager(
      this.claudeAdapter, 
      this.deepSeekAdapter,
      {
        maxRetries: options.maxRetries || 2,
        autoFallback: options.autoFallback !== false,
        autoRecovery: options.autoRecovery !== false,
        recoveryInterval: options.recoveryInterval || 300000
      }
    );
  }
  
  /**
   * 处理提示词
   * @param {string} prompt - 输入提示词
   * @param {Object} options - 可选参数
   * @returns {Promise<string>} 生成的响应
   */
  async processPrompt(prompt, options = {}) {
    console.log(`当前使用模型: ${this.modelManager.getCurrentModelName()}`);
    
    const startTime = Date.now();
    const response = await this.modelManager.processPrompt(prompt, options);
    const endTime = Date.now();
    
    console.log(`处理耗时: ${(endTime - startTime) / 1000}秒`);
    
    return response;
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
   * 设置自动降级
   * @param {boolean} enabled - 是否启用
   */
  setAutoFallback(enabled) {
    this.modelManager.setAutoFallback(enabled);
    console.log(`自动降级已${enabled ? "启用" : "禁用"}`);
  }
  
  /**
   * 设置自动恢复
   * @param {boolean} enabled - 是否启用
   * @param {number} minutes - 恢复检查间隔(分钟)
   */
  setAutoRecovery(enabled, minutes = 5) {
    this.modelManager.setAutoRecovery(enabled, minutes * 60000);
    console.log(`自动恢复已${enabled ? "启用" : "禁用"}，间隔${minutes}分钟`);
  }
  
  /**
   * 运行完整演示
   * @returns {Promise<void>}
   */
  async runDemo() {
    console.log("===== MCP真实API模型降级演示 =====");
    
    // 步骤1：使用Claude处理请求
    console.log("\n步骤1: 使用Claude处理请求");
    try {
      const response1 = await this.processPrompt(
        "请用一段话介绍量子计算的基本原理",
        { maxTokens: 200 }
      );
      console.log(`结果: ${response1}`);
    } catch (error) {
      console.error("步骤1失败:", error.message);
    }
    
    // 步骤2：模拟Claude故障
    console.log("\n步骤2: 模拟Claude故障");
    this.simulateClaudeFailure();
    
    // 步骤3：使用DeepSeek处理请求
    console.log("\n步骤3: 测试自动降级");
    try {
      const response2 = await this.processPrompt(
        "什么是量子纠缠现象？为什么它很重要？",
        { maxTokens: 200 }
      );
      console.log(`结果: ${response2}`);
    } catch (error) {
      console.error("步骤3失败:", error.message);
    }
    
    // 步骤4：重置模拟
    console.log("\n步骤4: 重置模拟");
    this.resetSimulation();
    
    // 步骤5：验证恢复
    console.log("\n步骤5: 验证恢复正常");
    try {
      const response3 = await this.processPrompt(
        "量子计算能解决哪些传统计算机难以解决的问题？",
        { maxTokens: 200 }
      );
      console.log(`结果: ${response3}`);
    } catch (error) {
      console.error("步骤5失败:", error.message);
    }
    
    console.log("\n===== 演示完成 =====");
  }
}

module.exports = { RealAPIFallbackDemo };
