const { ClaudeAdapter } = require("../mcp/adapters/claude-adapter-fixed");
const { DeepSeekAdapter } = require("../mcp/adapters/deepseek-adapter-real");
const { EnhancedModelManager } = require("../mcp/core/enhanced-model-manager");

/**
 * 使用修复后API的降级演示应用
 */
class FixedAPIFallbackDemo {
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
    
    // 日志
    this.verbose = options.verbose || false;
  }
  
  /**
   * 处理提示词
   * @param {string} prompt - 输入提示词
   * @param {Object} options - 可选参数
   * @returns {Promise<string>} 生成的响应
   */
  async processPrompt(prompt, options = {}) {
    const modelName = this.modelManager.getCurrentModelName();
    console.log(`当前使用模型: ${modelName}`);
    
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
   * 运行完整演示
   * @returns {Promise<void>}
   */
  async runDemo() {
    console.log("===== MCP修复版API模型降级演示 =====");
    
    // 首先检查API可用性
    console.log("检查API可用性...");
    try {
      const isClaudeAvailable = await this.claudeAdapter.checkAvailability();
      console.log(`Claude API可用性: ${isClaudeAvailable ? "可用" : "不可用"}`);
      
      const isDeepSeekAvailable = await this.deepSeekAdapter.checkAvailability();
      console.log(`DeepSeek API可用性: ${isDeepSeekAvailable ? "可用" : "不可用"}`);
    } catch (error) {
      console.error("API检查失败:", error.message);
    }
    
    // 步骤1：使用主模型
    console.log("步骤1: 使用主模型处理请求");
    try {
      const response1 = await this.processPrompt(
        "请简要描述人工智能的发展历程",
        { maxTokens: 200 }
      );
      console.log(`结果: ${response1}`);
    } catch (error) {
      console.error("步骤1失败:", error.message);
    }
    
    // 步骤2：模拟故障并自动降级
    console.log("步骤2: 模拟主模型故障");
    this.simulateClaudeFailure();
    
    console.log("步骤3: 测试自动降级");
    try {
      const response2 = await this.processPrompt(
        "人工智能技术面临哪些伦理挑战？",
        { maxTokens: 200 }
      );
      console.log(`结果: ${response2}`);
    } catch (error) {
      console.error("步骤3失败:", error.message);
    }
    
    // 步骤4：重置并测试恢复
    console.log("步骤4: 重置模拟");
    this.resetSimulation();
    
    console.log("步骤5: 验证恢复正常");
    try {
      const response3 = await this.processPrompt(
        "人工智能如何影响未来工作?",
        { maxTokens: 200 }
      );
      console.log(`结果: ${response3}`);
    } catch (error) {
      console.error("步骤5失败:", error.message);
    }
    
    console.log("===== 演示完成 =====");
  }
}

module.exports = { FixedAPIFallbackDemo };
