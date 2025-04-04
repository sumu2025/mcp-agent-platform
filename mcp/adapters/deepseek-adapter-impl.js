/**
 * DeepSeek API适配器，实现模型接口
 */
class DeepSeekAdapter {
  /**
   * 创建DeepSeek适配器实例
   * @param {string} apiKey - DeepSeek API密钥
   */
  constructor(apiKey) {
    if (!apiKey) throw new Error("API key is required");
    this.apiKey = apiKey;
    this.isAvailable = true;
  }

  /**
   * 生成文本响应
   * @param {string} prompt - 输入提示词
   * @param {Object} options - 可选参数
   * @returns {Promise<string>} 生成的文本
   */
  async generateText(prompt, options = {}) {
    // 模拟API调用
    console.log("🔍 DeepSeek正在处理...");
    await new Promise(resolve => setTimeout(resolve, 800));
    return `[DeepSeek] 响应: ${prompt.substring(0, 20)}...`;
  }

  /**
   * 检查模型是否可用
   * @returns {Promise<boolean>} 可用状态
   */
  async checkAvailability() {
    return this.isAvailable;
  }
}

module.exports = { DeepSeekAdapter };
