/**
 * Claude API适配器，实现模型接口
 */
class ClaudeAdapter {
  /**
   * 创建Claude适配器实例
   * @param {string} apiKey - Claude API密钥
   */
  constructor(apiKey) {
    if (!apiKey) throw new Error("API key is required");
    this.apiKey = apiKey;
    this.isAvailable = true;
    this.simulateFailure = false;
  }

  /**
   * 生成文本响应
   * @param {string} prompt - 输入提示词
   * @param {Object} options - 可选参数
   * @returns {Promise<string>} 生成的文本
   */
  async generateText(prompt, options = {}) {
    if (this.simulateFailure) {
      throw new Error("Claude service unavailable (simulated)");
    }

    // 模拟API调用
    console.log("🧠 Claude正在处理...");
    await new Promise(resolve => setTimeout(resolve, 1000));
    return `[Claude] 响应: ${prompt.substring(0, 20)}...`;
  }

  /**
   * 检查模型是否可用
   * @returns {Promise<boolean>} 可用状态
   */
  async checkAvailability() {
    if (this.simulateFailure) return false;
    return this.isAvailable;
  }

  /**
   * 设置故障模拟
   * @param {boolean} simulate - 是否模拟故障
   */
  setSimulateFailure(simulate) {
    this.simulateFailure = simulate;
  }
}

module.exports = { ClaudeAdapter };
