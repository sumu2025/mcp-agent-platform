/**
 * 模型管理器，处理多个模型和自动降级
 */
class ModelManager {
  /**
   * 创建模型管理器实例
   * @param {Object} primaryAdapter - 主模型适配器
   * @param {Object} fallbackAdapter - 备用模型适配器
   */
  constructor(primaryAdapter, fallbackAdapter) {
    this.primaryAdapter = primaryAdapter;
    this.fallbackAdapter = fallbackAdapter;
    this.currentAdapter = primaryAdapter;
  }
  
  /**
   * 处理提示词，带自动降级功能
   * @param {string} prompt - 输入提示词
   * @returns {Promise<string>} 生成的响应
   */
  async processPrompt(prompt) {
    try {
      // 检查主模型可用性
      if (this.currentAdapter === this.primaryAdapter) {
        const isAvailable = await this.primaryAdapter.checkAvailability();
        if (!isAvailable) {
          console.log("主要模型不可用，切换到备用模型");
          this.currentAdapter = this.fallbackAdapter;
        }
      }
      
      // 使用当前适配器
      return await this.currentAdapter.generateText(prompt);
    } catch (error) {
      // 主模型失败，尝试备用模型
      if (this.currentAdapter === this.primaryAdapter) {
        console.log("主要模型处理失败，切换到备用模型");
        this.currentAdapter = this.fallbackAdapter;
        return await this.currentAdapter.generateText(prompt);
      }
      
      // 备用模型也失败
      throw error;
    }
  }
  
  /**
   * 获取当前活跃模型名称
   * @returns {string} 模型名称
   */
  getCurrentModelName() {
    return this.currentAdapter === this.primaryAdapter ? "Claude" : "DeepSeek";
  }
  
  /**
   * 重置为主模型
   */
  resetToDefault() {
    this.currentAdapter = this.primaryAdapter;
  }
}

module.exports = { ModelManager };
