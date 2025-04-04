/**
 * 增强版模型管理器，添加自动恢复功能
 */
class EnhancedModelManager {
  constructor(primaryAdapter, fallbackAdapter, options = {}) {
    this.primaryAdapter = primaryAdapter;
    this.fallbackAdapter = fallbackAdapter;
    this.currentAdapter = primaryAdapter;
    
    // 配置选项
    this.maxRetries = options.maxRetries || 2;
    this.autoFallback = options.autoFallback !== false; // 默认启用自动降级
    this.autoRecovery = options.autoRecovery !== false; // 默认启用自动恢复
    this.recoveryInterval = options.recoveryInterval || 300000; // 5分钟
    
    // 内部状态
    this.lastFailureTime = null;
    this.recoveryTimer = null;
    
    // 初始化恢复计时器
    if (this.autoRecovery) {
      this._setupRecoveryTimer();
    }
  }
  
  /**
   * 处理提示词，带自动降级功能
   * @param {string} prompt - 输入提示词
   * @param {Object} options - 请求选项
   * @returns {Promise<string>} 生成的响应
   */
  async processPrompt(prompt, options = {}) {
    // 检查主模型是否需要恢复
    await this._checkRecovery();
    
    // 尝试使用当前适配器
    try {
      // 检查当前适配器可用性
      if (this.autoFallback && this.currentAdapter === this.primaryAdapter) {
        const isAvailable = await this.primaryAdapter.checkAvailability();
        if (!isAvailable) {
          console.log("主模型不可用，切换到备用模型");
          this._switchToFallback();
        }
      }
      
      // 尝试生成文本
      return await this._generateWithRetry(prompt, options);
    } catch (error) {
      // 如果主模型失败，尝试降级
      if (this.autoFallback && this.currentAdapter === this.primaryAdapter) {
        console.log("主模型处理失败，切换到备用模型");
        this._switchToFallback();
        return await this._generateWithRetry(prompt, options);
      }
      
      // 如果是备用模型失败，则抛出错误
      throw error;
    }
  }
  
  /**
   * 带重试的文本生成
   * @private
   */
  async _generateWithRetry(prompt, options) {
    let lastError;
    
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        return await this.currentAdapter.generateText(prompt, options);
      } catch (error) {
        lastError = error;
        console.warn(`尝试 ${attempt + 1}/${this.maxRetries + 1} 失败:`, error.message);
        
        if (attempt < this.maxRetries) {
          // 等待一段时间再重试
          await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
        }
      }
    }
    
    throw lastError;
  }
  
  /**
   * 切换到备用模型
   * @private
   */
  _switchToFallback() {
    this.currentAdapter = this.fallbackAdapter;
    this.lastFailureTime = Date.now();
    
    // 重新设置恢复计时器
    if (this.autoRecovery) {
      this._setupRecoveryTimer();
    }
  }
  
  /**
   * 设置恢复计时器
   * @private
   */
  _setupRecoveryTimer() {
    // 清除现有计时器
    if (this.recoveryTimer) {
      clearTimeout(this.recoveryTimer);
    }
    
    // 创建新计时器
    this.recoveryTimer = setTimeout(async () => {
      // 如果当前使用的是备用模型，尝试恢复到主模型
      if (this.currentAdapter === this.fallbackAdapter) {
        const isAvailable = await this.primaryAdapter.checkAvailability();
        if (isAvailable) {
          console.log("主模型已恢复可用");
          this.currentAdapter = this.primaryAdapter;
        } else {
          // 如果主模型仍不可用，重新设置计时器
          this._setupRecoveryTimer();
        }
      }
    }, this.recoveryInterval);
  }
  
  /**
   * 检查是否需要恢复到主模型
   * @private
   */
  async _checkRecovery() {
    // 如果正在使用备用模型，且时间已经过去足够久，检查主模型
    if (
      this.autoRecovery &&
      this.currentAdapter === this.fallbackAdapter &&
      this.lastFailureTime &&
      Date.now() - this.lastFailureTime >= this.recoveryInterval
    ) {
      const isAvailable = await this.primaryAdapter.checkAvailability();
      if (isAvailable) {
        console.log("主模型已恢复可用");
        this.currentAdapter = this.primaryAdapter;
      }
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
    this.lastFailureTime = null;
  }
  
  /**
   * 启用或禁用自动降级
   * @param {boolean} enabled - 是否启用
   */
  setAutoFallback(enabled) {
    this.autoFallback = enabled;
  }
  
  /**
   * 启用或禁用自动恢复
   * @param {boolean} enabled - 是否启用
   * @param {number} interval - 恢复检查间隔(毫秒)
   */
  setAutoRecovery(enabled, interval = 300000) {
    this.autoRecovery = enabled;
    this.recoveryInterval = interval;
    
    if (enabled) {
      this._setupRecoveryTimer();
    } else if (this.recoveryTimer) {
      clearTimeout(this.recoveryTimer);
      this.recoveryTimer = null;
    }
  }
}

module.exports = { EnhancedModelManager };
