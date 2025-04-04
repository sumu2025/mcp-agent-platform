/**
 * 连接状态监控器
 * 监控Obsidian和各种API的连接状态，触发降级模式
 */

const EventEmitter = require('events');

// 连接状态枚举
const ConnectionStatus = {
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  DEGRADED: 'degraded',
  UNKNOWN: 'unknown'
};

// 服务类型枚举
const ServiceType = {
  CLAUDE: 'claude',
  DEEPSEEK: 'deepseek',
  LOCAL_AI: 'local_ai',
  OBSIDIAN: 'obsidian'
};

class ConnectionMonitor extends EventEmitter {
  /**
   * 初始化连接状态监控器
   * @param {Object} options 配置选项
   */
  constructor(options = {}) {
    super();
    
    this.options = {
      checkInterval: options.checkInterval || 60000, // 默认检查间隔1分钟
      timeout: options.timeout || 5000, // 检查超时时间
      retryAttempts: options.retryAttempts || 3, // 重试次数
      ...options
    };
    
    // 服务状态
    this.serviceStatus = {
      [ServiceType.CLAUDE]: ConnectionStatus.UNKNOWN,
      [ServiceType.DEEPSEEK]: ConnectionStatus.UNKNOWN,
      [ServiceType.LOCAL_AI]: ConnectionStatus.UNKNOWN,
      [ServiceType.OBSIDIAN]: ConnectionStatus.UNKNOWN
    };
    
    // 服务检查器
    this.serviceCheckers = {};
    
    // 全局状态
    this.globalStatus = ConnectionStatus.UNKNOWN;
    
    // 检查定时器
    this.checkTimer = null;
    
    // 是否已启动
    this.running = false;
  }
  
  /**
   * 注册服务检查器
   * @param {string} serviceType 服务类型
   * @param {Function} checker 检查函数，返回Promise<boolean>
   */
  registerChecker(serviceType, checker) {
    if (!Object.values(ServiceType).includes(serviceType)) {
      throw new Error(`未知的服务类型: ${serviceType}`);
    }
    
    if (typeof checker !== 'function') {
      throw new Error('检查器必须是一个函数');
    }
    
    this.serviceCheckers[serviceType] = checker;
  }
  
  /**
   * 注册默认的服务检查器
   * @param {Object} obsidianInterface Obsidian接口实例
   * @param {Object} claudeClient Claude客户端实例
   * @param {Object} deepseekClient DeepSeek客户端实例
   * @param {Object} localAiClient 本地AI客户端实例
   */
  registerDefaultCheckers(obsidianInterface, claudeClient, deepseekClient, localAiClient) {
    // Obsidian检查器
    if (obsidianInterface) {
      this.registerChecker(ServiceType.OBSIDIAN, async () => {
        try {
          return await obsidianInterface.keepAlive();
        } catch (error) {
          return false;
        }
      });
    }
    
    // Claude检查器
    if (claudeClient) {
      this.registerChecker(ServiceType.CLAUDE, async () => {
        try {
          return await claudeClient.checkAvailability();
        } catch (error) {
          return false;
        }
      });
    }
    
    // DeepSeek检查器
    if (deepseekClient) {
      this.registerChecker(ServiceType.DEEPSEEK, async () => {
        try {
          return await deepseekClient.checkAvailability();
        } catch (error) {
          return false;
        }
      });
    }
    
    // 本地AI检查器
    if (localAiClient) {
      this.registerChecker(ServiceType.LOCAL_AI, async () => {
        try {
          return await localAiClient.checkAvailability();
        } catch (error) {
          return false;
        }
      });
    }
  }
  
  /**
   * 启动状态监控
   */
  start() {
    if (this.running) return;
    
    this.running = true;
    
    // 立即进行一次检查
    this.checkAllServices();
    
    // 设置定时检查
    this.checkTimer = setInterval(() => {
      this.checkAllServices();
    }, this.options.checkInterval);
  }
  
  /**
   * 停止状态监控
   */
  stop() {
    if (!this.running) return;
    
    this.running = false;
    
    // 清除定时器
    if (this.checkTimer) {
      clearInterval(this.checkTimer);
      this.checkTimer = null;
    }
  }
  
  /**
   * 检查所有服务状态
   */
  async checkAllServices() {
    const results = await Promise.all(
      Object.entries(this.serviceCheckers).map(async ([serviceType, checker]) => {
        const status = await this._checkServiceWithRetry(serviceType, checker);
        return { serviceType, status };
      })
    );
    
    // 更新服务状态
    let statusChanged = false;
    
    results.forEach(({ serviceType, status }) => {
      if (this.serviceStatus[serviceType] !== status) {
        statusChanged = true;
        
        // 记录状态变更
        const oldStatus = this.serviceStatus[serviceType];
        this.serviceStatus[serviceType] = status;
        
        // 触发服务状态变更事件
        this.emit('serviceStatusChanged', {
          service: serviceType,
          oldStatus,
          newStatus: status
        });
      }
    });
    
    // 更新全局状态
    const newGlobalStatus = this._determineGlobalStatus();
    
    if (newGlobalStatus !== this.globalStatus) {
      const oldGlobalStatus = this.globalStatus;
      this.globalStatus = newGlobalStatus;
      
      // 触发全局状态变更事件
      this.emit('globalStatusChanged', {
        oldStatus: oldGlobalStatus,
        newStatus: newGlobalStatus
      });
    }
    
    // 如果任何状态发生变化，触发状态更新事件
    if (statusChanged || newGlobalStatus !== this.globalStatus) {
      this.emit('statusUpdated', {
        globalStatus: this.globalStatus,
        serviceStatus: { ...this.serviceStatus }
      });
    }
  }
  
  /**
   * 带重试的服务检查
   * @param {string} serviceType 服务类型
   * @param {Function} checker 检查函数
   * @returns {Promise<string>} 连接状态
   * @private
   */
  async _checkServiceWithRetry(serviceType, checker) {
    let attempts = 0;
    
    while (attempts < this.options.retryAttempts) {
      try {
        // 设置超时
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error('检查超时')), this.options.timeout);
        });
        
        // 执行检查
        const checkPromise = checker();
        
        // 等待结果（带超时）
        const available = await Promise.race([checkPromise, timeoutPromise]);
        
        return available ? ConnectionStatus.CONNECTED : ConnectionStatus.DISCONNECTED;
      } catch (error) {
        attempts++;
        
        // 最后一次尝试失败
        if (attempts >= this.options.retryAttempts) {
          console.warn(`服务 ${serviceType} 检查失败: ${error.message}`);
          return ConnectionStatus.DISCONNECTED;
        }
        
        // 等待后重试
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    return ConnectionStatus.DISCONNECTED;
  }
  
  /**
   * 确定全局连接状态
   * @returns {string} 全局连接状态
   * @private
   */
  _determineGlobalStatus() {
    // 优先级：Obsidian > Claude > DeepSeek > 本地AI
    
    // Obsidian离线，系统无法工作
    if (this.serviceStatus[ServiceType.OBSIDIAN] === ConnectionStatus.DISCONNECTED) {
      return ConnectionStatus.DISCONNECTED;
    }
    
    // Claude在线，系统完全工作
    if (this.serviceStatus[ServiceType.CLAUDE] === ConnectionStatus.CONNECTED) {
      return ConnectionStatus.CONNECTED;
    }
    
    // Claude离线但DeepSeek在线，降级模式1
    if (this.serviceStatus[ServiceType.DEEPSEEK] === ConnectionStatus.CONNECTED) {
      return ConnectionStatus.DEGRADED;
    }
    
    // Claude和DeepSeek都离线但本地AI可用，降级模式2
    if (this.serviceStatus[ServiceType.LOCAL_AI] === ConnectionStatus.CONNECTED) {
      return ConnectionStatus.DEGRADED;
    }
    
    // 所有AI服务都不可用
    return ConnectionStatus.DISCONNECTED;
  }
  
  /**
   * 获取推荐的服务
   * @returns {string} 推荐的服务类型
   */
  getRecommendedService() {
    // 优先使用Claude
    if (this.serviceStatus[ServiceType.CLAUDE] === ConnectionStatus.CONNECTED) {
      return ServiceType.CLAUDE;
    }
    
    // 其次使用DeepSeek
    if (this.serviceStatus[ServiceType.DEEPSEEK] === ConnectionStatus.CONNECTED) {
      return ServiceType.DEEPSEEK;
    }
    
    // 最后使用本地AI
    if (this.serviceStatus[ServiceType.LOCAL_AI] === ConnectionStatus.CONNECTED) {
      return ServiceType.LOCAL_AI;
    }
    
    // 所有服务都不可用
    return null;
  }
  
  /**
   * 获取当前的全局状态
   * @returns {string} 全局状态
   */
  getGlobalStatus() {
    return this.globalStatus;
  }
  
  /**
   * 获取服务状态
   * @param {string} serviceType 服务类型
   * @returns {string} 服务状态
   */
  getServiceStatus(serviceType) {
    return this.serviceStatus[serviceType] || ConnectionStatus.UNKNOWN;
  }
  
  /**
   * 获取所有服务状态
   * @returns {Object} 服务状态映射
   */
  getAllServiceStatus() {
    return { ...this.serviceStatus };
  }
  
  /**
   * 手动设置服务状态（用于测试）
   * @param {string} serviceType 服务类型
   * @param {string} status 状态
   */
  setServiceStatus(serviceType, status) {
    if (!Object.values(ServiceType).includes(serviceType)) {
      throw new Error(`未知的服务类型: ${serviceType}`);
    }
    
    if (!Object.values(ConnectionStatus).includes(status)) {
      throw new Error(`未知的连接状态: ${status}`);
    }
    
    const oldStatus = this.serviceStatus[serviceType];
    this.serviceStatus[serviceType] = status;
    
    // 触发服务状态变更事件
    this.emit('serviceStatusChanged', {
      service: serviceType,
      oldStatus,
      newStatus: status
    });
    
    // 更新全局状态
    const newGlobalStatus = this._determineGlobalStatus();
    
    if (newGlobalStatus !== this.globalStatus) {
      const oldGlobalStatus = this.globalStatus;
      this.globalStatus = newGlobalStatus;
      
      // 触发全局状态变更事件
      this.emit('globalStatusChanged', {
        oldStatus: oldGlobalStatus,
        newStatus: newGlobalStatus
      });
    }
    
    // 触发状态更新事件
    this.emit('statusUpdated', {
      globalStatus: this.globalStatus,
      serviceStatus: { ...this.serviceStatus }
    });
  }
}

// 导出类和枚举
module.exports = {
  ConnectionMonitor,
  ConnectionStatus,
  ServiceType
};
