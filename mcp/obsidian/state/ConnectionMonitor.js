/**
 * ConnectionMonitor
 * 
 * 监控连接状态并管理模式切换
 */

const EventEmitter = require('events');
const https = require('https');
const http = require('http');

// 连接状态枚举
const ConnectionState = {
  ONLINE: 'online',           // 完全在线，Claude可用
  LIMITED: 'limited',         // 有限连接，Claude不可用但DeepSeek可用
  OFFLINE: 'offline',         // 完全离线
  UNKNOWN: 'unknown'          // 未知状态
};

// 模式枚举
const OperationMode = {
  CLAUDE: 'claude',           // 主要模式 - Claude API
  DEEPSEEK: 'deepseek',       // 一级降级 - DeepSeek API
  LOCAL: 'local'              // 二级降级 - 本地模型
};

/**
 * 连接监控器
 */
class ConnectionMonitor extends EventEmitter {
  /**
   * 初始化连接监控器
   * @param {Object} options 配置选项
   */
  constructor(options = {}) {
    super();
    
    this.options = {
      checkInterval: 60000,          // 检查间隔（毫秒）
      claudeEndpoint: 'https://api.anthropic.com/v1/messages',
      deepseekEndpoint: 'https://api.deepseek.com/v1/chat/completions',
      internetTestEndpoint: 'https://www.google.com',
      autoModeSwitch: true,          // 自动模式切换
      ...options
    };
    
    this.state = ConnectionState.UNKNOWN;
    this.currentMode = OperationMode.CLAUDE;
    this.checkTimer = null;
  }
  
  /**
   * 启动监控
   */
  start() {
    // 清除现有计时器
    if (this.checkTimer) {
      clearInterval(this.checkTimer);
    }
    
    // 立即执行一次状态检查
    this.checkConnectionState();
    
    // 设置定期检查
    this.checkTimer = setInterval(() => {
      this.checkConnectionState();
    }, this.options.checkInterval);
    
    this.emit('monitor:started');
  }
  
  /**
   * 停止监控
   */
  stop() {
    if (this.checkTimer) {
      clearInterval(this.checkTimer);
      this.checkTimer = null;
    }
    
    this.emit('monitor:stopped');
  }
  
  /**
   * 检查连接状态
   * @returns {Promise<ConnectionState>} 连接状态
   */
  async checkConnectionState() {
    try {
      // 检查Claude API可用性
      const isClaudeAvailable = await this.checkEndpoint(this.options.claudeEndpoint);
      
      if (isClaudeAvailable) {
        // Claude可用，设置为在线状态
        this._updateState(ConnectionState.ONLINE);
        
        // 自动切换到Claude模式
        if (this.options.autoModeSwitch && this.currentMode !== OperationMode.CLAUDE) {
          this.switchMode(OperationMode.CLAUDE);
        }
        
        return ConnectionState.ONLINE;
      }
      
      // Claude不可用，检查DeepSeek API
      const isDeepSeekAvailable = await this.checkEndpoint(this.options.deepseekEndpoint);
      
      if (isDeepSeekAvailable) {
        // DeepSeek可用，设置为有限连接状态
        this._updateState(ConnectionState.LIMITED);
        
        // 自动切换到DeepSeek模式
        if (this.options.autoModeSwitch && this.currentMode !== OperationMode.DEEPSEEK) {
          this.switchMode(OperationMode.DEEPSEEK);
        }
        
        return ConnectionState.LIMITED;
      }
      
      // 检查一般互联网连接
      const isInternetAvailable = await this.checkEndpoint(this.options.internetTestEndpoint);
      
      if (!isInternetAvailable) {
        // 互联网不可用，设置为离线状态
        this._updateState(ConnectionState.OFFLINE);
        
        // 自动切换到本地模式
        if (this.options.autoModeSwitch && this.currentMode !== OperationMode.LOCAL) {
          this.switchMode(OperationMode.LOCAL);
        }
        
        return ConnectionState.OFFLINE;
      }
      
      // 互联网可用但API不可用，仍设为离线状态（可能是API服务问题）
      this._updateState(ConnectionState.OFFLINE);
      
      // 自动切换到本地模式
      if (this.options.autoModeSwitch && this.currentMode !== OperationMode.LOCAL) {
        this.switchMode(OperationMode.LOCAL);
      }
      
      return ConnectionState.OFFLINE;
    } catch (error) {
      console.error('检查连接状态失败:', error);
      
      // 发生错误，默认设为离线状态
      this._updateState(ConnectionState.OFFLINE);
      
      return ConnectionState.OFFLINE;
    }
  }
  
  /**
   * 检查端点可用性
   * @param {string} endpoint 端点URL
   * @returns {Promise<boolean>} 端点是否可用
   */
  async checkEndpoint(endpoint) {
    return new Promise((resolve) => {
      // 选择合适的HTTP模块
      const httpModule = endpoint.startsWith('https') ? https : http;
      
      // 设置请求选项
      const options = {
        method: 'HEAD',
        timeout: 5000
      };
      
      // 发送请求
      const req = httpModule.request(endpoint, options, (res) => {
        // 任何响应（包括错误响应）都表示端点可达
        resolve(true);
      });
      
      // 处理错误
      req.on('error', () => {
        resolve(false);
      });
      
      // 处理超时
      req.on('timeout', () => {
        req.destroy();
        resolve(false);
      });
      
      // 发送请求
      req.end();
    });
  }
  
  /**
   * 切换操作模式
   * @param {OperationMode} mode 目标模式
   */
  switchMode(mode) {
    // 检查模式是否有效
    if (!Object.values(OperationMode).includes(mode)) {
      throw new Error(`无效的操作模式: ${mode}`);
    }
    
    // 如果模式未变化，不执行任何操作
    if (this.currentMode === mode) {
      return;
    }
    
    // 保存之前的模式
    const previousMode = this.currentMode;
    
    // 更新模式
    this.currentMode = mode;
    
    // 发出模式变更事件
    this.emit('mode:changed', {
      previousMode,
      currentMode: mode,
      timestamp: Date.now()
    });
    
    console.log(`操作模式已切换: ${previousMode} -> ${mode}`);
  }
  
  /**
   * 更新连接状态
   * @param {ConnectionState} newState 新状态
   * @private
   */
  _updateState(newState) {
    // 如果状态未变化，不执行任何操作
    if (this.state === newState) {
      return;
    }
    
    // 保存之前的状态
    const previousState = this.state;
    
    // 更新状态
    this.state = newState;
    
    // 发出状态变更事件
    this.emit('state:changed', {
      previousState,
      currentState: newState,
      timestamp: Date.now()
    });
    
    console.log(`连接状态已变更: ${previousState} -> ${newState}`);
  }
  
  /**
   * 获取当前连接状态
   * @returns {ConnectionState} 当前状态
   */
  getState() {
    return this.state;
  }
  
  /**
   * 获取当前操作模式
   * @returns {OperationMode} 当前模式
   */
  getMode() {
    return this.currentMode;
  }
  
  /**
   * 强制设置操作模式
   * @param {OperationMode} mode 目标模式
   */
  forceMode(mode) {
    // 检查模式是否有效
    if (!Object.values(OperationMode).includes(mode)) {
      throw new Error(`无效的操作模式: ${mode}`);
    }
    
    // 禁用自动切换
    const autoSwitchWasEnabled = this.options.autoModeSwitch;
    this.options.autoModeSwitch = false;
    
    // 切换模式
    this.switchMode(mode);
    
    // 发出强制模式事件
    this.emit('mode:forced', {
      mode,
      timestamp: Date.now()
    });
    
    // 如果之前启用了自动切换，1小时后重新启用
    if (autoSwitchWasEnabled) {
      setTimeout(() => {
        this.options.autoModeSwitch = true;
        console.log('自动模式切换已重新启用');
        
        // 重新检查连接状态
        this.checkConnectionState();
      }, 60 * 60 * 1000); // 1小时
    }
  }
}

module.exports = {
  ConnectionMonitor,
  ConnectionState,
  OperationMode
};
