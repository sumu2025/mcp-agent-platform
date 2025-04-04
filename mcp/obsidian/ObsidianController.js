/**
 * Obsidian主控制器
 * 整合所有Obsidian相关组件，提供统一的接口
 */

const LocalFileSystem = require('./LocalFileSystem');
const CommandParser = require('./CommandParser');
const { ConnectionMonitor, ServiceType } = require('./ConnectionMonitor');
const { EventEmitter } = require('events');

class ObsidianController extends EventEmitter {
  /**
   * 初始化Obsidian控制器
   * @param {Object} options 配置选项
   */
  constructor(options = {}) {
    super();
    
    this.options = options;
    
    // 创建文件系统接口
    this.fileSystem = new LocalFileSystem({
      vaultPath: options.vaultPath,
      ...options.fileSystemOptions
    });
    
    // 创建命令解析器
    this.commandParser = new CommandParser(this.fileSystem);
    
    // 创建连接监控器
    this.connectionMonitor = new ConnectionMonitor(options.monitorOptions);
    
    // 注册监控器事件
    this.connectionMonitor.on('statusUpdated', (status) => {
      this.emit('connectionStatusUpdated', status);
    });
    
    this.connectionMonitor.on('globalStatusChanged', (statusChange) => {
      this.emit('globalStatusChanged', statusChange);
    });
    
    // 控制器状态
    this._initialized = false;
  }
  
  /**
   * 初始化控制器
   * @returns {Promise<boolean>} 初始化结果
   */
  async initialize() {
    if (this._initialized) {
      return true;
    }
    
    try {
      // 初始化文件系统接口
      await this.fileSystem.initialize();
      
      // 注册Obsidian检查器
      this.connectionMonitor.registerChecker(ServiceType.OBSIDIAN, async () => {
        try {
          await this.fileSystem.getVaultInfo();
          return true;
        } catch (error) {
          return false;
        }
      });
      
      // 启动连接监控
      this.connectionMonitor.start();
      
      this._initialized = true;
      return true;
    } catch (error) {
      console.error('初始化Obsidian控制器失败:', error);
      throw error;
    }
  }
  
  /**
   * 注册服务客户端
   * @param {string} serviceType 服务类型
   * @param {Object} client 客户端实例
   */
  registerServiceClient(serviceType, client) {
    if (serviceType === ServiceType.CLAUDE || 
        serviceType === ServiceType.DEEPSEEK || 
        serviceType === ServiceType.LOCAL_AI) {
      
      // 注册服务检查器
      this.connectionMonitor.registerChecker(serviceType, async () => {
        try {
          if (typeof client.checkAvailability === 'function') {
            return await client.checkAvailability();
          } else if (typeof client.isAvailable === 'function') {
            return await client.isAvailable();
          } else {
            // 简单的可用性检查
            return client !== null && client !== undefined;
          }
        } catch (error) {
          return false;
        }
      });
      
      this.emit('serviceClientRegistered', { serviceType });
    } else {
      throw new Error(`不支持的服务类型: ${serviceType}`);
    }
  }
  
  /**
   * 执行Obsidian命令
   * @param {string|Object} command 命令字符串或对象
   * @returns {Promise<Object>} 执行结果
   */
  async executeCommand(command) {
    this._ensureInitialized();
    
    // 解析命令
    const parsedCommand = typeof command === 'string' 
      ? this.commandParser.parseCommand(command)
      : command;
    
    // 执行命令
    const result = await this.commandParser.executeCommand(parsedCommand);
    
    // 触发命令执行事件
    this.emit('commandExecuted', {
      command: parsedCommand,
      result
    });
    
    return result;
  }
  
  /**
   * 获取笔记内容
   * @param {string} path 笔记路径
   * @returns {Promise<string>} 笔记内容
   */
  async getNote(path) {
    this._ensureInitialized();
    return this.fileSystem.getNote(path);
  }
  
  /**
   * 创建或更新笔记
   * @param {string} path 笔记路径
   * @param {string} content 笔记内容
   * @returns {Promise<boolean>} 操作结果
   */
  async writeNote(path, content) {
    this._ensureInitialized();
    return this.fileSystem.writeNote(path, content);
  }
  
  /**
   * 附加内容到笔记
   * @param {string} path 笔记路径
   * @param {string} content 要附加的内容
   * @returns {Promise<boolean>} 操作结果
   */
  async appendToNote(path, content) {
    this._ensureInitialized();
    
    try {
      // 读取现有内容
      let existingContent = '';
      try {
        existingContent = await this.fileSystem.getNote(path);
      } catch (error) {
        // 笔记不存在，创建新笔记
        existingContent = '';
      }
      
      // 拼接内容
      const newContent = existingContent.trim() 
        ? `${existingContent}\n\n${content}`
        : content;
      
      // 写入笔记
      return this.fileSystem.writeNote(path, newContent);
    } catch (error) {
      console.error(`附加到笔记失败 (${path}):`, error);
      throw error;
    }
  }
  
  /**
   * S删除笔记
   * @param {string} path 笔记路径
   * @returns {Promise<boolean>} 操作结果
   */
  async deleteNote(path) {
    this._ensureInitialized();
    return this.fileSystem.deleteNote(path);
  }
  
  /**
   * 获取笔记列表
   * @param {string} folder 文件夹路径
   * @param {Object} options 过滤选项
   * @returns {Promise<Array>} 笔记列表
   */
  async listNotes(folder = '', options = {}) {
    this._ensureInitialized();
    return this.fileSystem.listNotes(folder, options);
  }
  
  /**
   * 搜索笔记
   * @param {string} query 搜索查询
   * @param {Object} options 搜索选项
   * @returns {Promise<Array>} 搜索结果
   */
  async searchNotes(query, options = {}) {
    this._ensureInitialized();
    return this.fileSystem.searchNotes(query, options);
  }
  
  /**
   * 获取笔记元数据
   * @param {string} path 笔记路径
   * @returns {Promise<Object>} 元数据对象
   */
  async getNoteMetadata(path) {
    this._ensureInitialized();
    return this.fileSystem.getNoteMetadata(path);
  }
  
  /**
   * 创建文件夹
   * @param {string} path 文件夹路径
   * @returns {Promise<boolean>} 操作结果
   */
  async createFolder(path) {
    this._ensureInitialized();
    return this.fileSystem.createFolder(path);
  }
  
  /**
   * 获取库信息
   * @returns {Promise<Object>} 库信息
   */
  async getVaultInfo() {
    this._ensureInitialized();
    return this.fileSystem.getVaultInfo();
  }
  
  /**
   * 获取当前连接状态
   * @returns {Object} 连接状态
   */
  getConnectionStatus() {
    return {
      globalStatus: this.connectionMonitor.getGlobalStatus(),
      serviceStatus: this.connectionMonitor.getAllServiceStatus(),
      recommendedService: this.connectionMonitor.getRecommendedService()
    };
  }
  
  /**
   * 停止控制器
   */
  stop() {
    if (this.connectionMonitor) {
      this.connectionMonitor.stop();
    }
    
    this._initialized = false;
  }
  
  /**
   * 确保控制器已初始化
   * @private
   */
  _ensureInitialized() {
    if (!this._initialized) {
      throw new Error('Obsidian控制器未初始化，请先调用initialize()');
    }
  }
}

module.exports = ObsidianController;
