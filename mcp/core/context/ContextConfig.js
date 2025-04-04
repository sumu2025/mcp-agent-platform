/**
 * 上下文系统配置
 * 管理上下文相关的所有配置项
 */

class ContextConfig {
  /**
   * 构造函数
   * @param {Object} options 配置选项
   */
  constructor(options = {}) {
    /**
     * 项目元数据路径
     * @type {string}
     */
    this.metadataPath = options.metadataPath;
    
    /**
     * 会话摘要存储路径
     * @type {string}
     */
    this.summaryPath = options.summaryPath;
    
    /**
     * 保留的最大摘要数量
     * @type {number}
     */
    this.maxSummaries = options.maxSummaries || 10;
    
    /**
     * 加载的默认摘要数量
     * @type {number}
     */
    this.defaultSummaryCount = options.defaultSummaryCount || 3;
    
    /**
     * 自动生成摘要
     * @type {boolean}
     */
    this.autoSummarize = options.autoSummarize !== false;
    
    /**
     * 会话结束时自动保存摘要
     * @type {boolean}
     */
    this.autoSave = options.autoSave !== false;
    
    /**
     * 自动更新项目元数据
     * @type {boolean}
     */
    this.autoUpdateMetadata = options.autoUpdateMetadata !== false;
    
    /**
     * 加载元数据指定的部分
     * @type {Array}
     */
    this.loadSections = options.loadSections || [
      'project',
      'architecture',
      'developmentPlan',
      'progress',
      'context'
    ];
    
    /**
     * 上下文提示最大长度
     * @type {number}
     */
    this.maxContextLength = options.maxContextLength || 4000;
    
    /**
     * 显示元数据更新通知
     * @type {boolean}
     */
    this.showUpdateNotifications = options.showUpdateNotifications !== false;
    
    /**
     * 自定义提示模板
     * @type {string}
     */
    this.promptTemplate = options.promptTemplate || null;
  }
  
  /**
   * 创建默认配置
   * @param {string} basePath 基础路径
   * @returns {ContextConfig} 配置对象
   */
  static createDefault(basePath) {
    const path = require('path');
    
    return new ContextConfig({
      metadataPath: path.join(basePath, 'project_metadata.json'),
      summaryPath: path.join(basePath, 'session_summaries'),
      maxSummaries: 10,
      defaultSummaryCount: 3,
      autoSummarize: true,
      autoSave: true,
      autoUpdateMetadata: true,
      maxContextLength: 4000
    });
  }
  
  /**
   * 加载配置
   * @param {string} configPath 配置文件路径
   * @returns {ContextConfig} 配置对象
   */
  static load(configPath) {
    try {
      const fs = require('fs');
      if (fs.existsSync(configPath)) {
        const configData = fs.readFileSync(configPath, 'utf8');
        const options = JSON.parse(configData);
        return new ContextConfig(options);
      }
    } catch (error) {
      console.warn(`加载上下文配置失败: ${error.message}`);
    }
    
    return ContextConfig.createDefault();
  }
  
  /**
   * 保存配置
   * @param {string} configPath 配置文件路径
   * @returns {boolean} 是否成功
   */
  save(configPath) {
    try {
      const fs = require('fs');
      const path = require('path');
      
      // 确保目录存在
      const configDir = path.dirname(configPath);
      if (!fs.existsSync(configDir)) {
        fs.mkdirSync(configDir, { recursive: true });
      }
      
      // 保存配置
      fs.writeFileSync(configPath, JSON.stringify(this, null, 2));
      return true;
    } catch (error) {
      console.error(`保存上下文配置失败: ${error.message}`);
      return false;
    }
  }
}

module.exports = ContextConfig;
