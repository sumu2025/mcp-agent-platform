/**
 * Obsidian接口
 * 定义与Obsidian交互的核心接口
 */

class ObsidianInterface {
  /**
   * 初始化Obsidian接口
   * @param {Object} options 配置选项
   */
  constructor(options = {}) {
    this.vaultPath = options.vaultPath || null;
    this.options = options;
  }
  
  /**
   * 初始化接口
   * @returns {Promise<boolean>} 初始化结果
   */
  async initialize() {
    throw new Error('ObsidianInterface.initialize() 必须由子类实现');
  }
  
  /**
   * 获取笔记内容
   * @param {string} path 笔记路径
   * @returns {Promise<string>} 笔记内容
   */
  async getNote(path) {
    throw new Error('ObsidianInterface.getNote() 必须由子类实现');
  }
  
  /**
   * 创建或更新笔记
   * @param {string} path 笔记路径
   * @param {string} content 笔记内容
   * @returns {Promise<boolean>} 操作结果
   */
  async writeNote(path, content) {
    throw new Error('ObsidianInterface.writeNote() 必须由子类实现');
  }
  
  /**
   * 删除笔记
   * @param {string} path 笔记路径
   * @returns {Promise<boolean>} 操作结果
   */
  async deleteNote(path) {
    throw new Error('ObsidianInterface.deleteNote() 必须由子类实现');
  }
  
  /**
   * 获取笔记列表
   * @param {string} folder 文件夹路径
   * @param {Object} options 过滤选项
   * @returns {Promise<Array>} 笔记列表
   */
  async listNotes(folder, options = {}) {
    throw new Error('ObsidianInterface.listNotes() 必须由子类实现');
  }
  
  /**
   * 搜索笔记
   * @param {string} query 搜索查询
   * @param {Object} options 搜索选项
   * @returns {Promise<Array>} 搜索结果
   */
  async searchNotes(query, options = {}) {
    throw new Error('ObsidianInterface.searchNotes() 必须由子类实现');
  }
  
  /**
   * 获取笔记元数据
   * @param {string} path 笔记路径
   * @returns {Promise<Object>} 元数据对象
   */
  async getNoteMetadata(path) {
    throw new Error('ObsidianInterface.getNoteMetadata() 必须由子类实现');
  }
  
  /**
   * 创建文件夹
   * @param {string} path 文件夹路径
   * @returns {Promise<boolean>} 操作结果
   */
  async createFolder(path) {
    throw new Error('ObsidianInterface.createFolder() 必须由子类实现');
  }
  
  /**
   * 获取库信息
   * @returns {Promise<Object>} 库信息
   */
  async getVaultInfo() {
    throw new Error('ObsidianInterface.getVaultInfo() 必须由子类实现');
  }
  
  /**
   * 保持连接活跃
   * @returns {Promise<boolean>} 操作结果
   */
  async keepAlive() {
    throw new Error('ObsidianInterface.keepAlive() 必须由子类实现');
  }
}

module.exports = ObsidianInterface;
