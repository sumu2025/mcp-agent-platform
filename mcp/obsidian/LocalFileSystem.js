/**
 * 本地文件系统接口
 * 通过直接访问文件系统实现Obsidian笔记操作
 */

const fs = require('fs').promises;
const path = require('path');
const ObsidianInterface = require('./ObsidianInterface');

class LocalFileSystem extends ObsidianInterface {
  /**
   * 初始化本地文件系统接口
   * @param {Object} options 配置选项
   */
  constructor(options = {}) {
    super(options);
    
    if (!options.vaultPath) {
      throw new Error('必须提供Obsidian保管库路径');
    }
    
    this.vaultPath = options.vaultPath;
    this._initialized = false;
  }
  
  /**
   * 初始化接口
   * @returns {Promise<boolean>} 初始化结果
   */
  async initialize() {
    try {
      // 检查vault路径是否存在
      const stats = await fs.stat(this.vaultPath);
      if (!stats.isDirectory()) {
        throw new Error(`保管库路径不是一个目录: ${this.vaultPath}`);
      }
      
      // 检查是否是Obsidian保管库
      const obsidianDir = path.join(this.vaultPath, '.obsidian');
      try {
        const obsStats = await fs.stat(obsidianDir);
        if (!obsStats.isDirectory()) {
          throw new Error(`保管库路径不是一个有效的Obsidian保管库: ${this.vaultPath}`);
        }
      } catch (err) {
        throw new Error(`未找到.obsidian目录，可能不是一个有效的Obsidian保管库: ${this.vaultPath}`);
      }
      
      this._initialized = true;
      return true;
    } catch (error) {
      console.error('初始化本地文件系统接口失败:', error);
      throw error;
    }
  }
  
  /**
   * 获取笔记内容
   * @param {string} notePath 笔记路径
   * @returns {Promise<string>} 笔记内容
   */
  async getNote(notePath) {
    this._ensureInitialized();
    
    try {
      const fullPath = this._resolveNotePath(notePath);
      const content = await fs.readFile(fullPath, 'utf8');
      return content;
    } catch (error) {
      console.error(`获取笔记内容失败 (${notePath}):`, error);
      throw error;
    }
  }
  
  /**
   * 创建或更新笔记
   * @param {string} notePath 笔记路径
   * @param {string} content 笔记内容
   * @returns {Promise<boolean>} 操作结果
   */
  async writeNote(notePath, content) {
    this._ensureInitialized();
    
    try {
      const fullPath = this._resolveNotePath(notePath);
      
      // 确保目录存在
      await this._ensureDirectoryExists(path.dirname(fullPath));
      
      // 写入文件
      await fs.writeFile(fullPath, content, 'utf8');
      return true;
    } catch (error) {
      console.error(`写入笔记失败 (${notePath}):`, error);
      throw error;
    }
  }
  
  /**
   * 删除笔记
   * @param {string} notePath 笔记路径
   * @returns {Promise<boolean>} 操作结果
   */
  async deleteNote(notePath) {
    this._ensureInitialized();
    
    try {
      const fullPath = this._resolveNotePath(notePath);
      await fs.unlink(fullPath);
      return true;
    } catch (error) {
      console.error(`删除笔记失败 (${notePath}):`, error);
      throw error;
    }
  }
  
  /**
   * 获取笔记列表
   * @param {string} folder 文件夹路径
   * @param {Object} options 过滤选项
   * @returns {Promise<Array>} 笔记列表
   */
  async listNotes(folder = '', options = {}) {
    this._ensureInitialized();
    
    try {
      const fullPath = path.join(this.vaultPath, folder);
      const entries = await fs.readdir(fullPath, { withFileTypes: true });
      
      const notes = [];
      
      for (const entry of entries) {
        const entryPath = path.join(folder, entry.name);
        
        // 忽略隐藏文件和文件夹
        if (entry.name.startsWith('.')) continue;
        
        // 递归处理文件夹
        if (entry.isDirectory() && options.recursive) {
          const subNotes = await this.listNotes(entryPath, options);
          notes.push(...subNotes);
          continue;
        }
        
        // 处理文件
        if (entry.isFile() && entry.name.endsWith('.md')) {
          // 应用过滤器
          if (options.filter && !options.filter(entryPath)) {
            continue;
          }
          
          // 获取基本文件信息
          const fullFilePath = path.join(this.vaultPath, entryPath);
          const stats = await fs.stat(fullFilePath);
          
          notes.push({
            path: entryPath,
            name: entry.name,
            created: stats.birthtime,
            modified: stats.mtime,
            size: stats.size
          });
          
          // 如果需要元数据和内容
          if (options.includeMetadata || options.includeContent) {
            const noteInfo = notes[notes.length - 1];
            
            if (options.includeContent) {
              noteInfo.content = await this.getNote(entryPath);
            }
            
            if (options.includeMetadata) {
              noteInfo.metadata = await this.getNoteMetadata(entryPath);
            }
          }
        }
      }
      
      return notes;
    } catch (error) {
      console.error(`获取笔记列表失败 (${folder}):`, error);
      throw error;
    }
  }
  
  /**
   * 搜索笔记
   * @param {string} query 搜索查询
   * @param {Object} options 搜索选项
   * @returns {Promise<Array>} 搜索结果
   */
  async searchNotes(query, options = {}) {
    this._ensureInitialized();
    
    try {
      // 实现简单的字符串搜索
      const notes = await this.listNotes('', { 
        recursive: true, 
        includeContent: true,
        includeMetadata: options.includeMetadata || false
      });
      
      const results = notes.filter(note => {
        // 搜索标题
        if (note.name.toLowerCase().includes(query.toLowerCase())) {
          return true;
        }
        
        // 搜索内容
        if (note.content && note.content.toLowerCase().includes(query.toLowerCase())) {
          return true;
        }
        
        // 搜索元数据
        if (note.metadata) {
          const metadataStr = JSON.stringify(note.metadata).toLowerCase();
          if (metadataStr.includes(query.toLowerCase())) {
            return true;
          }
        }
        
        return false;
      });
      
      // 限制结果数量
      if (options.limit && options.limit > 0) {
        return results.slice(0, options.limit);
      }
      
      return results;
    } catch (error) {
      console.error(`搜索笔记失败 (${query}):`, error);
      throw error;
    }
  }
  
  /**
   * 获取笔记元数据
   * @param {string} notePath 笔记路径
   * @returns {Promise<Object>} 元数据对象
   */
  async getNoteMetadata(notePath) {
    this._ensureInitialized();
    
    try {
      const content = await this.getNote(notePath);
      return this._extractFrontMatter(content);
    } catch (error) {
      console.error(`获取笔记元数据失败 (${notePath}):`, error);
      throw error;
    }
  }
  
  /**
   * 创建文件夹
   * @param {string} folderPath 文件夹路径
   * @returns {Promise<boolean>} 操作结果
   */
  async createFolder(folderPath) {
    this._ensureInitialized();
    
    try {
      const fullPath = path.join(this.vaultPath, folderPath);
      await this._ensureDirectoryExists(fullPath);
      return true;
    } catch (error) {
      console.error(`创建文件夹失败 (${folderPath}):`, error);
      throw error;
    }
  }
  
  /**
   * 获取库信息
   * @returns {Promise<Object>} 库信息
   */
  async getVaultInfo() {
    this._ensureInitialized();
    
    try {
      // 获取基本信息
      const stats = await fs.stat(this.vaultPath);
      
      // 获取配置文件
      let config = {};
      try {
        const configPath = path.join(this.vaultPath, '.obsidian', 'app.json');
        const configContent = await fs.readFile(configPath, 'utf8');
        config = JSON.parse(configContent);
      } catch (err) {
        // 配置文件不可读或格式错误，忽略错误
      }
      
      // 计算文件数量（异步但不等待）
      this._countFiles().then(count => {
        this.totalFiles = count;
      }).catch(err => {
        console.warn('计算文件数量失败:', err);
      });
      
      return {
        path: this.vaultPath,
        name: path.basename(this.vaultPath),
        created: stats.birthtime,
        modified: stats.mtime,
        config: config,
        totalFiles: this.totalFiles || '计算中...'
      };
    } catch (error) {
      console.error('获取库信息失败:', error);
      throw error;
    }
  }
  
  /**
   * 保持连接活跃
   * @returns {Promise<boolean>} 操作结果
   */
  async keepAlive() {
    // 本地文件系统不需要保持连接
    return true;
  }
  
  /**
   * 解析笔记路径
   * @param {string} notePath 笔记路径
   * @returns {string} 完整路径
   * @private
   */
  _resolveNotePath(notePath) {
    // 确保路径有.md扩展名
    if (!notePath.endsWith('.md')) {
      notePath = `${notePath}.md`;
    }
    
    return path.join(this.vaultPath, notePath);
  }
  
  /**
   * 确保目录存在
   * @param {string} dirPath 目录路径
   * @returns {Promise<void>}
   * @private
   */
  async _ensureDirectoryExists(dirPath) {
    try {
      await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
      if (error.code !== 'EEXIST') {
        throw error;
      }
    }
  }
  
  /**
   * 提取YAML前置元数据
   * @param {string} content 笔记内容
   * @returns {Object} 元数据对象
   * @private
   */
  _extractFrontMatter(content) {
    const metadata = {};
    
    // 检查是否有YAML前置元数据
    const match = content.match(/^---\n([\s\S]*?)\n---\n/);
    if (!match) return metadata;
    
    // 简单的YAML解析
    const yamlContent = match[1];
    const lines = yamlContent.split('\n');
    
    for (const line of lines) {
      const keyMatch = line.match(/^([^:]+):\s*(.*)$/);
      if (keyMatch) {
        const key = keyMatch[1].trim();
        let value = keyMatch[2].trim();
        
        // 简单的类型转换
        if (value === 'true') value = true;
        else if (value === 'false') value = false;
        else if (!isNaN(Number(value))) value = Number(value);
        
        metadata[key] = value;
      }
    }
    
    return metadata;
  }
  
  /**
   * 计算库中的文件数量
   * @returns {Promise<number>} 文件数量
   * @private
   */
  async _countFiles() {
    let count = 0;
    
    async function countDir(dir) {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        if (entry.name.startsWith('.')) continue;
        
        const fullPath = path.join(dir, entry.name);
        
        if (entry.isDirectory()) {
          await countDir(fullPath);
        } else if (entry.isFile() && entry.name.endsWith('.md')) {
          count++;
        }
      }
    }
    
    await countDir(this.vaultPath);
    return count;
  }
  
  /**
   * 确保接口已初始化
   * @private
   */
  _ensureInitialized() {
    if (!this._initialized) {
      throw new Error('本地文件系统接口未初始化，请先调用initialize()');
    }
  }
}

module.exports = LocalFileSystem;
