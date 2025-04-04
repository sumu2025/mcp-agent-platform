/**
 * ObsidianVault
 * 
 * 提供对Obsidian知识库的文件系统访问
 */

const fs = require('fs').promises;
const path = require('path');
const EventEmitter = require('events');
const chokidar = require('chokidar'); // 文件监视库，可能需要安装: npm install chokidar

/**
 * 表示Obsidian笔记的类
 */
class ObsidianNote {
  /**
   * 创建笔记对象
   * @param {string} path 笔记路径
   * @param {string} content 笔记内容
   * @param {Object} metadata 笔记元数据
   */
  constructor(path, content = null, metadata = {}) {
    this.path = path;
    this.filename = path.split('/').pop();
    this.content = content;
    this.metadata = metadata;
    this.links = [];
    this.backlinks = [];
  }
  
  /**
   * 获取笔记ID (路径的散列)
   * @returns {string} 笔记ID
   */
  get id() {
    return Buffer.from(this.path).toString('base64').replace(/[\/\+\=]/g, '');
  }
}

/**
 * Obsidian知识库管理器
 */
class ObsidianVault extends EventEmitter {
  /**
   * 初始化知识库管理器
   * @param {string} vaultPath 知识库路径
   * @param {Object} options 配置选项
   */
  constructor(vaultPath, options = {}) {
    super();
    
    this.vaultPath = vaultPath;
    this.options = {
      watchFiles: true,
      excludedFolders: ['.git', '.obsidian', 'node_modules'],
      excludedExtensions: ['.DS_Store', '.gitignore'],
      ...options
    };
    
    this.watcher = null;
    this.noteCache = new Map(); // 路径 -> ObsidianNote
    this._initialized = false;
  }
  
  /**
   * 初始化知识库
   */
  async initialize() {
    if (this._initialized) return;
    
    try {
      // 验证知识库路径
      const stats = await fs.stat(this.vaultPath);
      if (!stats.isDirectory()) {
        throw new Error(`知识库路径不是目录: ${this.vaultPath}`);
      }
      
      // 开始监视文件变更
      if (this.options.watchFiles) {
        this._setupWatcher();
      }
      
      this._initialized = true;
      this.emit('initialized', { path: this.vaultPath });
      
    } catch (error) {
      throw new Error(`初始化知识库失败: ${error.message}`);
    }
  }
  
  /**
   * 获取笔记
   * @param {string} notePath 笔记路径
   * @param {boolean} useCache 是否使用缓存
   * @returns {Promise<ObsidianNote>} 笔记对象
   */
  async getNote(notePath, useCache = true) {
    this._ensureInitialized();
    
    // 构建完整路径
    const fullPath = this._getFullPath(notePath);
    
    // 检查缓存
    if (useCache && this.noteCache.has(notePath)) {
      return this.noteCache.get(notePath);
    }
    
    try {
      // 读取文件内容
      const content = await fs.readFile(fullPath, 'utf-8');
      
      // 解析元数据
      const { metadata, text } = this._parseMetadata(content);
      
      // 创建笔记对象
      const note = new ObsidianNote(notePath, text, metadata);
      
      // 解析链接
      note.links = this._parseLinks(text);
      
      // 添加到缓存
      this.noteCache.set(notePath, note);
      
      return note;
    } catch (error) {
      throw new Error(`读取笔记失败 ${notePath}: ${error.message}`);
    }
  }
  
  /**
   * 保存笔记
   * @param {string} notePath 笔记路径
   * @param {string} content 笔记内容
   * @param {Object} metadata 笔记元数据
   * @returns {Promise<ObsidianNote>} 保存的笔记
   */
  async saveNote(notePath, content, metadata = null) {
    this._ensureInitialized();
    
    try {
      // 构建完整路径
      const fullPath = this._getFullPath(notePath);
      
      // 确保目录存在
      await this._ensureDirectory(path.dirname(fullPath));
      
      // 获取现有笔记（如果存在）
      let existingNote = null;
      try {
        existingNote = await this.getNote(notePath, false);
      } catch (error) {
        // 笔记可能不存在，这是正常的
      }
      
      // 如果提供了元数据，合并现有元数据
      let finalContent = content;
      if (metadata) {
        const existingMetadata = existingNote ? existingNote.metadata : {};
        const mergedMetadata = { ...existingMetadata, ...metadata };
        finalContent = this._addMetadataToContent(content, mergedMetadata);
      }
      
      // 写入文件
      await fs.writeFile(fullPath, finalContent, 'utf-8');
      
      // 更新缓存
      const { parsedMetadata, text } = this._parseMetadata(finalContent);
      const note = new ObsidianNote(notePath, text, parsedMetadata);
      note.links = this._parseLinks(text);
      this.noteCache.set(notePath, note);
      
      // 发出事件
      this.emit('noteChanged', { path: notePath, action: existingNote ? 'update' : 'create' });
      
      return note;
    } catch (error) {
      throw new Error(`保存笔记失败 ${notePath}: ${error.message}`);
    }
  }
  
  /**
   * 删除笔记
   * @param {string} notePath 笔记路径
   * @returns {Promise<boolean>} 是否成功删除
   */
  async deleteNote(notePath) {
    this._ensureInitialized();
    
    try {
      // 构建完整路径
      const fullPath = this._getFullPath(notePath);
      
      // 检查文件是否存在
      try {
        await fs.access(fullPath);
      } catch (error) {
        throw new Error(`笔记不存在: ${notePath}`);
      }
      
      // 删除文件
      await fs.unlink(fullPath);
      
      // 从缓存中移除
      this.noteCache.delete(notePath);
      
      // 发出事件
      this.emit('noteChanged', { path: notePath, action: 'delete' });
      
      return true;
    } catch (error) {
      throw new Error(`删除笔记失败 ${notePath}: ${error.message}`);
    }
  }
  
  /**
   * 列出目录中的文件和文件夹
   * @param {string} dirPath 目录路径
   * @returns {Promise<Object>} 包含文件和文件夹列表的对象
   */
  async listDirectory(dirPath = '') {
    this._ensureInitialized();
    
    try {
      // 构建完整路径
      const fullPath = this._getFullPath(dirPath);
      
      // 读取目录内容
      const entries = await fs.readdir(fullPath, { withFileTypes: true });
      
      // 分类结果
      const result = {
        files: [],
        directories: []
      };
      
      for (const entry of entries) {
        const entryPath = path.join(dirPath, entry.name);
        
        // 排除指定的文件和文件夹
        if (this._shouldExclude(entryPath)) {
          continue;
        }
        
        if (entry.isDirectory()) {
          result.directories.push({
            path: entryPath,
            name: entry.name
          });
        } else if (entry.isFile()) {
          result.files.push({
            path: entryPath,
            name: entry.name,
            extension: path.extname(entry.name)
          });
        }
      }
      
      return result;
    } catch (error) {
      throw new Error(`列出目录失败 ${dirPath}: ${error.message}`);
    }
  }
  
  /**
   * 搜索笔记
   * @param {string} query 搜索查询
   * @param {Object} options 搜索选项
   * @returns {Promise<Array<ObsidianNote>>} 匹配的笔记列表
   */
  async searchNotes(query, options = {}) {
    this._ensureInitialized();
    
    const searchOptions = {
      caseSensitive: false,
      includeMetadata: true,
      limit: 20,
      ...options
    };
    
    const results = [];
    const searchRegex = new RegExp(query, searchOptions.caseSensitive ? 'g' : 'gi');
    
    try {
      // 递归扫描文件
      const scanDirectory = async (dirPath) => {
        const { files, directories } = await this.listDirectory(dirPath);
        
        // 搜索文件
        for (const file of files) {
          // 检查文件是否为Markdown
          if (path.extname(file.name).toLowerCase() !== '.md') {
            continue;
          }
          
          // 获取笔记内容
          const note = await this.getNote(file.path);
          
          // 搜索内容
          if (searchRegex.test(note.content)) {
            results.push(note);
            
            // 如果达到限制，停止搜索
            if (results.length >= searchOptions.limit) {
              return;
            }
          } else if (searchOptions.includeMetadata) {
            // 搜索元数据
            const metadataString = JSON.stringify(note.metadata);
            if (searchRegex.test(metadataString)) {
              results.push(note);
              
              // 如果达到限制，停止搜索
              if (results.length >= searchOptions.limit) {
                return;
              }
            }
          }
        }
        
        // 递归搜索子目录
        for (const directory of directories) {
          await scanDirectory(directory.path);
          
          // 如果达到限制，停止搜索
          if (results.length >= searchOptions.limit) {
            return;
          }
        }
      };
      
      // 开始扫描
      await scanDirectory('');
      
      return results;
    } catch (error) {
      throw new Error(`搜索笔记失败: ${error.message}`);
    }
  }
  
  /**
   * 创建目录
   * @param {string} dirPath 目录路径
   * @returns {Promise<boolean>} 是否成功创建
   */
  async createDirectory(dirPath) {
    this._ensureInitialized();
    
    try {
      // 构建完整路径
      const fullPath = this._getFullPath(dirPath);
      
      // 创建目录
      await this._ensureDirectory(fullPath);
      
      return true;
    } catch (error) {
      throw new Error(`创建目录失败 ${dirPath}: ${error.message}`);
    }
  }
  
  /**
   * 关闭知识库管理器
   */
  close() {
    if (this.watcher) {
      this.watcher.close();
      this.watcher = null;
    }
    
    this.noteCache.clear();
    this._initialized = false;
  }
  
  /**
   * 确保初始化完成
   * @private
   */
  _ensureInitialized() {
    if (!this._initialized) {
      throw new Error('知识库管理器未初始化，请先调用initialize()');
    }
  }
  
  /**
   * 获取相对路径的完整路径
   * @param {string} relativePath 相对路径
   * @returns {string} 完整路径
   * @private
   */
  _getFullPath(relativePath) {
    return path.join(this.vaultPath, relativePath);
  }
  
  /**
   * 确保目录存在
   * @param {string} dirPath 目录路径
   * @returns {Promise<void>}
   * @private
   */
  async _ensureDirectory(dirPath) {
    await fs.mkdir(dirPath, { recursive: true });
  }
  
  /**
   * 设置文件监视器
   * @private
   */
  _setupWatcher() {
    // 创建文件监视器
    this.watcher = chokidar.watch(this.vaultPath, {
      ignored: (filePath) => {
        const relativePath = path.relative(this.vaultPath, filePath);
        return this._shouldExclude(relativePath);
      },
      persistent: true,
      ignoreInitial: true
    });
    
    // 添加事件处理
    this.watcher
      .on('add', filePath => {
        const relativePath = path.relative(this.vaultPath, filePath);
        this.emit('noteChanged', { path: relativePath, action: 'create' });
        this.noteCache.delete(relativePath); // 清除缓存
      })
      .on('change', filePath => {
        const relativePath = path.relative(this.vaultPath, filePath);
        this.emit('noteChanged', { path: relativePath, action: 'update' });
        this.noteCache.delete(relativePath); // 清除缓存
      })
      .on('unlink', filePath => {
        const relativePath = path.relative(this.vaultPath, filePath);
        this.emit('noteChanged', { path: relativePath, action: 'delete' });
        this.noteCache.delete(relativePath); // 清除缓存
      });
  }
  
  /**
   * 判断是否应该排除文件或目录
   * @param {string} relativePath 相对路径
   * @returns {boolean} 是否应该排除
   * @private
   */
  _shouldExclude(relativePath) {
    // 检查是否在排除的文件夹中
    for (const folder of this.options.excludedFolders) {
      if (relativePath.startsWith(folder + path.sep) || relativePath === folder) {
        return true;
      }
    }
    
    // 检查是否有排除的扩展名
    for (const ext of this.options.excludedExtensions) {
      if (relativePath.endsWith(ext)) {
        return true;
      }
    }
    
    return false;
  }
  
  /**
   * 解析笔记元数据
   * @param {string} content 笔记内容
   * @returns {Object} 包含元数据和正文的对象
   * @private
   */
  _parseMetadata(content) {
    const metadata = {};
    let text = content;
    
    // 检查是否有YAML前置元数据
    const yamlRegex = /^---\n([\s\S]*?)\n---\n/;
    const yamlMatch = content.match(yamlRegex);
    
    if (yamlMatch) {
      // 提取YAML内容
      const yamlContent = yamlMatch[1];
      
      // 解析YAML（简单实现，实际应使用yaml库）
      yamlContent.split('\n').forEach(line => {
        const match = line.match(/^(.+?):\s*(.+)$/);
        if (match) {
          const [, key, value] = match;
          metadata[key.trim()] = value.trim();
        }
      });
      
      // 从内容中移除YAML部分
      text = content.replace(yamlMatch[0], '');
    }
    
    // 解析Obsidian属性（如Property:: Value）
    const propertyRegex = /^([^:]+)::\s*(.+)$/gm;
    let propertyMatch;
    
    while ((propertyMatch = propertyRegex.exec(text)) !== null) {
      const [, key, value] = propertyMatch;
      metadata[key.trim()] = value.trim();
    }
    
    return { metadata, text };
  }
  
  /**
   * 将元数据添加到内容
   * @param {string} content 笔记内容
   * @param {Object} metadata 元数据
   * @returns {string} 添加了元数据的内容
   * @private
   */
  _addMetadataToContent(content, metadata) {
    if (!metadata || Object.keys(metadata).length === 0) {
      return content;
    }
    
    // 构建YAML前置元数据
    let yamlContent = '---\n';
    for (const [key, value] of Object.entries(metadata)) {
      yamlContent += `${key}: ${value}\n`;
    }
    yamlContent += '---\n\n';
    
    // 检查内容是否已有YAML前置元数据
    const yamlRegex = /^---\n[\s\S]*?\n---\n/;
    if (yamlRegex.test(content)) {
      // 替换现有的YAML部分
      return content.replace(yamlRegex, yamlContent);
    } else {
      // 添加到内容前面
      return yamlContent + content;
    }
  }
  
  /**
   * 解析内部链接
   * @param {string} content 笔记内容
   * @returns {Array<string>} 链接数组
   * @private
   */
  _parseLinks(content) {
    const links = [];
    const linkRegex = /\[\[([^\]]+)\]\]/g;
    let match;
    
    while ((match = linkRegex.exec(content)) !== null) {
      links.push(match[1]);
    }
    
    return links;
  }
}

module.exports = {
  ObsidianVault,
  ObsidianNote
};
