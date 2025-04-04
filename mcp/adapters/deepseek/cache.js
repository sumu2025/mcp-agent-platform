/**
 * DeepSeek响应缓存系统
 * 用于缓存API响应，减少API调用频率
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class ResponseCache {
  /**
   * 初始化缓存系统
   * @param {Object} options 缓存配置
   * @param {string} options.cacheDir 缓存目录
   * @param {number} options.maxAge 最大缓存时间（毫秒），默认24小时
   * @param {number} options.maxSize 最大缓存大小（条目数），默认100
   */
  constructor(options = {}) {
    this.options = {
      cacheDir: options.cacheDir || path.join(process.env.HOME || process.env.USERPROFILE, '.mcp', 'cache', 'deepseek'),
      maxAge: options.maxAge || 24 * 60 * 60 * 1000, // 24小时
      maxSize: options.maxSize || 100,
      ...options
    };
    
    // 确保缓存目录存在
    if (!fs.existsSync(this.options.cacheDir)) {
      fs.mkdirSync(this.options.cacheDir, { recursive: true });
    }
    
    // 缓存统计
    this.stats = {
      hits: 0,
      misses: 0,
      size: 0
    };
    
    // 加载现有缓存信息
    this._loadCacheInfo();
  }
  
  /**
   * 从请求参数生成缓存键
   * @param {Object} params 请求参数
   * @returns {string} 缓存键
   */
  generateKey(params) {
    // 提取关键参数用于缓存键生成
    const keyParams = {
      model: params.model,
      messages: params.messages,
      temperature: params.temperature || 0.7,
      max_tokens: params.max_tokens
      // 忽略stream等与响应内容无关的参数
    };
    
    // 创建哈希
    const hash = crypto.createHash('md5')
      .update(JSON.stringify(keyParams))
      .digest('hex');
      
    return hash;
  }
  
  /**
   * 获取缓存的响应
   * @param {string} key 缓存键
   * @returns {Object|null} 缓存的响应或null
   */
  get(key) {
    const cachePath = path.join(this.options.cacheDir, `${key}.json`);
    
    try {
      if (fs.existsSync(cachePath)) {
        const data = JSON.parse(fs.readFileSync(cachePath, 'utf8'));
        
        // 检查缓存是否过期
        if (Date.now() - data.timestamp > this.options.maxAge) {
          // 删除过期缓存
          fs.unlinkSync(cachePath);
          this.stats.misses++;
          return null;
        }
        
        // 更新访问时间
        data.accessTime = Date.now();
        fs.writeFileSync(cachePath, JSON.stringify(data));
        
        this.stats.hits++;
        return data.response;
      }
    } catch (error) {
      console.warn('读取缓存失败:', error.message);
    }
    
    this.stats.misses++;
    return null;
  }
  
  /**
   * 缓存响应
   * @param {string} key 缓存键
   * @param {Object} response 响应对象
   */
  set(key, response) {
    const cachePath = path.join(this.options.cacheDir, `${key}.json`);
    
    try {
      const data = {
        timestamp: Date.now(),
        accessTime: Date.now(),
        response
      };
      
      fs.writeFileSync(cachePath, JSON.stringify(data));
      
      // 更新缓存大小
      this.stats.size++;
      
      // 如果超过最大大小，清理最旧的缓存
      this._cleanupIfNeeded();
    } catch (error) {
      console.warn('写入缓存失败:', error.message);
    }
  }
  
  /**
   * 清空缓存
   */
  clear() {
    try {
      const files = fs.readdirSync(this.options.cacheDir);
      
      for (const file of files) {
        if (file.endsWith('.json')) {
          fs.unlinkSync(path.join(this.options.cacheDir, file));
        }
      }
      
      this.stats.size = 0;
      
      // 重置缓存信息
      this._saveCacheInfo();
    } catch (error) {
      console.warn('清空缓存失败:', error.message);
    }
  }
  
  /**
   * 获取缓存统计信息
   * @returns {Object} 缓存统计
   */
  getStats() {
    return { ...this.stats };
  }
  
  /**
   * 如果需要，清理旧缓存
   * @private
   */
  _cleanupIfNeeded() {
    if (this.stats.size <= this.options.maxSize) {
      return;
    }
    
    try {
      // 获取所有缓存文件及其信息
      const files = fs.readdirSync(this.options.cacheDir)
        .filter(file => file.endsWith('.json'))
        .map(file => {
          const filePath = path.join(this.options.cacheDir, file);
          const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
          
          return {
            path: filePath,
            file,
            accessTime: data.accessTime || data.timestamp
          };
        })
        .sort((a, b) => a.accessTime - b.accessTime); // 按访问时间排序
      
      // 删除最旧的文件
      const filesToRemove = files.slice(0, files.length - this.options.maxSize);
      
      for (const file of filesToRemove) {
        fs.unlinkSync(file.path);
        this.stats.size--;
      }
    } catch (error) {
      console.warn('清理缓存失败:', error.message);
    }
  }
  
  /**
   * 加载缓存信息
   * @private
   */
  _loadCacheInfo() {
    const infoPath = path.join(this.options.cacheDir, 'cache-info.json');
    
    try {
      if (fs.existsSync(infoPath)) {
        const info = JSON.parse(fs.readFileSync(infoPath, 'utf8'));
        this.stats = { ...this.stats, ...info };
      } else {
        // 计算当前缓存大小
        const files = fs.readdirSync(this.options.cacheDir)
          .filter(file => file.endsWith('.json') && file !== 'cache-info.json');
          
        this.stats.size = files.length;
        this._saveCacheInfo();
      }
    } catch (error) {
      console.warn('加载缓存信息失败:', error.message);
    }
  }
  
  /**
   * 保存缓存信息
   * @private
   */
  _saveCacheInfo() {
    const infoPath = path.join(this.options.cacheDir, 'cache-info.json');
    
    try {
      fs.writeFileSync(infoPath, JSON.stringify(this.stats));
    } catch (error) {
      console.warn('保存缓存信息失败:', error.message);
    }
  }
}

module.exports = ResponseCache;
