/**
 * ObsidianMetadata
 * 
 * 处理Obsidian笔记的元数据、属性和特殊结构
 */

class ObsidianMetadata {
  /**
   * 解析笔记内容，提取元数据和结构
   * @param {string} content 笔记内容
   * @returns {Object} 解析结果
   */
  static parse(content) {
    const result = {
      frontmatter: {},      // YAML前置元数据
      properties: {},       // 内联属性
      tags: [],             // 标签
      links: [],            // 内部链接
      embeds: [],           // 嵌入内容
      callouts: [],         // 提示块
      codeBlocks: [],       // 代码块
      headings: [],         // 标题结构
      dataviews: [],        // Dataview查询
      cleanContent: content // 无元数据的纯内容
    };
    
    // 解析YAML前置元数据
    const { frontmatter, contentWithoutFrontmatter } = this._parseFrontmatter(content);
    result.frontmatter = frontmatter;
    result.cleanContent = contentWithoutFrontmatter;
    
    // 解析内联属性
    const { properties, contentWithoutProperties } = this._parseProperties(result.cleanContent);
    result.properties = properties;
    result.cleanContent = contentWithoutProperties;
    
    // 解析标签
    result.tags = this._parseTags(result.cleanContent);
    
    // 解析内部链接
    result.links = this._parseLinks(result.cleanContent);
    
    // 解析嵌入内容
    result.embeds = this._parseEmbeds(result.cleanContent);
    
    // 解析提示块
    result.callouts = this._parseCallouts(result.cleanContent);
    
    // 解析代码块
    result.codeBlocks = this._parseCodeBlocks(result.cleanContent);
    
    // 解析标题结构
    result.headings = this._parseHeadings(result.cleanContent);
    
    // 解析Dataview查询
    result.dataviews = this._parseDataviews(result.cleanContent);
    
    return result;
  }
  
  /**
   * 序列化元数据到Markdown内容
   * @param {Object} metadata 元数据对象
   * @param {string} content 原始内容
   * @returns {string} 包含元数据的内容
   */
  static serialize(metadata, content = '') {
    let result = content;
    
    // 添加YAML前置元数据
    if (metadata.frontmatter && Object.keys(metadata.frontmatter).length > 0) {
      // 移除现有的前置元数据
      result = this._removeFrontmatter(result);
      
      // 添加新的前置元数据
      let yamlContent = '---\n';
      for (const [key, value] of Object.entries(metadata.frontmatter)) {
        yamlContent += `${key}: ${this._formatYamlValue(value)}\n`;
      }
      yamlContent += '---\n\n';
      
      result = yamlContent + result;
    }
    
    // 添加内联属性
    if (metadata.properties && Object.keys(metadata.properties).length > 0) {
      // 移除现有的内联属性
      result = this._removeProperties(result);
      
      // 添加新的内联属性
      let propertiesContent = '';
      for (const [key, value] of Object.entries(metadata.properties)) {
        propertiesContent += `${key}:: ${value}\n`;
      }
      
      result = propertiesContent + '\n' + result;
    }
    
    return result;
  }
  
  /**
   * 解析YAML前置元数据
   * @param {string} content 笔记内容
   * @returns {Object} 解析结果
   * @private
   */
  static _parseFrontmatter(content) {
    const frontmatter = {};
    let contentWithoutFrontmatter = content;
    
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
          frontmatter[key.trim()] = value.trim();
        }
      });
      
      // 从内容中移除YAML部分
      contentWithoutFrontmatter = content.replace(yamlMatch[0], '');
    }
    
    return { frontmatter, contentWithoutFrontmatter };
  }
  
  /**
   * 解析内联属性
   * @param {string} content 笔记内容
   * @returns {Object} 解析结果
   * @private
   */
  static _parseProperties(content) {
    const properties = {};
    let contentWithoutProperties = content;
    
    // 匹配属性行（如 key:: value）
    const propertyRegex = /^([^:\n]+)::\s*(.+)$/gm;
    let match;
    const propertyLines = [];
    
    while ((match = propertyRegex.exec(content)) !== null) {
      const [fullMatch, key, value] = match;
      properties[key.trim()] = value.trim();
      propertyLines.push(fullMatch);
    }
    
    // 移除属性行
    propertyLines.forEach(line => {
      contentWithoutProperties = contentWithoutProperties.replace(line + '\n', '');
    });
    
    return { properties, contentWithoutProperties };
  }
  
  /**
   * 解析标签
   * @param {string} content 笔记内容
   * @returns {Array<string>} 标签数组
   * @private
   */
  static _parseTags(content) {
    const tags = [];
    
    // 匹配标准标签（如 #tag）
    const standardTagRegex = /(?:^|\s)#([^\s#]+)/g;
    let match;
    
    while ((match = standardTagRegex.exec(content)) !== null) {
      tags.push(match[1]);
    }
    
    // 匹配前置元数据中的标签
    const yamlRegex = /^---\n([\s\S]*?)\n---\n/;
    const yamlMatch = content.match(yamlRegex);
    
    if (yamlMatch) {
      const yamlContent = yamlMatch[1];
      
      // 查找tags字段
      const tagsMatch = yamlContent.match(/^tags:\s*\[(.*)\]$/m);
      if (tagsMatch && tagsMatch[1]) {
        const yamlTags = tagsMatch[1].split(/,\s*/).map(tag => tag.trim().replace(/"/g, ''));
        tags.push(...yamlTags);
      }
    }
    
    // 移除重复项
    return [...new Set(tags)];
  }
  
  /**
   * 解析内部链接
   * @param {string} content 笔记内容
   * @returns {Array<Object>} 链接数组
   * @private
   */
  static _parseLinks(content) {
    const links = [];
    
    // 匹配内部链接（如 [[链接]]）
    const linkRegex = /\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g;
    let match;
    
    while ((match = linkRegex.exec(content)) !== null) {
      links.push({
        target: match[1].trim(),
        alias: match[2] ? match[2].trim() : undefined,
        original: match[0]
      });
    }
    
    return links;
  }
  
  /**
   * 解析嵌入内容
   * @param {string} content 笔记内容
   * @returns {Array<Object>} 嵌入内容数组
   * @private
   */
  static _parseEmbeds(content) {
    const embeds = [];
    
    // 匹配嵌入内容（如 ![[嵌入]]）
    const embedRegex = /!\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g;
    let match;
    
    while ((match = embedRegex.exec(content)) !== null) {
      embeds.push({
        target: match[1].trim(),
        params: match[2] ? match[2].trim() : undefined,
        original: match[0]
      });
    }
    
    return embeds;
  }
  
  /**
   * 解析提示块
   * @param {string} content 笔记内容
   * @returns {Array<Object>} 提示块数组
   * @private
   */
  static _parseCallouts(content) {
    const callouts = [];
    
    // 匹配提示块（如 > [!NOTE] Title）
    const calloutRegex = /> \[!(\w+)\](?: (.+?))?\n((?:>.*\n)*)/g;
    let match;
    
    while ((match = calloutRegex.exec(content)) !== null) {
      // 提取内容行
      const contentLines = match[3].split('\n').map(line => {
        return line.startsWith('> ') ? line.slice(2) : line;
      }).join('\n');
      
      callouts.push({
        type: match[1],
        title: match[2] ? match[2].trim() : undefined,
        content: contentLines.trim(),
        original: match[0]
      });
    }
    
    return callouts;
  }
  
  /**
   * 解析代码块
   * @param {string} content 笔记内容
   * @returns {Array<Object>} 代码块数组
   * @private
   */
  static _parseCodeBlocks(content) {
    const codeBlocks = [];
    
    // 匹配代码块（如 ```language code ```）
    const codeBlockRegex = /```([a-zA-Z0-9]*)\n([\s\S]*?)```/g;
    let match;
    
    while ((match = codeBlockRegex.exec(content)) !== null) {
      codeBlocks.push({
        language: match[1].trim(),
        code: match[2],
        original: match[0]
      });
    }
    
    return codeBlocks;
  }
  
  /**
   * 解析标题结构
   * @param {string} content 笔记内容
   * @returns {Array<Object>} 标题结构数组
   * @private
   */
  static _parseHeadings(content) {
    const headings = [];
    
    // 匹配标题（如 ## 标题）
    const headingRegex = /^(#{1,6})\s+(.+)$/gm;
    let match;
    
    while ((match = headingRegex.exec(content)) !== null) {
      headings.push({
        level: match[1].length,
        text: match[2].trim(),
        original: match[0]
      });
    }
    
    return headings;
  }
  
  /**
   * 解析Dataview查询
   * @param {string} content 笔记内容
   * @returns {Array<Object>} Dataview查询数组
   * @private
   */
  static _parseDataviews(content) {
    const dataviews = [];
    
    // 匹配Dataview查询块（如 ```dataview query ```）
    const dataviewRegex = /```dataview\n([\s\S]*?)```/g;
    let match;
    
    while ((match = dataviewRegex.exec(content)) !== null) {
      dataviews.push({
        query: match[1].trim(),
        original: match[0]
      });
    }
    
    // 匹配内联Dataview查询（如 `= query`）
    const inlineDataviewRegex = /`=\s*(.+?)\s*`/g;
    
    while ((match = inlineDataviewRegex.exec(content)) !== null) {
      dataviews.push({
        query: match[1].trim(),
        inline: true,
        original: match[0]
      });
    }
    
    return dataviews;
  }
  
  /**
   * 移除YAML前置元数据
   * @param {string} content 笔记内容
   * @returns {string} 移除后的内容
   * @private
   */
  static _removeFrontmatter(content) {
    const yamlRegex = /^---\n[\s\S]*?\n---\n/;
    return content.replace(yamlRegex, '');
  }
  
  /**
   * 移除内联属性
   * @param {string} content 笔记内容
   * @returns {string} 移除后的内容
   * @private
   */
  static _removeProperties(content) {
    const propertyRegex = /^([^:\n]+)::\s*(.+)\n/gm;
    return content.replace(propertyRegex, '');
  }
  
  /**
   * 格式化YAML值
   * @param {*} value 值
   * @returns {string} 格式化后的值
   * @private
   */
  static _formatYamlValue(value) {
    if (typeof value === 'string') {
      // 如果值包含特殊字符，使用引号
      if (/[:\[\]{}]/.test(value)) {
        return `"${value}"`;
      }
      return value;
    } else if (Array.isArray(value)) {
      return `[${value.map(v => this._formatYamlValue(v)).join(', ')}]`;
    } else if (typeof value === 'object' && value !== null) {
      // 不支持嵌套对象，返回字符串表示
      return JSON.stringify(value);
    }
    return String(value);
  }
}

module.exports = ObsidianMetadata;
