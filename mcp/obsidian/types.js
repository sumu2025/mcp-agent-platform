/**
 * Obsidian集成类型定义
 */

/**
 * Obsidian笔记类型
 * @typedef {Object} ObsidianNote
 * @property {string} path 笔记路径
 * @property {string} name 笔记名称
 * @property {string} content 笔记内容
 * @property {Object} metadata 元数据
 * @property {Date} created 创建时间
 * @property {Date} modified 修改时间
 * @property {Array<string>} tags 标签
 * @property {Array<string>} links 链接
 * @property {Array<string>} backlinks 反向链接
 */

/**
 * Obsidian文件夹类型
 * @typedef {Object} ObsidianFolder
 * @property {string} path 文件夹路径
 * @property {string} name 文件夹名称
 * @property {Array<ObsidianNote|ObsidianFolder>} children 子项目
 */

/**
 * Obsidian变更类型
 * @typedef {Object} ObsidianChange
 * @property {string} path 变更路径
 * @property {string} type 变更类型 (create, modify, delete)
 * @property {Date} timestamp 变更时间
 */

/**
 * Obsidian搜索结果类型
 * @typedef {Object} ObsidianSearchResult
 * @property {string} path 笔记路径
 * @property {string} content 匹配内容
 * @property {number} score 相关度分数
 * @property {Array<{start: number, end: number}>} matches 匹配位置
 */

/**
 * Obsidian操作选项
 * @typedef {Object} ObsidianOptions
 * @property {boolean} createIfNotExists 不存在时创建
 * @property {boolean} overwrite 覆盖已有内容
 * @property {boolean} includeFrontMatter 包含YAML前置元数据
 * @property {boolean} includeChildren 包含子项目
 * @property {number} depth 递归深度
 * @property {Function} filter 过滤函数
 */

module.exports = {};
