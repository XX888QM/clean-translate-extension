# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概述

YX 纯净网页翻译（v1.5.0）— Chrome 浏览器扩展，采用无侵入式文本节点替换将网页翻译为目标语言。专为 React/Vue 等 SPA 框架设计，使用 TreeWalker 遍历 DOM 文本节点而非替换 innerHTML，翻译后不破坏组件状态。

## 开发与调试

纯原生 JavaScript 项目，**无构建步骤、无依赖安装**。

```
# 加载扩展
1. chrome://extensions/ → 启用"开发者模式"
2. "加载未打包的扩展程序" → 选择本目录

# 修改后刷新
chrome://extensions/ 页面点击扩展卡片上的刷新图标

# 调试日志
- background.js: chrome://extensions/ → 点击"Service Worker"
- content.js: 目标网页的 DevTools Console
- popup.js: 右键扩展图标 → "检查弹出窗口"
- IndexedDB 缓存: DevTools → Application → IndexedDB → yx-translate-cache

# 运行测试
node tests/run-tests.js
```

## 架构（三层通信模型）

```
popup.js (UI控制 + 引擎配置)
    ↓ chrome.tabs.sendMessage
content.js (DOM操作 + 内存LRU缓存 + UX交互)
    ↓ chrome.runtime.sendMessage (带重试)
background.js (多引擎翻译路由 + API调用 + 术语校对 + IndexedDB持久缓存 + SW生命周期保护)
```

### 消息协议

| 消息类型 | 方向 | 说明 |
|---------|------|------|
| `START_TRANSLATE` | popup → content | 触发页面翻译（可带 `recordPreference: true` 记录网站偏好） |
| `RESTORE_ORIGINAL` | popup → content | 还原原文 |
| `TRANSLATE_TEXT_BATCH` | content → background | 批量翻译请求 |
| `TRANSLATE_COMPARE` | content → background | 划词翻译引擎对比（返回主引擎 + Google免费结果） |
| `TRANSLATE_SELECTION` | background → content | 右键菜单划词翻译 |
| `TRANSLATION_DONE` | content → background | 翻译完成（更新图标徽章） |
| `CACHE_GET_ALL` | popup/content → background | 获取全量 IndexedDB 缓存 |
| `CACHE_PUT_BATCH` | popup/content → background | 批量写入 IndexedDB 缓存 |
| `CACHE_CLEAR` | popup/content → background | 清空 IndexedDB 缓存 |
| `CACHE_COUNT` | popup → background | 获取缓存条目数 |
| `GET_ENGINE_CONFIG` | popup → background | 获取当前引擎和API密钥 |
| `SAVE_ENGINE_CONFIG` | popup → background | 保存引擎配置 |

### 核心文件职责

#### background.js (Service Worker)
- **多引擎路由**：`translateByEngine()` 根据用户选择的引擎分发到对应翻译函数
- 支持 9 个引擎：Google免费、Google Cloud、DeepL、百度、OpenAI、Codex、DeepSeek、MiniMax、智谱GLM
- 批量翻译：按字符量分组（`MAX_BULK_CHARS=1500`），最多 8 并发，15秒超时
- LLM 翻译使用编号列表格式（`[1] text`）实现批量翻译，解析时用正则 `/^\[(\d+)\]\s*(.+)$/` 匹配
- 所有非免费引擎失败时自动回退到免费 Google 翻译
- AI 术语校对（`AI_GLOSSARY`）：仅对中文目标语言生效
- 百度翻译使用纯 JS 实现的 `md5()` 函数签名（Service Worker 不支持同步 crypto）
- DeepL 自动区分 Free/Pro（密钥以 `:fx` 结尾为 Free）
- **IndexedDB 持久缓存**：`openCacheDB()` / `cacheGetAll()` / `cachePutBatch()` / `cacheClearAll()` / `cacheCount()`
- **SW 生命周期保护**：`startKeepAlive()` / `stopKeepAlive()`，翻译进行中每 25 秒心跳防止 SW 被终止
- **引擎对比翻译**：`TRANSLATE_COMPARE` 消息，并行调用当前引擎 + Google免费
- **升级迁移**：`onInstalled` 中将旧版 `translation_cache` 迁移到 IndexedDB，`excluded_domains` 迁移到 `site_preferences`

#### content.js (Content Script)
- DOM 遍历：`TreeWalker` + `SHOW_TEXT` 过滤，`WeakMap` 存储原文
- **翻译进度条**：页面顶部 3px 蓝色渐变条
- **视口按需翻译**：`IntersectionObserver` + 200px rootMargin 预加载
- **鼠标悬停显示原文**：事件委托，1秒延迟
- **双层缓存**：内存 `Map`（LRU）+ background IndexedDB（通过消息通信）
- **消息重试**：`sendMessageWithRetry()`，SW 被终止后自动重试（最多 2 次）
- `MutationObserver` 动态翻译 + 200ms 防抖
- **划词翻译引擎对比**：气泡显示多引擎结果
- **翻译模式**：支持 auto_all / whitelist / manual 三种模式
- **网站偏好**：per-domain 记忆（auto / never），手动翻译时自动记录

#### popup.js (UI 控制层)
- 翻译/还原按钮（手动翻译时通过 `recordPreference: true` 记录偏好）
- **翻译模式选择器**：自动翻译所有外语 / 仅翻译白名单 / 全部手动
- **网站偏好管理**：显示当前网站偏好状态，支持设为自动/从不/清除
- **缓存管理**：通过 background 消息操作 IndexedDB
- **引擎选择**：下拉菜单切换引擎
- **API Key 管理**：根据引擎动态显示/隐藏输入面板

#### popup.html
- Popup 界面 HTML + CSS（含完整暗色模式 `@media (prefers-color-scheme: dark)`）

#### tests/run-tests.js
- 轻量测试运行器（Node.js），测试核心纯函数
- `node tests/run-tests.js` 运行

### Chrome Storage 键

| 键 | 类型 | 说明 |
|---|---|---|
| `translate_mode` | `'auto_all' \| 'whitelist' \| 'manual'` | 翻译模式（默认 `'auto_all'`） |
| `whitelist_domains` | string[] | 白名单域名列表（翻译模式为 whitelist 时使用） |
| `site_preferences` | `{ [domain]: 'auto' \| 'never' }` | 网站翻译偏好（优先级高于翻译模式） |
| `auto_translate_enabled` | boolean | **旧版**，已被 `translate_mode` 取代，仅用于兼容 |
| `excluded_domains` | string[] | **旧版**，已被 `site_preferences` 取代，仅用于兼容 |
| `translate_engine` | string | 当前翻译引擎标识（默认 `'google_free'`） |
| `api_keys` | object | 各引擎 API 密钥 |
| `target_lang` | string | 目标语言代码（默认 `'zh-CN'`） |
| `bilingual_mode` | boolean | 双语对照模式 |

### IndexedDB 缓存

| 数据库 | Object Store | 说明 |
|--------|-------------|------|
| `yx-translate-cache` | `translations` | 翻译缓存，key=原文, value=译文 |

### 关键配置常量

| 常量 | 位置 | 值 | 说明 |
|------|------|------|------|
| `MAX_BULK_CHARS` | background.js | 1500 | 单次合并翻译最大字符数 |
| `PARALLEL` | background.js | 8 | 翻译最大并发数 |
| 超时 | background.js | 15000ms | API 请求超时（含 LLM） |
| `CHUNK_SIZE` | content.js | 200 | content → background 分块大小 |
| `MAX_CACHE_SIZE` | content.js | 10000 | LRU 缓存最大条目数 |
| `MIN_TEXT_LENGTH` | content.js | 2 | 最小可翻译文本长度 |

## 代码规范

- 所有注释使用中文
- 原生 ES6+ JavaScript，无构建工具
- Chrome Extension Manifest v3
- 使用 `chrome.runtime?.id` 检测扩展上下文有效性
- 使用 `sendMessageWithRetry()` 发送消息到 background（自动处理 SW 重启）
- XSS 防护：使用 DOM API（textContent、createElement）避免 innerHTML 注入
- 新增翻译引擎时需同步修改三处：`background.js`（引擎路由 + 配置 + ENGINE_NAMES）、`popup.html`（下拉选项）、`popup.js`（ENGINE_HINTS + ENGINE_KEY_MAP）
