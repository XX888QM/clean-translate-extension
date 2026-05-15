# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

YX 纯净网页翻译（v1.5.1）— Chrome Manifest V3 扩展，把网页内容翻译为目标语言。**核心思想**：用 `TreeWalker` 遍历 DOM 文本节点逐节点替换 `nodeValue`，而非替换 `innerHTML`，避免破坏 React/Vue/SPA 框架的组件状态与事件绑定。

## 开发命令

纯原生 ES6+ JS，**无构建步骤、无依赖**。

```bash
# 测试（仅覆盖纯函数；异步链路、DOM 还原、IDB 操作未覆盖，改完核心逻辑必须实测）
node tests/run-tests.js

# 单个 describe block 没有筛选机制，要单独跑就临时改 tests/run-tests.js 注释掉别的 describe
```

```text
# 加载扩展
chrome://extensions/ → 启用"开发者模式" → "加载未打包的扩展程序" → 选择本目录

# 修改源码后必须在 chrome://extensions/ 点该扩展卡片的刷新图标，否则代码不更新

# 调试
- background.js (Service Worker): chrome://extensions/ → 该扩展 → "Service Worker"
- content.js: 目标网页 DevTools Console
- popup.js: 右键扩展图标 → "检查弹出窗口"
- IndexedDB: DevTools → Application → IndexedDB → yx-translate-cache → translations
```

## 架构（三层 message-passing）

```
popup.js (扩展页：UI 控制 + 引擎配置)
    ↓ chrome.tabs.sendMessage(tabId, ...)
content.js (注入到 <all_urls>：DOM 遍历 + 内存 LRU + UX 交互)
    ↓ chrome.runtime.sendMessage（经 sendMessageWithRetry 自动重试 SW 重启）
background.js (Service Worker：9 引擎路由 + IDB 持久缓存 + 术语校对 + 缓存淘汰)
```

### 消息协议

| 类型 | 方向 | 说明 |
|---|---|---|
| `START_TRANSLATE` | popup/background → content | 触发整页翻译（可带 `recordPreference: true` 自动记 'auto' 偏好） |
| `RESTORE_ORIGINAL` | popup/background → content | 还原原文 |
| `TRANSLATE_SELECTION` | background → content | 右键菜单划词翻译 |
| `TRANSLATE_TEXT_BATCH` | content → background | 批量翻译请求 |
| `TRANSLATE_COMPARE` | content → background | 划词翻译多引擎对比（当前引擎 + Google Free） |
| `TRANSLATION_DONE` | content → background | 更新图标徽章 ✓ |
| `CACHE_GET_ALL` / `CACHE_PUT_BATCH` / `CACHE_CLEAR` / `CACHE_COUNT` | popup/content → background | IDB 缓存读写 |
| `CACHE_TOUCH` | content → background | 命中缓存后批量更新 `lastAccess`（防活跃数据被 TTL 清掉） |
| `GET_ENGINE_CONFIG` / `SAVE_ENGINE_CONFIG` | popup → background | **仅扩展页可调**，content script 调用会返回 `forbidden` |

### 关键安全/可靠性机制

1. **消息 sender 校验**：`SAVE_ENGINE_CONFIG` / `GET_ENGINE_CONFIG` 入口检查 `sender.tab` 是否存在且 `sender.id === chrome.runtime.id`，content script 不能读写 API Key。新增涉及 API Key / 配置 的消息必须照此模式收紧。
2. **AbortController + 15s 超时**：`_handleBatchTranslation` 为每组创建 `AbortController`，15 秒触发 `controller.abort()`，所有 fetch 链路（`translateBulk` / `translateGoogleCloud` / `translateDeepL` / `translateBaidu` / `translateWithLLM` / `translateFallback` / `translateSingle`）都接收并透传 `signal`。**新增引擎或修改 fetch 链路必须传 signal**；catch 块要识别 `signal?.aborted || e?.name === 'AbortError'` 不进入回退逻辑（防止 abort 后又打 google_free 重新发起请求）。
3. **`normalizeTargetLang` / `normalizeEngine`**：`background.js` 顶部定义白名单。从 `chrome.storage.local` 读出来的 `target_lang` / `translate_engine` 必须经过 normalize 才能用，防止脏数据进 URL 或 LLM prompt。
4. **API 错误净化**：4xx/5xx 响应**不透传 body**，只保留 status，避免 key 片段 / 请求摘要泄漏。
5. **performTranslation `.catch()` 兜底**：`content.js` 里两处调用点（`triggerAutoTranslate` 和 `START_TRANSLATE` 处理）都必须有 catch：失败时 `hideProgressBar()` + `showToast('翻译失败','error')` + `isTranslating = false` + `autoTranslateTriggered = false`。

## background.js（Service Worker）

- **9 个翻译引擎**：`google_free` / `google_cloud` / `deepl` / `baidu` / `openai` / `claude` / `deepseek` / `minimax` / `glm`
- **路由**：`translateByEngine(texts, targetLang, engine, apiKeys, signal)` 按 `engine` 分发
- **LLM 批量协议**：编号列表格式 `[1] text\n[2] text`，回包用 `/^\[(\d+)\]\s*(.+)$/` 匹配。未匹配的条目保留原文
- **回退链**：任何非 `google_free` 引擎失败（非 abort）→ `translateBulk`（Google Free，行数不匹配/限流时 → `translateBulkFallback`（逐条 `translateSingle`）→ MyMemory `translateFallback`）→ 原文
- **DeepL 自动区分 Free/Pro**：密钥以 `:fx` 结尾走 `api-free.deepl.com`
- **百度签名**：自带纯 JS `md5()`（SW 不支持同步 `crypto.subtle`）。`md5` 函数内嵌于 `background.js` 第 ~600 行起，处理 UTF-8 字节
- **术语校对（`refineTranslation`）**：用 `AI_GLOSSARY`（内置 420+ AI 相关术语）+ 用户自定义术语（`storage.sync.user_glossary`）。**仅 `targetLang.startsWith('zh')` 生效**。Google Cloud / DeepL / 百度的批量结果走 `Promise.all` 并行校对（不要回到串行 await）。`buildCompiledGlossary()` 有进程内缓存，`chrome.storage.onChanged` 监听 `user_glossary` 变化时置空缓存
- **SW 心跳保活**：`startKeepAlive()` 用 `setInterval(25s)` 调 `chrome.runtime.getPlatformInfo()` 保活，**仅在 `activeTranslations > 0` 时启用**。注：未来若改用 `chrome.runtime.connect` Port 可以删除心跳
- **`onInstalled` 升级迁移**：`translation_cache` → IndexedDB，`auto_translate_enabled === false` → `translate_mode: 'manual'`，`excluded_domains` → `site_preferences` 标记 `'never'`

### IndexedDB 缓存（v2 schema）

| 项 | 值 |
|---|---|
| DB | `yx-translate-cache` (version 2) |
| Store | `translations` |
| key | 原文字符串 |
| value | `{ v: 译文, t: 最近访问时间戳 ms }`（v1 旧纯字符串通过 `unwrapCacheValue` 自动兼容） |

- `cachePutBatch(entries)`：写入时强制设 `t = Date.now()`
- `cacheTouchBatch(keys)`：批量更新 `t` 为 now，由 content.js 在命中缓存后异步调用
- `cleanupCache()`：阶段 1 删除 `t < now - 30天` 的条目；阶段 2 总字节超过 50 MB 时按 `t` 升序继续删
- **两个 alarm 触发清理**：
  - `yx-cache-cleanup`：每 24 小时（`periodInMinutes: 24*60`）
  - `yx-cache-cleanup-after-put`：写入后 1 分钟一次性 alarm（`delayInMinutes: 1`），通过 `chrome.alarms.get()` 防重排
- **不要用 `setTimeout` 做后台清理**：MV3 SW 可能秒级休眠，timer 会丢

## content.js（Content Script）

- **DOM 遍历**：`TreeWalker(SHOW_TEXT)` + `isTranslatable()` 过滤（跳过 `IGNORED_TAGS` / contenteditable / icon class / 纯数字 / 纯符号 / 全大写常量 / snake_case / JSON-like / 太短）
- **原文存储**：`originalTextMap` (WeakMap Node → string)，`originalAttrMap` (WeakMap Element → {attr: value})
- **双层缓存**：
  - 内存 LRU：`translationCache` (Map, 上限 `MAX_CACHE_SIZE = 10000`)
  - 持久 IDB：通过 background 的 `CACHE_*` 消息
  - **`cacheLoaded` 标志**：`ensureCacheLoaded()` 只在首次整页翻译时拉一次 `CACHE_GET_ALL`；MutationObserver / IntersectionObserver 触发的子树翻译复用内存缓存。语言切换 / `clearCache()` 时重置标志
- **`cacheTouchedKeys` + `flushCacheTouches()`**：命中缓存的 key 收集起来；在 _doTranslation 早退、正常结束、`performTranslation` 的 `finally` **三处都必须 flush**（异常退出兜底）
- **`translatedAttrRefs` 内存管理**：
  - WeakRef 数组存放已翻译属性元素
  - `recordTranslatedAttrElement()` 用 WeakSet 查重，避免同元素重复 push
  - 超过 `TRANSLATED_ATTR_REFS_COMPACT_THRESHOLD = 5000` 时调 `compactTranslatedAttrRefs()` 清失效引用
  - 还原原文 / 语言切换调 `resetTranslatedAttrTracker()` 同时重建 WeakSet（注意：函数体是 `translatedAttrRefs.length = 0; translatedAttrElementSet = new WeakSet();`，不要写成自调用）
- **MutationObserver**：监听 `document.body subtree:true`，200ms 防抖，`pendingNodes` 集合上限 `MAX_PENDING_NODES = 100`
- **IntersectionObserver**：rootMargin 200px 视口预加载翻译
- **悬停显示原文**：事件委托 + 1 秒延迟
- **划词翻译多引擎对比**：发 `TRANSLATE_COMPARE` 消息，气泡同时显示当前引擎和 Google Free 结果
- **三种翻译模式 + 网站偏好**：模式 `auto_all` / `whitelist` / `manual`；偏好 `site_preferences[domain]: 'auto' | 'never'`，**优先级高于模式**

## popup.js（扩展页 UI）

- **翻译/还原**：手动翻译用 `recordPreference: true` 让 content 自动记录 'auto' 偏好
- **自定义术语删除**：用 `glossarySignature(item)` = `${keyword}${badWord}${goodWord}` 作为稳定签名，`deleteGlossaryItemBySignature(sig)` 按签名 filter 删除，**不要回到 `splice(index, 1)`**（连点删错条目）
- **API Key 管理**：根据引擎动态显示输入面板。新增引擎需同时改 `ENGINE_HINTS` 和 `ENGINE_KEY_MAP`

## Chrome Storage 键

| 键 | 区域 | 类型 | 说明 |
|---|---|---|---|
| `translate_mode` | local | `'auto_all' \| 'whitelist' \| 'manual'` | 翻译模式（默认 `auto_all`） |
| `whitelist_domains` | local | string[] | whitelist 模式使用 |
| `site_preferences` | local | `{ [domain]: 'auto' \| 'never' }` | 优先级高于 mode |
| `translate_engine` | local | string | 经 `normalizeEngine` 校验 |
| `target_lang` | local | string | 经 `normalizeTargetLang` 校验 |
| `api_keys` | local | object | **存 local 不存 sync**，避免上传 Google 账号 |
| `bilingual_mode` | local | boolean | 双语对照 |
| `user_glossary` | **sync** | `{keyword, badWord, goodWord}[]` | **会同步到 Google 账号**，隐私政策已声明 |
| `auto_translate_enabled` / `excluded_domains` | local | — | **旧版**，仅 `onInstalled` 迁移用 |

## 关键配置常量

| 常量 | 位置 | 值 | 说明 |
|---|---|---|---|
| `MAX_BULK_CHARS` | background.js | 1500 | 单次合并翻译最大字符数 |
| `PARALLEL` | background.js | 8 | 翻译最大并发数 |
| `TRANSLATE_TIMEOUT_MS` | background.js | 15000 | 每组 fetch 超时（AbortController） |
| `CACHE_TTL_MS` | background.js | 30 天 | IDB 缓存过期阈值 |
| `CACHE_MAX_BYTES` | background.js | 50 MB | IDB 缓存硬上限 |
| `ALLOWED_TARGET_LANGS` / `ALLOWED_ENGINES` | background.js | Set | 白名单 |
| `CHUNK_SIZE` | content.js | 200 | content → background 单批文本数 |
| `MAX_CACHE_SIZE` | content.js | 10000 | 内存 LRU 上限 |
| `MAX_PENDING_NODES` | content.js | 100 | MutationObserver 待处理上限 |
| `MIN_TEXT_LENGTH` | content.js | 2 | 可翻译最小字符数 |
| `TRANSLATED_ATTR_REFS_COMPACT_THRESHOLD` | content.js | 5000 | WeakRef 数组压缩阈值 |

## 添加新翻译引擎的清单

新增一个引擎需要同步改 **5 处**：
1. `background.js`: `ALLOWED_ENGINES` 加引擎 id
2. `background.js`: `translateByEngine` switch 加 case（或加入 LLM 分支） + 实现 `translate<Name>(texts, targetLang, apiKey, signal)`（必须接收并透传 signal）
3. `background.js`: `TRANSLATE_COMPARE` 处理里的 `ENGINE_NAMES` 加显示名
4. `popup.html`: `<select id="engineSelect">` 加 `<option>`
5. `popup.js`: `ENGINE_HINTS`（API Key 申请地址提示） + `ENGINE_KEY_MAP`（密钥字段名）

如果是 LLM 类引擎，只需在 `background.js` 的 `translateWithLLM` 里 `engineConfig` 加一项（url / model / headers / buildBody），其他都按 LLM 路径走。

## 代码规范

- 所有注释中文，UI 文案中文
- 原生 ES6+，无构建工具
- DOM 写入：用 `textContent` / `createElement`，不要 `innerHTML`（XSS 防护，本项目 `<all_urls>` 注入）
- 扩展上下文检查：所有 chrome API 调用前用 `chrome.runtime?.id` 守卫，处理 SW 重启 / 扩展被卸载
- 消息发送：content → background 用 `sendMessageWithRetry()`，默认重试 2 次
- 新加涉及 API Key / 配置 的消息处理器：必须校验 `sender.tab` 不存在 + `sender.id === chrome.runtime.id`
- 新加 fetch 调用：必须接 `signal` 参数，catch 块识别 abort 不进入回退
- 新加 chrome.storage.local 读出来的字段：考虑是否需要白名单/类型校验
- 新加后台延迟任务：用 `chrome.alarms`，不要用 `setTimeout`
