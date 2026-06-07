# YX Clean Translator (YX 纯净网页翻译)

<div align="center">
  <img src="icons/icon128.png" alt="Logo" width="100"/>
  <h3>A clean, fast, and crash-free webpage translator for Chrome.</h3>
  <p>专为极致阅读体验设计的 Chrome 网页翻译插件。无侵入式文本替换，完美解决 React/Vue 页面翻译崩溃问题。</p>
</div>

---

## ✨ Features (功能亮点)

- 🚀 **Crash-Free Translation / 零崩溃翻译**
  Uses non-invasive text node replacement to safely translate complex web apps (React, Vue, SPA) without breaking the DOM.
  采用 `TreeWalker` 逐节点替换 `nodeValue`，不触碰 `innerHTML`，完美兼容 React / Vue / 任意 SPA 框架。

- 🌐 **9 Translation Engines / 9 大翻译引擎**
  Google Free（默认，免配置）、Google Cloud、DeepL（自动识别 Free / Pro）、百度、OpenAI、Anthropic Claude、DeepSeek、MiniMax、智谱 GLM。失败自动回退到 Google Free。

- ⚡ **Smart Caching / 智能缓存**
  内存 LRU + IndexedDB 双层缓存；30 天未访问自动过期，总量超过 50 MB 自动按访问时间淘汰，永远不会撑爆磁盘。

- 🎯 **Per-Site Preferences / 网站偏好**
  支持「自动翻译全部 / 仅白名单 / 全部手动」三种模式，单个域名可单独设为「自动」或「从不」。

- 🤖 **AI Glossary / AI 术语校对**
  内置 420+ AI/ML 专有术语词表，翻译为中文时自动校正常见错译（如 "Agent" → 智能体、"Token" → Token），支持用户自定义术语。

- 🪟 **Selection Comparison / 划词多引擎对比**
  右键选中文本可同时获得当前引擎和 Google Free 的翻译，对比择优。

- 🔒 **Privacy-First / 隐私优先**
  无自建服务器；API 密钥仅存 `chrome.storage.local`，不上云；`<input>` / `<textarea>` / contenteditable 元素绝不进入翻译请求。

## 🛠 Installation (安装指南)

### From Chrome Web Store (Coming Soon)
Visit the Chrome Web Store link (link to be added) and click "Add to Chrome".

### Manual Installation (开发版安装)
1. Clone this repository:
   ```bash
   git clone https://github.com/XX888QM/clean-translate-extension.git
   ```
2. Open Chrome and navigate to `chrome://extensions/`.
3. Enable **Developer mode** (top right corner).
4. Click **Load unpacked** and select the directory.
5. （可选）在扩展弹窗里选择翻译引擎并填入对应的 API Key。

## ⌨️ Shortcuts (快捷键)

| 操作 | 默认 |
|---|---|
| 翻译当前页面 | `Alt + T` |
| 还原原文 | `Alt + R` |

## 📦 Project Structure

```
.
├── manifest.json       # Manifest V3 配置（permissions / commands / icons）
├── background.js       # Service Worker：9 引擎路由、IDB 缓存、缓存淘汰、术语校对
├── content.js          # Content Script：DOM 遍历、内存缓存、UI 交互
├── popup.html / .js    # 弹窗 UI：引擎切换、API Key、翻译模式、网站偏好
├── icons/              # 应用图标
├── tests/run-tests.js  # 纯函数单测（Node.js 跑）
├── CLAUDE.md           # Claude Code 用项目说明
├── AGENTS.md           # Codex CLI 用项目说明（与 CLAUDE.md 同步）
└── PRIVACY_POLICY.md   # 双语隐私政策
```

## 🔐 Permissions (权限说明)

| 权限 | 用途 |
|---|---|
| `activeTab` | 当前 Tab 内容访问 |
| `storage` | 保存设置、API Key、翻译缓存 |
| `contextMenus` | "翻译选中文本" / "翻译整个页面" 右键菜单 |
| `alarms` | 每日定时清理 IndexedDB 缓存中过期条目 |
| `<all_urls>` content script | 翻译脚本注入到任意网页（不主动翻译时不读取数据） |

详见 [Privacy Policy](PRIVACY_POLICY.md)。

## 🧪 Development

```bash
# 跑测试（仅覆盖纯函数）
node tests/run-tests.js

# 加载扩展：chrome://extensions/ → 开发者模式 → 加载未打包 → 选择本目录
# 修改源码后：在该扩展卡片上点击刷新按钮
```

更多开发细节（架构、消息协议、安全机制）见 [CLAUDE.md](CLAUDE.md)。

## 📝 Changelog

### v1.5.2 (current)
- 隐私：敏感域名（邮箱/网银/本机/私网/内网/`file://`）默认不自动翻译，堵住 whitelist 模式与历史 `auto` 偏好等绕过路径；敏感站手动翻译不再静默记为长期自动外发
- 可靠性：翻译代际令牌，「还原 / 切换语言」后丢弃在途结果，杜绝译文回冒；视口子树翻译用独立 touch 集合避免并发互相清空；熔断改为批级（间歇失败不再永远攒不到阈值）；IndexedDB 不可用时优雅降级
- 性能：术语校对 `refineTranslation` 预编译短路正则，整页中文校对实测快 16~42x（行为零变化）
- 重构：目标语言表（`LANGS`）与翻译引擎表（`ENGINE_REGISTRY`）改为单一真相源派生，消除 4 张语言表 / 6 处引擎元数据分散
- 体验：付费引擎未配置 API Key 时弹提示（不再静默回退免费 Google）；LLM 提示词补全 ar/th/vi/pt-BR 等语言名；`translateBulk` 组内去重省重复请求
- 测试：改为 `require` 真源码（不再测复制粘贴副本，曾因此漏过崩溃 bug），用例 29 → 83
- 工程：统一行尾为 LF + `.gitattributes`；删除操作废弃 storage 键的死代码消息

### v1.5.1
- fetch 全链路 AbortController + 15s 超时取消，杜绝请求悬空
- IDB 缓存升级 schema v2：`lastAccess` 字段 + 30 天过期 + 50 MB 上限
- 缓存清理改用 `chrome.alarms`（daily + 写后一次性 alarm），SW 休眠也可靠
- `SAVE/GET_ENGINE_CONFIG` 严格限定扩展页调用，content script 无法读写 API Key
- `translatedAttrRefs` 加 WeakSet 去重 + 失效引用清理，SPA 长跑无内存泄漏
- 子树翻译复用内存缓存，不再每次 dump 全量 IDB
- Google Cloud / DeepL / 百度的术语校对改并行 (`Promise.all`)
- `target_lang` / `engine` 强制白名单校验，API 错误响应不再透传细节
- 自定义术语按内容签名删除，连点不再删错
- 隐私政策补齐 9 引擎说明 + storage.sync 透明声明
- manifest 补 `author` / `homepage_url` / `minimum_chrome_version`

### v1.5.0
- 翻译核心与 UI 大幅重构，新增 9 引擎（含 OpenAI / Claude / DeepL / 百度等）
- 引入 IndexedDB 持久缓存与内存 LRU 双层缓存
- 翻译模式（auto_all / whitelist / manual）与网站偏好

### v1.2.2
- 限流保护与自动重试
- AI 术语表扩充至 420+
- 修复并发翻译竞态

## 🤝 Contributing

Contributions welcome. PRs should keep `node tests/run-tests.js` green and follow the conventions in `CLAUDE.md`.

## 📄 License

MIT — see [LICENSE](LICENSE).
