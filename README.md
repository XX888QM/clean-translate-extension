# YX Clean Translator (YX 纯净网页翻译)

<div align="center">
  <img src="icons/icon128.png" alt="Logo" width="100"/>
  <h3>A clean, fast, and crash-free webpage translator for Chrome.</h3>
  <p>专为极致阅读体验设计的 Chrome 网页翻译插件。无侵入式文本替换，完美解决 React/Vue 页面翻译崩溃问题。</p>
</div>

---

## ✨ Features (功能亮点)

- 🚀 **Crash-Free Translation**: Uses non-invasive text node replacement to safely translate complex web apps (React, Vue, SPA) without breaking the DOM.
  - **零崩溃**：采用无侵入式文本节点替换技术，完美兼容 React/Vue 等复杂单页应用，告别页面报错。
- ⚡ **High Performance**: Optimized concurrency (batch size 18) and local caching for instant translation.
  - **极致性能**：优化的高并发请求（18线程）与本地缓存策略，实现秒级即时翻译。
- 🤖 **AI-Optimized**: Built-in glossary for accurate translation of AI technical terms (e.g., Agent, Transformer, Token).
  - **AI 术语校对**：内置 AI 专业术语库，精确翻译 "Agent", "Transformer" 等专业词汇。
- 🔄 **Smart Caching**: Automatically caches translated text to verify instant loading on revisiting pages.
  - **智能缓存**：自动缓存已翻译内容，再次访问同一页面时实现 0 延迟加载。
- 🎨 **Visual Feedback**: Elegant toast notifications for translation status.
  - **优雅交互**：极简的 Toast 提示，实时反馈翻译进度与状态。

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
4. Click **Load unpacked**.
5. Select the directory where you cloned this repository.

## 📦 Project Structure

```
.
├── manifest.json       # Config: Permissions, version, icons
├── background.js       # Core: Handles API requests, concurrency, and glossaries
├── content.js          # Logic: DOM traversal, text replacement, and UI injection
├── popup.html          # UI: The extension popup interface
├── popup.js            # UI Logic: Settings and toggle interactions
├── icons/              # Assets: App icons
└── PRIVACY_POLICY.md   # Legal: Bilingual privacy policy
```

## 🔐 Privacy (隐私安全)

- **Pure Local Logic**: No user data is sent to private servers.
- **Minimal Permissions**: Only requests necessary permissions (`activeTab`, `storage`, `contextMenus`).
- **Transparency**: Fully open-source.
- [Read Privacy Policy](PRIVACY_POLICY.md)

## 📝 Changelog (更新日志)

### v1.2.2
- 添加 API 限流保护和自动重试机制
- 优化 AI 术语表查询性能（420+ 术语）
- 新增主流平台术语支持（GitHub, Twitter, Reddit, Discord 等）
- 修复并发翻译竞态条件
- 优化缓存策略，容量提升至 10000 条
- 完善错误处理和用户提示

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
