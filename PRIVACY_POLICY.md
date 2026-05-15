# Privacy Policy for YX Clean Translator / YX 纯净网页翻译隐私政策

**Last updated / 最后更新日期:** May 16, 2026

---

## English Version

YX Clean Translator ("we", "our", or "us") is committed to protecting your privacy. This Privacy Policy explains how our Chrome extension handles your information.

### 1. Data Collection and Usage

**We do not operate any server, and we do not collect, store, or transmit any of your personal data to our own infrastructure.**

*   **No Personal Information:** We do not collect your name, email address, or any other personally identifiable information.
*   **No Browsing History:** We do not track the websites you visit or your browsing history.
*   **Local Processing:** Most extension settings (translate mode, engine choice, API keys, site preferences, translation cache) are stored locally on your device using Chrome's `storage.local` API and IndexedDB. This data never leaves your browser, except for the third-party translation requests described in Section 2.

### 2. Third-Party Translation Services

To provide translation functionality, **text content from the webpages you visit** is sent to the translation engine you select. The extension supports the following engines; only the engine you actively choose will receive your data:

| Engine | Endpoint | When data is sent |
|---|---|---|
| **Google Translate (Free)** *(default)* | `translate.googleapis.com` | Always available; used as fallback |
| **Google Cloud Translation** | `translation.googleapis.com` | Only if you provide a Google Cloud API key and select this engine |
| **DeepL** | `api-free.deepl.com` / `api.deepl.com` | Only if you provide a DeepL API key and select this engine |
| **Baidu Translate** | `fanyi-api.baidu.com` | Only if you provide a Baidu AppID + key and select this engine |
| **OpenAI** | `api.openai.com` | Only if you provide an OpenAI API key and select this engine |
| **Anthropic Claude** | `api.anthropic.com` | Only if you provide a Claude API key and select this engine |
| **DeepSeek** | `api.deepseek.com` | Only if you provide a DeepSeek API key and select this engine |
| **MiniMax** | `api.minimax.chat` | Only if you provide a MiniMax API key and select this engine |
| **Zhipu GLM** | `open.bigmodel.cn` | Only if you provide a GLM API key and select this engine |
| **MyMemory** *(fallback)* | `api.mymemory.translated.net` | Used only as a fallback when Google Free is unavailable |

What is sent to the selected engine:

*   The **visible text content** of pages you choose to translate (or pages auto-translated according to your settings).
*   Your **API key** for the selected engine (sent to that engine only, transmitted over HTTPS).
*   No cookies or personal identifiers.

Inputs (`<input>`, `<textarea>`) and contenteditable elements are excluded from translation requests. However, you should still be cautious on pages that display sensitive information as plain text (e.g., personal data, financial details). We recommend disabling auto-translate on those sites via the site preferences feature.

Each third-party service has its own privacy policy that governs how they process the data you send through this extension. Please consult their respective policies:

*   [Google Privacy Policy](https://policies.google.com/privacy)
*   [DeepL Privacy Policy](https://www.deepl.com/privacy)
*   [Baidu Translate Privacy Policy](https://fanyi.baidu.com/static/translation/widget/help.html)
*   [OpenAI Privacy Policy](https://openai.com/policies/privacy-policy)
*   [Anthropic Privacy Policy](https://www.anthropic.com/legal/privacy)
*   [DeepSeek Privacy Policy](https://www.deepseek.com/privacy)
*   [MiniMax Privacy Policy](https://www.minimaxi.com/privacy)
*   [Zhipu AI Privacy Policy](https://open.bigmodel.cn/usercenter/agreement)
*   [MyMemory Privacy Policy](https://mymemory.translated.net/doc/usagelimits.php)

### 3. Local Data Storage

*   **`chrome.storage.local`** stores: translation mode, target language, selected engine, API keys, site preferences, and migrated legacy settings. This data stays on your device.
*   **IndexedDB (`yx-translate-cache`)** stores translation results to reduce repeated API calls. Cached entries older than 30 days or exceeding a total size of approximately 50 MB are automatically purged.
*   **`chrome.storage.sync`** is used for **user-defined glossary entries only** (`user_glossary`). Because this uses Chrome Sync, those glossary entries will be synchronized to your Google account if you are signed in to Chrome with sync enabled, and may be transmitted to and stored on Google's servers as part of Chrome Sync. If you do not want your custom glossary synced, sign out of Chrome Sync or disable the "Extensions" category in Chrome Sync settings. **No API keys or browsing data are stored in `chrome.storage.sync`.**

### 4. Permissions

Our extension requests the following permissions:

*   **`activeTab`**: Access to the current tab's content when you initiate a translation or open the popup.
*   **`storage`**: Save your preferences, API keys, glossary, and translation cache.
*   **`contextMenus`**: Add "Translate Selection" and "Translate Page" entries to the right-click menu.
*   **`alarms`**: Run a daily background task that cleans up expired cache entries.
*   **`<all_urls>` content script**: The translation script needs to be able to run on any webpage you want translated. It does **not** read or transmit data from pages you do not actively translate.

### 5. Changes to This Policy

We may update our Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page.

### 6. Contact Us

If you have any questions about this Privacy Policy, please contact us at: [Your Contact Email]

---

## 中文版 (Chinese Version)

YX 纯净网页翻译（以下简称"我们"）致力于保护您的隐私。本隐私政策说明了我们的 Chrome 扩展程序如何处理您的信息。

### 1. 数据收集与使用

**我们不运营任何服务器，也不会向自己的基础设施收集、存储或传输您的任何个人数据。**

*   **无个人信息**：我们不会收集您的姓名、电子邮件地址或任何其他个人身份信息。
*   **无浏览记录**：我们不会跟踪您访问的网站或浏览历史。
*   **本地处理**：扩展的大部分设置（翻译模式、引擎选择、API 密钥、网站偏好、翻译缓存）均通过 Chrome 的 `storage.local` 与 IndexedDB 存储在您的设备本地。除第 2 节描述的第三方翻译请求外，这些数据不会离开您的浏览器。

### 2. 第三方翻译服务

为了提供翻译功能，**您访问的网页中的文本内容**将被发送到您所选择的翻译引擎。本扩展支持以下引擎，**只有您主动选择的那个引擎会接收数据**：

| 引擎 | 接口域名 | 何时发送数据 |
|---|---|---|
| **Google 翻译（免费版）**（默认） | `translate.googleapis.com` | 始终可用；同时作为其他引擎失败时的回退 |
| **Google Cloud 翻译** | `translation.googleapis.com` | 仅在您提供 Google Cloud API 密钥并选择此引擎时 |
| **DeepL** | `api-free.deepl.com` / `api.deepl.com` | 仅在您提供 DeepL API 密钥并选择此引擎时 |
| **百度翻译** | `fanyi-api.baidu.com` | 仅在您提供百度 AppID 与密钥并选择此引擎时 |
| **OpenAI** | `api.openai.com` | 仅在您提供 OpenAI API 密钥并选择此引擎时 |
| **Anthropic Claude** | `api.anthropic.com` | 仅在您提供 Claude API 密钥并选择此引擎时 |
| **DeepSeek** | `api.deepseek.com` | 仅在您提供 DeepSeek API 密钥并选择此引擎时 |
| **MiniMax** | `api.minimax.chat` | 仅在您提供 MiniMax API 密钥并选择此引擎时 |
| **智谱 GLM** | `open.bigmodel.cn` | 仅在您提供 GLM API 密钥并选择此引擎时 |
| **MyMemory**（备选） | `api.mymemory.translated.net` | 仅在 Google 免费翻译不可用时作为兜底 |

发送给所选引擎的内容：

*   您选择翻译的页面上的**可见文本内容**（或根据您的设置被自动翻译的页面）。
*   所选引擎对应的 **API 密钥**（仅发送给该引擎，全程 HTTPS 传输）。
*   不附带 Cookie 或任何个人标识符。

`<input>` / `<textarea>` 等输入框以及 `contenteditable` 元素**不会**进入翻译请求。但对于以普通文本形式显示的敏感信息（例如个人资料、财务数据等），您仍应谨慎，建议通过本扩展的"网站偏好"功能在此类站点上关闭自动翻译。

各第三方服务有各自的隐私政策，您发送的数据由其自行处理，请分别参考：

*   [Google 隐私政策](https://policies.google.com/privacy)
*   [DeepL 隐私政策](https://www.deepl.com/privacy)
*   [百度翻译使用说明](https://fanyi.baidu.com/static/translation/widget/help.html)
*   [OpenAI 隐私政策](https://openai.com/policies/privacy-policy)
*   [Anthropic 隐私政策](https://www.anthropic.com/legal/privacy)
*   [DeepSeek 隐私政策](https://www.deepseek.com/privacy)
*   [MiniMax 隐私政策](https://www.minimaxi.com/privacy)
*   [智谱 AI 用户协议](https://open.bigmodel.cn/usercenter/agreement)
*   [MyMemory 使用条款](https://mymemory.translated.net/doc/usagelimits.php)

### 3. 本地数据存储

*   **`chrome.storage.local`**：存储翻译模式、目标语言、所选引擎、API 密钥、网站偏好以及旧版兼容设置。数据仅保留在您的设备本地。
*   **IndexedDB（`yx-translate-cache`）**：存储翻译结果以减少重复 API 调用。**超过 30 天未访问的条目或总体积超过约 50 MB 的条目会自动清理。**
*   **`chrome.storage.sync`** **仅**用于您自定义的术语表（`user_glossary`）。由于该 API 会通过 Chrome Sync 同步，如果您当前已登录 Chrome 并开启了同步功能，您自定义的术语条目会被同步到您的 Google 账号，并可能作为 Chrome Sync 数据被传输并存储在 Google 服务器上。如果您不希望同步术语表，请退出 Chrome Sync 或在 Chrome 同步设置中关闭"扩展程序"分类。**API 密钥和浏览数据不会写入 `chrome.storage.sync`。**

### 4. 权限说明

本扩展程序申请以下权限：

*   **`activeTab`**：在您启动翻译或打开 popup 时访问当前标签页内容。
*   **`storage`**：保存偏好、API 密钥、术语表与翻译缓存。
*   **`contextMenus`**：在右键菜单中添加"翻译选中文本"与"翻译整个页面"选项。
*   **`alarms`**：每天后台执行一次过期缓存清理任务。
*   **`<all_urls>` 内容脚本**：翻译脚本需要在任意您希望翻译的网页上运行。**对于您未主动触发翻译的网页，扩展不会读取或上传其数据。**

### 5. 政策变更

我们可能会不时更新本隐私政策。如有更改，我们将在本页面发布新的隐私政策以通知您。

### 6. 联系我们

如果您对本隐私政策有任何疑问，请联系我们：[您的联系邮箱]
