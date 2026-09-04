# Privacy Policy for YX Clean Translator / YX 纯净网页翻译隐私政策

**Last updated / 最后更新日期:** September 4, 2026

---

## English Version

YX Clean Translator ("we", "our", or "us") is committed to protecting your privacy. This Privacy Policy explains how our Chrome extension handles your information.

### 1. Data Collection and Usage

**We do not operate any server, and we do not collect, store, or transmit any of your personal data to our own infrastructure.**

*   **No Personal Information:** We do not collect your name, email address, or any other personally identifiable information.
*   **No General Browsing History:** We do not build or share a history of the websites you visit. If you explicitly set a per-site preference, that domain and your `auto` / `never` choice are stored locally so the preference can work.
*   **Local Processing:** Most extension settings (translate mode, engine choice, API keys, site preferences, translation cache) are stored locally on your device using Chrome's `storage.local` API and IndexedDB. This data never leaves your browser, except for the third-party translation requests described in Section 2.

Chrome Web Store data categories handled by the extension are **authentication information** (API keys), **website content** (translatable page text, the page title, and supported label or hint attributes), and **web history** (domains explicitly saved in per-site preferences). These items are used only to provide the extension's translation and preference features.

### 2. Third-Party Translation Services

To provide translation functionality, **text content from the webpages you translate** is sent to the selected translation engine. Depending on the enabled comparison and fallback behavior, the same text may also be sent to Google Translate (Free) and, if Google fails, MyMemory:

| Engine | Endpoint | When data is sent |
|---|---|---|
| **Google Translate (Free)** *(default engine)* | `translate.googleapis.com` | When selected; also used for non-Google selection comparison and as fallback when another engine fails |
| **Google Cloud Translation** | `translation.googleapis.com` | Only if you provide a Google Cloud API key and select this engine |
| **DeepL** | `api-free.deepl.com` / `api.deepl.com` | Only if you provide a DeepL API key and select this engine |
| **Baidu Translate** | `fanyi-api.baidu.com` | Only if you provide a Baidu AppID + key and select this engine |
| **OpenAI** | `api.openai.com` | Only if you provide an OpenAI API key and select this engine |
| **Anthropic Claude** | `api.anthropic.com` | Only if you provide a Claude API key and select this engine |
| **DeepSeek** | `api.deepseek.com` | Only if you provide a DeepSeek API key and select this engine |
| **MiniMax** | `api.minimax.chat` | Only if you provide a MiniMax API key and select this engine |
| **Zhipu GLM** | `open.bigmodel.cn` | Only if you provide a GLM API key and select this engine |
| **MyMemory** *(fallback)* | `api.mymemory.translated.net` | Used only when Google Free translation fails, including after another engine has fallen back to Google Free |

When a non-Google engine is selected, selection comparison sends the selected text to both that engine and Google Translate (Free). If a selected non-Google engine fails, page text is retried with Google Translate (Free); if Google Free then fails, individual text may be sent to MyMemory. API keys are never forwarded to fallback services.

What is sent to the selected engine:

*   The **translatable page text, page title, and supported label or hint attributes** of pages you choose to translate (or pages auto-translated according to your settings).
*   Your **API key** for the selected engine (sent to that engine only, transmitted over HTTPS).
*   No cookies or personal identifiers.

Text typed into `<input>` / `<textarea>` controls and editable text inside `contenteditable` elements is excluded from translation requests. Labels or hint attributes on otherwise supported elements may still be translated. You should remain cautious on pages that display sensitive information as ordinary page text (e.g., personal data or financial details); sites recognized as sensitive by the extension are not auto-translated, and you can set any site to "never" in site preferences.

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

*   **`chrome.storage.local`** stores: translation mode, target language, selected engine, API keys, site preferences (domain plus `auto` / `never` choice), and migrated legacy settings. This data stays on your device.
*   **IndexedDB (`yx-translate-cache`)** stores translation results to reduce repeated API calls. Cached entries older than 30 days or exceeding a total size of approximately 50 MB are automatically purged.
*   **`chrome.storage.sync`** is used for **user-defined glossary entries only** (`user_glossary`). Because this uses Chrome Sync, those glossary entries will be synchronized to your Google account if you are signed in to Chrome with sync enabled, and may be transmitted to and stored on Google's servers as part of Chrome Sync. If you do not want your custom glossary synced, sign out of Chrome Sync or disable the "Extensions" category in Chrome Sync settings. **No API keys or browsing data are stored in `chrome.storage.sync`.**

### 4. Permissions

Our extension requests the following permissions:

*   **`activeTab`**: Access to the current tab's content when you initiate a translation or open the popup.
*   **`storage`**: Save your preferences, API keys, glossary, and translation cache.
*   **`contextMenus`**: Add "Translate Selection" and "Translate Page" entries to the right-click menu.
*   **`alarms`**: Run a daily background task that cleans up expired cache entries.
*   **`<all_urls>` content script**: The script runs on matching webpages to read local settings and decide whether translation is enabled. In the default manual mode, or when a site is set to "never", page text is not sent. If you explicitly enable automatic translation globally or for a site, visible translatable text may be sent automatically.

### 5. Limited Use

We use and transfer data only as needed to provide the extension's translation, comparison, fallback, storage, and preference features. We do not sell user data, use it for advertising or credit decisions, or allow our staff to read it. The use of information received from Google APIs will adhere to the Chrome Web Store User Data Policy, including Limited Use requirements.

### 6. Changes to This Policy

We may update our Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page.

### 7. Contact Us

If you have any questions about this Privacy Policy, please contact us through: https://github.com/XX888QM/clean-translate-extension/issues

---

## 中文版 (Chinese Version)

YX 纯净网页翻译（以下简称"我们"）致力于保护您的隐私。本隐私政策说明了我们的 Chrome 扩展程序如何处理您的信息。

### 1. 数据收集与使用

**我们不运营任何服务器，也不会向自己的基础设施收集、存储或传输您的任何个人数据。**

*   **无个人信息**：我们不会收集您的姓名、电子邮件地址或任何其他个人身份信息。
*   **无通用浏览记录**：我们不会建立或共享您的网站访问历史。只有当您明确设置单站偏好时，该域名及 `自动` / `从不` 选项才会保存在本地，以便偏好生效。
*   **本地处理**：扩展的大部分设置（翻译模式、引擎选择、API 密钥、网站偏好、翻译缓存）均通过 Chrome 的 `storage.local` 与 IndexedDB 存储在您的设备本地。除第 2 节描述的第三方翻译请求外，这些数据不会离开您的浏览器。

按照 Chrome 应用商店的数据分类，本扩展会处理**身份验证信息**（API 密钥）、**网站内容**（可翻译的页面文本、页面标题及受支持的标签或提示属性）和**网络记录**（明确保存为单站偏好的域名）。这些数据仅用于提供翻译与偏好功能。

### 2. 第三方翻译服务

为了提供翻译功能，**您选择翻译的网页文本**会发送给所选翻译引擎。根据已启用的对比和回退行为，同一文本还可能发送给 Google 免费翻译；若 Google 也失败，则可能发送给 MyMemory：

| 引擎 | 接口域名 | 何时发送数据 |
|---|---|---|
| **Google 翻译（免费版）**（默认引擎） | `translate.googleapis.com` | 选中时使用；非 Google 引擎划词对比时同时使用；其他引擎失败时作为回退 |
| **Google Cloud 翻译** | `translation.googleapis.com` | 仅在您提供 Google Cloud API 密钥并选择此引擎时 |
| **DeepL** | `api-free.deepl.com` / `api.deepl.com` | 仅在您提供 DeepL API 密钥并选择此引擎时 |
| **百度翻译** | `fanyi-api.baidu.com` | 仅在您提供百度 AppID 与密钥并选择此引擎时 |
| **OpenAI** | `api.openai.com` | 仅在您提供 OpenAI API 密钥并选择此引擎时 |
| **Anthropic Claude** | `api.anthropic.com` | 仅在您提供 Claude API 密钥并选择此引擎时 |
| **DeepSeek** | `api.deepseek.com` | 仅在您提供 DeepSeek API 密钥并选择此引擎时 |
| **MiniMax** | `api.minimax.chat` | 仅在您提供 MiniMax API 密钥并选择此引擎时 |
| **智谱 GLM** | `open.bigmodel.cn` | 仅在您提供 GLM API 密钥并选择此引擎时 |
| **MyMemory**（备选） | `api.mymemory.translated.net` | 仅在 Google 免费翻译失败时使用，包括其他引擎先回退到 Google 后仍失败的情况 |

选择非 Google 引擎时，划词对比会把所选文本同时发送给该引擎与 Google 免费翻译。所选非 Google 引擎失败后，页面文本会回退给 Google 免费翻译；Google 免费翻译仍失败时，单条文本可能继续发送给 MyMemory。API 密钥不会转发给任何回退服务。

发送给所选引擎的内容：

*   您选择翻译的页面上的**可翻译文本、页面标题及受支持的标签或提示属性**（或根据您的设置被自动翻译的页面）。
*   所选引擎对应的 **API 密钥**（仅发送给该引擎，全程 HTTPS 传输）。
*   不附带 Cookie 或任何个人标识符。

用户在 `<input>` / `<textarea>` 中输入的内容，以及 `contenteditable` 元素内的可编辑文本**不会**进入翻译请求；其他受支持元素的标签或提示属性仍可能被翻译。对于以普通页面文本形式显示的敏感信息（例如个人资料、财务数据等），您仍应谨慎；被扩展识别为敏感的站点不会自动翻译，也可通过“网站偏好”将任意站点设为“从不翻译”。

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

*   **`chrome.storage.local`**：存储翻译模式、目标语言、所选引擎、API 密钥、网站偏好（域名及 `自动` / `从不` 选项）以及旧版兼容设置。数据仅保留在您的设备本地。
*   **IndexedDB（`yx-translate-cache`）**：存储翻译结果以减少重复 API 调用。**超过 30 天未访问的条目或总体积超过约 50 MB 的条目会自动清理。**
*   **`chrome.storage.sync`** **仅**用于您自定义的术语表（`user_glossary`）。由于该 API 会通过 Chrome Sync 同步，如果您当前已登录 Chrome 并开启了同步功能，您自定义的术语条目会被同步到您的 Google 账号，并可能作为 Chrome Sync 数据被传输并存储在 Google 服务器上。如果您不希望同步术语表，请退出 Chrome Sync 或在 Chrome 同步设置中关闭"扩展程序"分类。**API 密钥和浏览数据不会写入 `chrome.storage.sync`。**

### 4. 权限说明

本扩展程序申请以下权限：

*   **`activeTab`**：在您启动翻译或打开 popup 时访问当前标签页内容。
*   **`storage`**：保存偏好、API 密钥、术语表与翻译缓存。
*   **`contextMenus`**：在右键菜单中添加"翻译选中文本"与"翻译整个页面"选项。
*   **`alarms`**：每天后台执行一次过期缓存清理任务。
*   **`<all_urls>` 内容脚本**：脚本会在匹配网页上读取本地设置并判断是否启用翻译。默认手动模式或网站设为“从不翻译”时，不会发送页面文本；若您明确开启全局或单站自动翻译，可见且符合条件的文本可能被自动发送。

### 5. 有限使用

我们仅在提供本扩展的翻译、对比、回退、存储与偏好功能所必需的范围内使用和传输数据。我们不会出售用户数据，不会将其用于广告或信用判断，也不会允许我们的工作人员读取这些数据。通过 Google API 获得的信息将遵守 Chrome 应用商店用户数据政策，包括有限使用要求。

### 6. 政策变更

我们可能会不时更新本隐私政策。如有更改，我们将在本页面发布新的隐私政策以通知您。

### 7. 联系我们

如果您对本隐私政策有任何疑问，请通过以下地址联系我们：https://github.com/XX888QM/clean-translate-extension/issues
