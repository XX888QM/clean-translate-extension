// ===== IndexedDB 翻译缓存 =====
const IDB_NAME = 'yx-translate-cache';
const IDB_VERSION = 1;
const IDB_STORE = 'translations';

function openCacheDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(IDB_NAME, IDB_VERSION);
        request.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains(IDB_STORE)) {
                db.createObjectStore(IDB_STORE);
            }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function cacheGetAll() {
    const db = await openCacheDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(IDB_STORE, 'readonly');
        const store = tx.objectStore(IDB_STORE);
        const results = {};
        const cursorReq = store.openCursor();
        cursorReq.onsuccess = (e) => {
            const cursor = e.target.result;
            if (cursor) {
                results[cursor.key] = cursor.value;
                cursor.continue();
            } else {
                resolve(results);
            }
        };
        cursorReq.onerror = () => reject(cursorReq.error);
    });
}

async function cachePutBatch(entries) {
    if (!entries || Object.keys(entries).length === 0) return;
    const db = await openCacheDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(IDB_STORE, 'readwrite');
        const store = tx.objectStore(IDB_STORE);
        for (const [key, value] of Object.entries(entries)) {
            store.put(value, key);
        }
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

async function cacheClearAll() {
    const db = await openCacheDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(IDB_STORE, 'readwrite');
        const store = tx.objectStore(IDB_STORE);
        const req = store.clear();
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

async function cacheCount() {
    const db = await openCacheDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(IDB_STORE, 'readonly');
        const store = tx.objectStore(IDB_STORE);
        const req = store.count();
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

// ===== Service Worker 生命周期保护 =====
let activeTranslations = 0;
let keepAliveInterval = null;

function startKeepAlive() {
    if (keepAliveInterval) return;
    keepAliveInterval = setInterval(() => {
        if (activeTranslations > 0) {
            // 轻量级 API 调用保持 Service Worker 活跃
            chrome.runtime.getPlatformInfo(() => {});
        } else {
            stopKeepAlive();
        }
    }, 25000);
}

function stopKeepAlive() {
    if (keepAliveInterval) {
        clearInterval(keepAliveInterval);
        keepAliveInterval = null;
    }
}

// 消息监听（统一处理所有消息类型）
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'TRANSLATE_TEXT_BATCH') {
    (async () => {
      try {
        const results = await handleBatchTranslation(request.texts);
        sendResponse({ success: true, results });
      } catch (error) {
        console.error("批量翻译失败:", error);
        sendResponse({ success: false, error: error.message });
      }
    })();
    return true;
  }

  // IndexedDB 缓存消息
  if (request.type === 'CACHE_GET_ALL') {
    cacheGetAll()
      .then(results => sendResponse({ success: true, results }))
      .catch(e => sendResponse({ success: false, error: e.message }));
    return true;
  }

  if (request.type === 'CACHE_PUT_BATCH') {
    cachePutBatch(request.entries)
      .then(() => sendResponse({ success: true }))
      .catch(e => sendResponse({ success: false, error: e.message }));
    return true;
  }

  if (request.type === 'CACHE_CLEAR') {
    cacheClearAll()
      .then(() => sendResponse({ success: true }))
      .catch(e => sendResponse({ success: false, error: e.message }));
    return true;
  }

  if (request.type === 'CACHE_COUNT') {
    cacheCount()
      .then(count => sendResponse({ success: true, count }))
      .catch(e => sendResponse({ success: false, error: e.message }));
    return true;
  }

  // 划词翻译引擎对比
  if (request.type === 'TRANSLATE_COMPARE') {
    (async () => {
      try {
        const text = request.text;
        const settings = await chrome.storage.local.get(['target_lang', 'translate_engine', 'api_keys']);
        const targetLang = settings.target_lang || 'zh-CN';
        const engine = settings.translate_engine || 'google_free';
        const apiKeys = settings.api_keys || {};

        const ENGINE_NAMES = {
          google_free: 'Google翻译(免费)', google_cloud: 'Google Cloud',
          deepl: 'DeepL', baidu: '百度翻译', openai: 'OpenAI GPT',
          claude: 'Claude', deepseek: 'DeepSeek', minimax: 'MiniMax', glm: '智谱GLM'
        };

        const results = {};

        if (engine === 'google_free') {
          // 当前引擎是 google_free，只返回一个结果
          const r = await translateBulk([text], targetLang);
          results.primary = { engine: ENGINE_NAMES.google_free, text: r[text] || text };
        } else {
          // 并行调用：当前引擎 + google_free
          const [primaryResult, googleResult] = await Promise.allSettled([
            translateByEngine([text], targetLang, engine, apiKeys),
            translateBulk([text], targetLang)
          ]);

          results.primary = {
            engine: ENGINE_NAMES[engine] || engine,
            text: primaryResult.status === 'fulfilled' ? (primaryResult.value[text] || text) : text,
            error: primaryResult.status === 'rejected' ? primaryResult.reason.message : null
          };
          results.secondary = {
            engine: ENGINE_NAMES.google_free,
            text: googleResult.status === 'fulfilled' ? (googleResult.value[text] || text) : text,
            error: googleResult.status === 'rejected' ? googleResult.reason.message : null
          };
        }

        sendResponse({ success: true, results });
      } catch (error) {
        sendResponse({ success: false, error: error.message });
      }
    })();
    return true;
  }

  if (request.type === 'GET_CACHE_SIZE') {
    chrome.storage.local.get(['translation_cache'], (result) => {
      const cache = result.translation_cache || {};
      const size = Object.keys(cache).length;
      sendResponse({ size });
    });
    return true;
  }

  if (request.type === 'CLEAR_CACHE') {
    chrome.storage.local.remove(['translation_cache', 'cache_lang'], () => {
      sendResponse({ success: true });
    });
    return true;
  }

  // 获取引擎配置
  if (request.type === 'GET_ENGINE_CONFIG') {
    chrome.storage.local.get(['translate_engine', 'api_keys'], (result) => {
      sendResponse({
        engine: result.translate_engine || 'google_free',
        apiKeys: result.api_keys || {}
      });
    });
    return true;
  }

  // 保存引擎配置
  if (request.type === 'SAVE_ENGINE_CONFIG') {
    const data = {};
    if (request.engine) data.translate_engine = request.engine;
    if (request.apiKeys) data.api_keys = request.apiKeys;
    chrome.storage.local.set(data, () => {
      sendResponse({ success: true });
    });
    return true;
  }

  // 翻译完成，更新图标状态
  if (request.type === 'TRANSLATION_DONE' && sender.tab) {
    chrome.action.setBadgeText({ text: '✓', tabId: sender.tab.id });
    chrome.action.setBadgeBackgroundColor({ color: '#188038', tabId: sender.tab.id });

    // 3秒后清除 badge
    setTimeout(() => {
      chrome.action.setBadgeText({ text: '', tabId: sender.tab.id });
    }, 3000);
  }
});

// 快捷键监听
chrome.commands.onCommand.addListener(async (command) => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id) return;

  if (command === 'translate-page') {
    chrome.tabs.sendMessage(tab.id, { type: 'START_TRANSLATE' });
    // 更新图标状态
    chrome.action.setBadgeText({ text: '...', tabId: tab.id });
    chrome.action.setBadgeBackgroundColor({ color: '#1a73e8', tabId: tab.id });
  } else if (command === 'restore-page') {
    chrome.tabs.sendMessage(tab.id, { type: 'RESTORE_ORIGINAL' });
    chrome.action.setBadgeText({ text: '', tabId: tab.id });
  }
});

// 右键菜单
chrome.runtime.onInstalled.addListener(async (details) => {
  // 从旧版 chrome.storage.local 迁移缓存到 IndexedDB
  if (details.reason === 'update') {
    try {
      const data = await chrome.storage.local.get(['translation_cache']);
      if (data.translation_cache && Object.keys(data.translation_cache).length > 0) {
        await cachePutBatch(data.translation_cache);
        chrome.storage.local.remove(['translation_cache', 'cache_lang']);
        console.log('YX翻译: 缓存已迁移到 IndexedDB');
      }
    } catch (e) {
      console.warn('YX翻译: 缓存迁移失败', e);
    }

    // 迁移旧版设置到新版翻译模式
    try {
      const settings = await chrome.storage.local.get(['auto_translate_enabled', 'excluded_domains', 'translate_mode']);
      if (!settings.translate_mode && settings.auto_translate_enabled === false) {
        chrome.storage.local.set({ translate_mode: 'manual' });
      }
      // 迁移 excluded_domains 到 site_preferences
      if (settings.excluded_domains && settings.excluded_domains.length > 0) {
        const existingPrefs = (await chrome.storage.local.get(['site_preferences'])).site_preferences || {};
        const newPrefs = { ...existingPrefs };
        for (const domain of settings.excluded_domains) {
          if (!newPrefs[domain]) {
            newPrefs[domain] = 'never';
          }
        }
        chrome.storage.local.set({ site_preferences: newPrefs });
        console.log('YX翻译: 域名排除列表已迁移到网站偏好');
      }
    } catch (e) {
      console.warn('YX翻译: 设置迁移失败', e);
    }
  }

  // 创建右键菜单：翻译选中文本
  chrome.contextMenus.create({
    id: 'translate-selection',
    title: '翻译选中文本',
    contexts: ['selection']
  });

  // 创建右键菜单：翻译整个页面
  chrome.contextMenus.create({
    id: 'translate-page',
    title: '翻译整个页面',
    contexts: ['page']
  });
});

// 右键菜单点击处理
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!tab || !tab.id) return;

  if (info.menuItemId === 'translate-selection') {
    // 翻译选中文本（触发划词翻译气泡）
    chrome.tabs.sendMessage(tab.id, {
      type: 'TRANSLATE_SELECTION',
      text: info.selectionText
    });
  } else if (info.menuItemId === 'translate-page') {
    chrome.tabs.sendMessage(tab.id, { type: 'START_TRANSLATE' });
    chrome.action.setBadgeText({ text: '...', tabId: tab.id });
    chrome.action.setBadgeBackgroundColor({ color: '#1a73e8', tabId: tab.id });
  }
});

async function handleBatchTranslation(texts) {
  activeTranslations++;
  startKeepAlive();
  try {
    return await _handleBatchTranslation(texts);
  } finally {
    activeTranslations--;
    if (activeTranslations <= 0) {
      activeTranslations = 0;
      setTimeout(() => {
        if (activeTranslations <= 0) stopKeepAlive();
      }, 5000);
    }
  }
}

async function _handleBatchTranslation(texts) {
  // 一次性读取目标语言和引擎设置
  let targetLang = 'zh-CN';
  let engine = 'google_free';
  let apiKeys = {};
  try {
    const settings = await chrome.storage.local.get(['target_lang', 'translate_engine', 'api_keys']);
    if (settings.target_lang) targetLang = settings.target_lang;
    if (settings.translate_engine) engine = settings.translate_engine;
    if (settings.api_keys) apiKeys = settings.api_keys;
  } catch (e) { /* 使用默认值 */ }

  // 将文本按字符总量分组，每组合并为一次 API 请求
  const MAX_BULK_CHARS = 1500; // 单次请求最大原文字符数
  const bulkGroups = [];
  let currentGroup = [];
  let currentLen = 0;

  for (const text of texts) {
    if (currentLen + text.length > MAX_BULK_CHARS && currentGroup.length > 0) {
      bulkGroups.push(currentGroup);
      currentGroup = [];
      currentLen = 0;
    }
    currentGroup.push(text);
    currentLen += text.length + 1; // +1 for \n separator
  }
  if (currentGroup.length > 0) bulkGroups.push(currentGroup);

  // 并行发送合并请求（最多 8 个并发）
  const results = {};
  const PARALLEL = 8;

  for (let i = 0; i < bulkGroups.length; i += PARALLEL) {
    const batch = bulkGroups.slice(i, i + PARALLEL);
    const promises = batch.map(group =>
      Promise.race([
        translateByEngine(group, targetLang, engine, apiKeys),
        new Promise(resolve => {
          setTimeout(() => {
            const fallback = {};
            group.forEach(t => fallback[t] = t);
            resolve(fallback);
          }, 15000); // LLM引擎需要更长超时
        })
      ])
    );
    const batchResults = await Promise.all(promises);
    batchResults.forEach(r => Object.assign(results, r));
  }
  return results;
}

// 根据引擎选择路由到不同翻译函数
async function translateByEngine(texts, targetLang, engine, apiKeys) {
  try {
    switch (engine) {
      case 'google_cloud':
        return await translateGoogleCloud(texts, targetLang, apiKeys.google_cloud);
      case 'deepl':
        return await translateDeepL(texts, targetLang, apiKeys.deepl);
      case 'baidu':
        return await translateBaidu(texts, targetLang, apiKeys.baidu_appid, apiKeys.baidu_key);
      case 'openai':
      case 'claude':
      case 'deepseek':
      case 'minimax':
      case 'glm':
        return await translateWithLLM(texts, targetLang, engine, apiKeys[engine]);
      case 'google_free':
      default:
        return await translateBulk(texts, targetLang);
    }
  } catch (e) {
    console.warn(`YX翻译: ${engine} 引擎翻译失败，回退到免费Google翻译`, e.message);
    // 非google_free引擎失败时回退到免费Google翻译
    if (engine !== 'google_free') {
      return await translateBulk(texts, targetLang);
    }
    // google_free本身失败，返回原文
    const fallback = {};
    texts.forEach(t => fallback[t] = t);
    return fallback;
  }
}

// 多条文本合并为一次 API 请求（用 \n 分隔）
async function translateBulk(texts, targetLang) {
  // 单条文本走原有逻辑
  if (texts.length === 1) {
    const r = await translateSingle(texts[0], 0, targetLang);
    return { [r.original]: r.translated };
  }

  try {
    const joined = texts.join('\n');
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=${encodeURIComponent(targetLang)}&dt=t&q=${encodeURIComponent(joined)}`;
    const response = await fetch(url);

    if (response.status === 429) {
      // 限流时回退到逐条翻译（带重试）
      return await translateBulkFallback(texts, targetLang);
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();

    // 拼接完整译文
    let fullTranslation = '';
    if (data && Array.isArray(data[0])) {
      fullTranslation = data[0]
        .filter(s => s && s[0])
        .map(s => s[0])
        .join('');
    }

    // 按 \n 拆分回各条译文
    const translated = fullTranslation.split('\n');

    // 行数匹配：直接映射
    if (translated.length === texts.length) {
      const results = {};
      for (let i = 0; i < texts.length; i++) {
        let t = translated[i];
        if (targetLang.startsWith('zh')) {
          t = await refineTranslation(texts[i], t);
        }
        results[texts[i]] = t || texts[i];
      }
      return results;
    }

    // 行数不匹配：回退逐条翻译
    console.warn(`YX翻译: 合并翻译行数不匹配 (期望 ${texts.length}, 得到 ${translated.length})，回退逐条翻译`);
    return await translateBulkFallback(texts, targetLang);
  } catch (e) {
    console.warn('YX翻译: 合并翻译失败，回退逐条翻译', e.message);
    return await translateBulkFallback(texts, targetLang);
  }
}

// 合并翻译失败时的逐条回退
async function translateBulkFallback(texts, targetLang) {
  const results = {};
  const promises = texts.map(text =>
    translateSingle(text, 0, targetLang)
      .catch(() => ({ original: text, translated: text }))
  );
  const individual = await Promise.all(promises);
  individual.forEach(r => {
    results[r.original] = r.translated;
  });
  return results;
}

// ========== Google Cloud Translation API v2 ==========
async function translateGoogleCloud(texts, targetLang, apiKey) {
  if (!apiKey) throw new Error('Google Cloud API密钥未配置');

  const results = {};
  // Google Cloud API 支持批量翻译，直接发送数组
  const url = `https://translation.googleapis.com/language/translate/v2?key=${encodeURIComponent(apiKey)}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      q: texts,
      target: targetLang, // Google Cloud API 支持 'zh-CN'、'zh-TW' 等完整语言代码
      format: 'text'
    })
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Google Cloud API 错误 ${response.status}: ${err}`);
  }

  const data = await response.json();
  const translations = data.data?.translations || [];

  for (let i = 0; i < texts.length; i++) {
    let translated = translations[i]?.translatedText || texts[i];
    // 术语校对
    if (targetLang.startsWith('zh')) {
      translated = await refineTranslation(texts[i], translated);
    }
    results[texts[i]] = translated;
  }
  return results;
}

// ========== DeepL API ==========
async function translateDeepL(texts, targetLang, apiKey) {
  if (!apiKey) throw new Error('DeepL API密钥未配置');

  // 判断是 Free 还是 Pro API（Free 密钥以 ':fx' 结尾）
  const isFree = apiKey.endsWith(':fx');
  const baseUrl = isFree
    ? 'https://api-free.deepl.com/v2/translate'
    : 'https://api.deepl.com/v2/translate';

  // DeepL 目标语言映射
  const deeplLangMap = {
    'zh-CN': 'ZH-HANS', 'zh-TW': 'ZH-HANT', 'zh': 'ZH-HANS',
    'en': 'EN-US', 'pt': 'PT-BR', 'pt-BR': 'PT-BR', 'pt-PT': 'PT-PT'
  };
  const deeplTarget = deeplLangMap[targetLang] || targetLang.toUpperCase();

  // DeepL 支持批量文本（通过多个 text 参数）
  const params = new URLSearchParams();
  texts.forEach(t => params.append('text', t));
  params.append('target_lang', deeplTarget);

  const response = await fetch(baseUrl, {
    method: 'POST',
    headers: {
      'Authorization': `DeepL-Auth-Key ${apiKey}`,
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: params.toString()
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`DeepL API 错误 ${response.status}: ${err}`);
  }

  const data = await response.json();
  const results = {};
  const translations = data.translations || [];

  for (let i = 0; i < texts.length; i++) {
    let translated = translations[i]?.text || texts[i];
    if (targetLang.startsWith('zh')) {
      translated = await refineTranslation(texts[i], translated);
    }
    results[texts[i]] = translated;
  }
  return results;
}

// ========== 百度翻译 API ==========

// 纯JS实现的MD5函数（Service Worker不支持crypto.subtle.digest同步生成MD5）
function md5(string) {
  function md5cycle(x, k) {
    let a = x[0], b = x[1], c = x[2], d = x[3];
    a = ff(a, b, c, d, k[0], 7, -680876936);
    d = ff(d, a, b, c, k[1], 12, -389564586);
    c = ff(c, d, a, b, k[2], 17, 606105819);
    b = ff(b, c, d, a, k[3], 22, -1044525330);
    a = ff(a, b, c, d, k[4], 7, -176418897);
    d = ff(d, a, b, c, k[5], 12, 1200080426);
    c = ff(c, d, a, b, k[6], 17, -1473231341);
    b = ff(b, c, d, a, k[7], 22, -45705983);
    a = ff(a, b, c, d, k[8], 7, 1770035416);
    d = ff(d, a, b, c, k[9], 12, -1958414417);
    c = ff(c, d, a, b, k[10], 17, -42063);
    b = ff(b, c, d, a, k[11], 22, -1990404162);
    a = ff(a, b, c, d, k[12], 7, 1804603682);
    d = ff(d, a, b, c, k[13], 12, -40341101);
    c = ff(c, d, a, b, k[14], 17, -1502002290);
    b = ff(b, c, d, a, k[15], 22, 1236535329);
    a = gg(a, b, c, d, k[1], 5, -165796510);
    d = gg(d, a, b, c, k[6], 9, -1069501632);
    c = gg(c, d, a, b, k[11], 14, 643717713);
    b = gg(b, c, d, a, k[0], 20, -373897302);
    a = gg(a, b, c, d, k[5], 5, -701558691);
    d = gg(d, a, b, c, k[10], 9, 38016083);
    c = gg(c, d, a, b, k[15], 14, -660478335);
    b = gg(b, c, d, a, k[4], 20, -405537848);
    a = gg(a, b, c, d, k[9], 5, 568446438);
    d = gg(d, a, b, c, k[14], 9, -1019803690);
    c = gg(c, d, a, b, k[3], 14, -187363961);
    b = gg(b, c, d, a, k[8], 20, 1163531501);
    a = gg(a, b, c, d, k[13], 5, -1444681467);
    d = gg(d, a, b, c, k[2], 9, -51403784);
    c = gg(c, d, a, b, k[7], 14, 1735328473);
    b = gg(b, c, d, a, k[12], 20, -1926607734);
    a = hh(a, b, c, d, k[5], 4, -378558);
    d = hh(d, a, b, c, k[8], 11, -2022574463);
    c = hh(c, d, a, b, k[11], 16, 1839030562);
    b = hh(b, c, d, a, k[14], 23, -35309556);
    a = hh(a, b, c, d, k[1], 4, -1530992060);
    d = hh(d, a, b, c, k[4], 11, 1272893353);
    c = hh(c, d, a, b, k[7], 16, -155497632);
    b = hh(b, c, d, a, k[10], 23, -1094730640);
    a = hh(a, b, c, d, k[13], 4, 681279174);
    d = hh(d, a, b, c, k[0], 11, -358537222);
    c = hh(c, d, a, b, k[3], 16, -722521979);
    b = hh(b, c, d, a, k[6], 23, 76029189);
    a = hh(a, b, c, d, k[9], 4, -640364487);
    d = hh(d, a, b, c, k[12], 11, -421815835);
    c = hh(c, d, a, b, k[15], 16, 530742520);
    b = hh(b, c, d, a, k[2], 23, -995338651);
    a = ii(a, b, c, d, k[0], 6, -198630844);
    d = ii(d, a, b, c, k[7], 10, 1126891415);
    c = ii(c, d, a, b, k[14], 15, -1416354905);
    b = ii(b, c, d, a, k[5], 21, -57434055);
    a = ii(a, b, c, d, k[12], 6, 1700485571);
    d = ii(d, a, b, c, k[3], 10, -1894986606);
    c = ii(c, d, a, b, k[10], 15, -1051523);
    b = ii(b, c, d, a, k[1], 21, -2054922799);
    a = ii(a, b, c, d, k[8], 6, 1873313359);
    d = ii(d, a, b, c, k[15], 10, -30611744);
    c = ii(c, d, a, b, k[6], 15, -1560198380);
    b = ii(b, c, d, a, k[13], 21, 1309151649);
    a = ii(a, b, c, d, k[4], 6, -145523070);
    d = ii(d, a, b, c, k[11], 10, -1120210379);
    c = ii(c, d, a, b, k[2], 15, 718787259);
    b = ii(b, c, d, a, k[9], 21, -343485551);
    x[0] = add32(a, x[0]);
    x[1] = add32(b, x[1]);
    x[2] = add32(c, x[2]);
    x[3] = add32(d, x[3]);
  }

  function cmn(q, a, b, x, s, t) {
    a = add32(add32(a, q), add32(x, t));
    return add32((a << s) | (a >>> (32 - s)), b);
  }
  function ff(a, b, c, d, x, s, t) { return cmn((b & c) | ((~b) & d), a, b, x, s, t); }
  function gg(a, b, c, d, x, s, t) { return cmn((b & d) | (c & (~d)), a, b, x, s, t); }
  function hh(a, b, c, d, x, s, t) { return cmn(b ^ c ^ d, a, b, x, s, t); }
  function ii(a, b, c, d, x, s, t) { return cmn(c ^ (b | (~d)), a, b, x, s, t); }

  function md5blk(s) {
    const md5blks = [];
    for (let i = 0; i < 64; i += 4) {
      md5blks[i >> 2] = s.charCodeAt(i) + (s.charCodeAt(i + 1) << 8) +
        (s.charCodeAt(i + 2) << 16) + (s.charCodeAt(i + 3) << 24);
    }
    return md5blks;
  }

  function md5blk_array(a) {
    const md5blks = [];
    for (let i = 0; i < 64; i += 4) {
      md5blks[i >> 2] = a[i] + (a[i + 1] << 8) + (a[i + 2] << 16) + (a[i + 3] << 24);
    }
    return md5blks;
  }

  function add32(a, b) {
    return (a + b) & 0xFFFFFFFF;
  }

  function rhex(n) {
    const hex_chr = '0123456789abcdef';
    let s = '';
    for (let j = 0; j < 4; j++) {
      s += hex_chr.charAt((n >> (j * 8 + 4)) & 0x0F) + hex_chr.charAt((n >> (j * 8)) & 0x0F);
    }
    return s;
  }

  function hex(x) {
    return x.map(rhex).join('');
  }

  // 将UTF-8字符串转为字节数组
  function toUTF8Array(str) {
    const utf8 = [];
    for (let i = 0; i < str.length; i++) {
      let charcode = str.charCodeAt(i);
      if (charcode < 0x80) utf8.push(charcode);
      else if (charcode < 0x800) {
        utf8.push(0xc0 | (charcode >> 6), 0x80 | (charcode & 0x3f));
      } else if (charcode < 0xd800 || charcode >= 0xe000) {
        utf8.push(0xe0 | (charcode >> 12), 0x80 | ((charcode >> 6) & 0x3f), 0x80 | (charcode & 0x3f));
      } else {
        i++;
        charcode = 0x10000 + (((charcode & 0x3ff) << 10) | (str.charCodeAt(i) & 0x3ff));
        utf8.push(0xf0 | (charcode >> 18), 0x80 | ((charcode >> 12) & 0x3f),
          0x80 | ((charcode >> 6) & 0x3f), 0x80 | (charcode & 0x3f));
      }
    }
    return utf8;
  }

  const bytes = toUTF8Array(string);
  const n = bytes.length;
  let state = [1732584193, -271733879, -1732584194, 271733878];
  let i;

  // 处理完整的64字节块
  for (i = 64; i <= n; i += 64) {
    const block = bytes.slice(i - 64, i);
    md5cycle(state, md5blk_array(block));
  }

  // 填充剩余字节
  const tail = bytes.slice(i - 64);
  const padded = new Array(64).fill(0);
  for (let j = 0; j < tail.length; j++) padded[j] = tail[j];
  padded[tail.length] = 0x80;

  if (tail.length > 55) {
    md5cycle(state, md5blk_array(padded));
    padded.fill(0);
  }

  // 添加长度（位数，小端序）
  const bitLen = n * 8;
  padded[56] = bitLen & 0xFF;
  padded[57] = (bitLen >> 8) & 0xFF;
  padded[58] = (bitLen >> 16) & 0xFF;
  padded[59] = (bitLen >> 24) & 0xFF;
  // 对于超过 2^32 位的消息需要高32位，但翻译文本不会那么长
  md5cycle(state, md5blk_array(padded));

  return hex(state);
}

async function translateBaidu(texts, targetLang, appId, key) {
  if (!appId || !key) throw new Error('百度翻译AppID或密钥未配置');

  // 百度翻译目标语言映射
  const baiduLangMap = {
    'zh-CN': 'zh', 'zh-TW': 'cht', 'zh': 'zh',
    'en': 'en', 'ja': 'jp', 'ko': 'kor',
    'fr': 'fra', 'de': 'de', 'ru': 'ru',
    'es': 'spa', 'pt': 'pt', 'it': 'it',
    'ar': 'ara', 'th': 'th', 'vi': 'vie'
  };
  const to = baiduLangMap[targetLang] || targetLang;

  // 百度 API 支持用 \n 分隔的多条文本
  const query = texts.join('\n');
  const salt = Date.now().toString();
  const sign = md5(appId + query + salt + key);

  const params = new URLSearchParams({
    q: query, from: 'auto', to, appid: appId, salt, sign
  });

  const response = await fetch('https://fanyi-api.baidu.com/api/trans/vip/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: params.toString()
  });

  if (!response.ok) throw new Error(`百度翻译 HTTP ${response.status}`);

  const data = await response.json();
  if (data.error_code) throw new Error(`百度翻译错误 ${data.error_code}: ${data.error_msg}`);

  const results = {};
  const transResult = data.trans_result || [];

  // 百度翻译返回 src/dst 对，按 \n 分隔的文本会返回多条结果
  if (transResult.length === texts.length) {
    for (let i = 0; i < texts.length; i++) {
      let translated = transResult[i]?.dst || texts[i];
      if (targetLang.startsWith('zh')) {
        translated = await refineTranslation(texts[i], translated);
      }
      results[texts[i]] = translated;
    }
  } else {
    // 行数不匹配时尝试按原文匹配
    const dstMap = new Map();
    transResult.forEach(r => dstMap.set(r.src, r.dst));
    for (const text of texts) {
      let translated = dstMap.get(text) || text;
      if (targetLang.startsWith('zh') && translated !== text) {
        translated = await refineTranslation(text, translated);
      }
      results[text] = translated;
    }
  }
  return results;
}

// ========== LLM 统一翻译接口（OpenAI / Claude / DeepSeek） ==========
async function translateWithLLM(texts, targetLang, engine, apiKey) {
  if (!apiKey) throw new Error(`${engine} API密钥未配置`);

  // 目标语言名称映射（用于提示词）
  const langNames = {
    'zh-CN': '简体中文', 'zh-TW': '繁体中文', 'zh': '中文',
    'en': '英文', 'ja': '日文', 'ko': '韩文',
    'fr': '法文', 'de': '德文', 'ru': '俄文',
    'es': '西班牙文', 'pt': '葡萄牙文', 'it': '意大利文'
  };
  const langName = langNames[targetLang] || targetLang;

  // 将多条文本用编号列表格式发送，减少API调用次数
  const numberedTexts = texts.map((t, i) => `[${i + 1}] ${t}`).join('\n');

  const systemPrompt = `你是专业翻译。将以下编号文本逐条翻译为${langName}，保持原文格式。每行输出格式：[编号] 译文。只输出翻译结果，不要解释。`;
  const userMessage = numberedTexts;

  // 各引擎配置
  const engineConfig = {
    openai: {
      url: 'https://api.openai.com/v1/chat/completions',
      model: 'gpt-4o-mini',
      headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      buildBody: (sys, user) => JSON.stringify({
        model: 'gpt-4o-mini', messages: [
          { role: 'system', content: sys },
          { role: 'user', content: user }
        ], temperature: 0.1
      })
    },
    claude: {
      url: 'https://api.anthropic.com/v1/messages',
      model: 'claude-sonnet-4-20250514',
      headers: {
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json'
      },
      buildBody: (sys, user) => JSON.stringify({
        model: 'claude-sonnet-4-20250514', max_tokens: 4096,
        system: sys,
        messages: [{ role: 'user', content: user }],
        temperature: 0.1
      })
    },
    deepseek: {
      url: 'https://api.deepseek.com/v1/chat/completions',
      model: 'deepseek-chat',
      headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      buildBody: (sys, user) => JSON.stringify({
        model: 'deepseek-chat', messages: [
          { role: 'system', content: sys },
          { role: 'user', content: user }
        ], temperature: 0.1
      })
    },
    minimax: {
      url: 'https://api.minimax.chat/v1/text/chatcompletion_v2',
      model: 'MiniMax-M2.5',
      headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      buildBody: (sys, user) => JSON.stringify({
        model: 'MiniMax-M2.5', messages: [
          { role: 'system', content: sys },
          { role: 'user', content: user }
        ], temperature: 0.1
      })
    },
    glm: {
      url: 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
      model: 'glm-5',
      headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      buildBody: (sys, user) => JSON.stringify({
        model: 'glm-5', messages: [
          { role: 'system', content: sys },
          { role: 'user', content: user }
        ], temperature: 0.1
      })
    }
  };

  const config = engineConfig[engine];
  if (!config) throw new Error(`不支持的LLM引擎: ${engine}`);

  const response = await fetch(config.url, {
    method: 'POST',
    headers: config.headers,
    body: config.buildBody(systemPrompt, userMessage)
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`${engine} API 错误 ${response.status}: ${err}`);
  }

  const data = await response.json();

  // 提取回复内容（Claude 和 OpenAI/DeepSeek 的响应格式不同）
  let replyText = '';
  if (engine === 'claude') {
    replyText = data.content?.[0]?.text || '';
  } else {
    replyText = data.choices?.[0]?.message?.content || '';
  }

  // 解析编号格式的翻译结果
  const results = {};
  const lines = replyText.split('\n').filter(l => l.trim());

  for (const line of lines) {
    // 匹配 [数字] 译文 格式
    const match = line.match(/^\[(\d+)\]\s*(.+)$/);
    if (match) {
      const idx = parseInt(match[1]) - 1;
      if (idx >= 0 && idx < texts.length) {
        results[texts[idx]] = match[2].trim();
      }
    }
  }

  // 未匹配到的文本保留原文
  for (const text of texts) {
    if (!results[text]) {
      results[text] = text;
    }
  }

  // LLM翻译不需要术语校对（大模型翻译质量足够好）
  return results;
}

const AI_GLOSSARY = {
  // ========== AI 模型名称（保持原文不翻译） ==========
  // OpenAI 系列
  "chatgpt": [["聊天GPT", "ChatGPT"], ["聊天gpt", "ChatGPT"]],
  "gpt-4": [["GPT-4", "GPT-4"]],
  "gpt-4o": [["GPT-4o", "GPT-4o"]],
  "gpt-4 turbo": [["GPT-4涡轮", "GPT-4 Turbo"]],
  "gpt-5": [["GPT-5", "GPT-5"]],
  "dall-e": [["达尔·E", "DALL-E"], ["达利", "DALL-E"]],
  "dall·e": [["达尔·E", "DALL-E"], ["达利", "DALL-E"]],
  "whisper": [["低语", "Whisper"], ["耳语", "Whisper"]],
  "codex": [["法典", "Codex"], ["抄本", "Codex"]],
  "sora": [["索拉", "Sora"]],

  // Anthropic 系列
  "claude": [["克劳德", "Claude"], ["克洛德", "Claude"]],
  "claude opus": [["克劳德作品", "Claude Opus"]],
  "claude sonnet": [["克劳德十四行诗", "Claude Sonnet"]],
  "claude haiku": [["克劳德俳句", "Claude Haiku"]],

  // Google 系列
  "gemini": [["双子座", "Gemini"]],
  "gemini pro": [["双子座专业版", "Gemini Pro"]],
  "gemini ultra": [["双子座超级版", "Gemini Ultra"]],
  "gemini nano": [["双子座纳米", "Gemini Nano"]],
  "gemini flash": [["双子座闪光", "Gemini Flash"]],
  "gemma": [["宝石", "Gemma"]],
  "palm": [["棕榈", "PaLM"], ["手掌", "PaLM"]],
  "palm 2": [["棕榈2", "PaLM 2"]],
  "bard": [["吟游诗人", "Bard"]],
  "lamda": [["拉姆达", "LaMDA"]],
  "t5": [["T5", "T5"]],

  // Meta 系列
  "llama": [["美洲驼", "LLaMA"], ["羊驼", "LLaMA"], ["骆驼", "LLaMA"]],
  "llama 2": [["美洲驼2", "LLaMA 2"], ["羊驼2", "LLaMA 2"]],
  "llama 3": [["美洲驼3", "LLaMA 3"], ["羊驼3", "LLaMA 3"]],
  "llama 4": [["美洲驼4", "LLaMA 4"], ["羊驼4", "LLaMA 4"]],
  "code llama": [["代码羊驼", "Code LLaMA"]],
  "segment anything": [["分割任何东西", "Segment Anything"]],
  "sam": [["山姆", "SAM"]],
  "imagebind": [["图像绑定", "ImageBind"]],

  // Mistral AI 系列
  "mistral": [["西北风", "Mistral"]],
  "mistral large": [["西北风大型", "Mistral Large"]],
  "mistral small": [["西北风小型", "Mistral Small"]],
  "mistral medium": [["西北风中型", "Mistral Medium"]],
  "mixtral": [["混合", "Mixtral"]],
  "pixtral": [["像素", "Pixtral"]],

  // 阿里巴巴 系列
  "qwen": [["清雯", "Qwen通义千问"]],
  "qwen2": [["清雯2", "Qwen2"]],
  "qwq": [["QwQ", "QwQ"]],
  "tongyi qianwen": [["通义千问", "通义千问"]],

  // DeepSeek 系列
  "deepseek": [["深度搜索", "DeepSeek"], ["深度探索", "DeepSeek"], ["深度寻求", "DeepSeek"]],
  "deepseek-v2": [["深度搜索V2", "DeepSeek-V2"]],
  "deepseek-v3": [["深度搜索V3", "DeepSeek-V3"]],
  "deepseek-r1": [["深度搜索R1", "DeepSeek-R1"]],
  "deepseek coder": [["深度搜索编码器", "DeepSeek Coder"]],

  // 微软 系列
  "phi": [["斐", "Phi"], ["披", "Phi"]],
  "phi-2": [["斐-2", "Phi-2"]],
  "phi-3": [["斐-3", "Phi-3"]],
  "phi-4": [["斐-4", "Phi-4"]],
  "copilot": [["副驾驶", "Copilot"]],
  "bing chat": [["必应聊天", "Bing Chat"]],

  // xAI 系列
  "grok": [["格罗克", "Grok"]],
  "grok-2": [["格罗克-2", "Grok-2"]],
  "grok-3": [["格罗克-3", "Grok-3"]],

  // 其他知名模型
  "falcon": [["猎鹰", "Falcon"]],
  "falcon 40b": [["猎鹰40B", "Falcon 40B"]],
  "falcon 180b": [["猎鹰180B", "Falcon 180B"]],
  "cohere": [["连贯", "Cohere"]],
  "command r": [["命令R", "Command R"]],
  "command r+": [["命令R+", "Command R+"]],
  "granite": [["花岗岩", "Granite"]],
  "nemotron": [["尼莫特龙", "Nemotron"]],
  "vicuna": [["骆马", "Vicuna"], ["小羊驼", "Vicuna"]],
  "alpaca": [["羊驼", "Alpaca"]],
  "dolly": [["多莉", "Dolly"]],
  "bloom": [["绽放", "BLOOM"], ["开花", "BLOOM"]],
  "starcoder": [["星际编码器", "StarCoder"]],
  "codestral": [["代码星", "Codestral"]],
  "stable diffusion": [["稳定扩散", "Stable Diffusion"]],
  "midjourney": [["中途", "Midjourney"], ["中间旅程", "Midjourney"]],
  "runway": [["跑道", "Runway"]],
  "hugging face": [["拥抱脸", "Hugging Face"]],
  "huggingface": [["拥抱脸", "Hugging Face"]],
  "yi": [["易", "Yi零一万物"]],
  "yi-34b": [["易-34B", "Yi-34B"]],
  "baichuan": [["百川", "百川"]],
  "glm": [["GLM", "GLM"]],
  "chatglm": [["聊天GLM", "ChatGLM"]],
  "zhipu": [["智谱", "智谱"]],
  "minimax": [["极小极大", "MiniMax"]],
  "moonshot": [["月球射击", "Moonshot月之暗面"]],
  "kimi": [["基米", "Kimi"]],
  "ernie": [["厄尼", "文心一言ERNIE"]],
  "ernie bot": [["厄尼机器人", "文心一言"]],
  "wenxin": [["文心", "文心"]],
  "spark": [["火花", "讯飞星火"]],

  // 图像生成模型
  "imagen": [["图像", "Imagen"]],
  "muse": [["缪斯", "Muse"]],
  "parti": [["派对", "Parti"]],
  "phenaki": [["费纳基", "Phenaki"]],
  "make-a-video": [["制作视频", "Make-A-Video"]],

  // ========== 基础 AI/ML 术语 ==========
  "agent": [["代理", "智能体"], ["经纪人", "智能体"], ["代理人", "智能体"]],
  "agents": [["代理", "智能体"], ["经纪人", "智能体"], ["代理人", "智能体"]],
  "agentic": [["代理性", "智能体化"], ["代理的", "智能体化"]],
  "transformer": [["变压器", "Transformer"]],
  "transformers": [["变压器", "Transformers"]],
  "token": [["代币", "Token"], ["令牌", "Token"], ["标记", "Token"]],
  "tokens": [["代币", "Tokens"], ["令牌", "Tokens"]],
  "tokenizer": [["分词器", "Tokenizer"], ["标记器", "Tokenizer"]],
  "tokenization": [["标记化", "分词"]],
  "prompt": [["迅速", "提示词"], ["提示", "提示词"], ["促使", "提示词"]],
  "prompts": [["提示", "提示词"]],
  "prompting": [["提示", "提示工程"]],
  "prompt engineering": [["提示工程", "提示词工程"]],
  "zero-shot": [["零射", "零样本"], ["零镜头", "零样本"], ["零次射击", "零样本"]],
  "few-shot": [["少射", "少样本"], ["几射", "少样本"], ["几次射击", "少样本"]],
  "one-shot": [["一次性", "单样本"], ["一枪", "单样本"], ["一次射击", "单样本"]],
  "chain of thought": [["思想链", "思维链"]],
  "chain-of-thought": [["思想链", "思维链"]],
  "cot": [["婴儿床", "CoT思维链"]],
  "robustness": [["稳健性", "鲁棒性"], ["健壮性", "鲁棒性"]],
  "corpus": [["尸体", "语料库"], ["全集", "语料库"], ["身体", "语料库"]],
  "corpora": [["尸体", "语料库"], ["全集", "语料库"]],
  "epoch": [["时代", "轮次"], ["纪元", "轮次"], ["时期", "轮次"]],
  "epochs": [["时代", "轮次"], ["纪元", "轮次"]],

  // ========== 模型训练相关 ==========
  "fine-tune": [["微调", "微调"], ["精调", "微调"], ["罚款", "微调"]],
  "fine-tuning": [["微调", "微调"], ["精调", "微调"]],
  "finetune": [["微调", "微调"], ["罚款调整", "微调"]],
  "finetuning": [["微调", "微调"]],
  "pre-train": [["预训练", "预训练"], ["预先训练", "预训练"]],
  "pre-training": [["预训练", "预训练"]],
  "pretrain": [["预训练", "预训练"]],
  "pretraining": [["预训练", "预训练"]],
  "inference": [["推断", "推理"], ["推论", "推理"]],
  "embedding": [["嵌入", "嵌入向量"], ["埋入", "嵌入向量"], ["镶嵌", "嵌入向量"]],
  "embeddings": [["嵌入", "嵌入向量"], ["埋入", "嵌入向量"]],
  "latent": [["潜在", "隐变量"], ["潜伏", "隐变量"]],
  "latent space": [["潜在空间", "隐空间"]],
  "attention": [["注意", "注意力机制"], ["关注", "注意力机制"]],
  "self-attention": [["自我关注", "自注意力"], ["自我注意", "自注意力"]],
  "cross-attention": [["交叉注意", "交叉注意力"]],
  "gradient": [["坡度", "梯度"], ["渐变", "梯度"], ["斜率", "梯度"]],
  "gradients": [["坡度", "梯度"], ["渐变", "梯度"]],
  "backpropagation": [["反向传播", "反向传播"]],
  "backprop": [["反向传播", "反向传播"]],
  "overfitting": [["过度拟合", "过拟合"], ["过度配合", "过拟合"]],
  "underfitting": [["欠拟合", "欠拟合"], ["拟合不足", "欠拟合"]],
  "regularization": [["正则化", "正则化"]],
  "dropout": [["辍学", "Dropout"], ["退出", "Dropout"], ["丢失", "Dropout"]],
  "batch size": [["批量大小", "批大小"]],
  "learning rate": [["学习率", "学习率"]],
  "hyperparameter": [["超参数", "超参数"]],
  "hyperparameters": [["超参数", "超参数"]],
  "loss function": [["损失函数", "损失函数"]],
  "loss": [["损失", "损失值"], ["丢失", "损失值"]],
  "optimizer": [["优化器", "优化器"]],
  "convergence": [["收敛", "收敛"], ["聚合", "收敛"]],
  "weights": [["权重", "权重"], ["重量", "权重"]],
  "weight": [["权重", "权重"], ["重量", "权重"]],
  "bias": [["偏见", "偏置"], ["偏向", "偏置"]],
  "activation": [["激活", "激活函数"]],
  "relu": [["热卢", "ReLU"]],
  "sigmoid": [["S形", "Sigmoid"]],
  "softmax": [["软最大", "Softmax"]],
  "cross entropy": [["交叉熵", "交叉熵"]],
  "cross-entropy": [["交叉熵", "交叉熵"]],

  // ========== 神经网络架构 ==========
  "neural network": [["神经网络", "神经网络"]],
  "neural networks": [["神经网络", "神经网络"]],
  "deep learning": [["深度学习", "深度学习"]],
  "machine learning": [["机器学习", "机器学习"]],
  "cnn": [["有线电视新闻网", "CNN卷积神经网络"]],
  "convolutional neural network": [["卷积神经网络", "卷积神经网络"]],
  "rnn": [["循环神经网络", "RNN循环神经网络"]],
  "recurrent neural network": [["循环神经网络", "循环神经网络"]],
  "lstm": [["长短期记忆", "LSTM"]],
  "long short-term memory": [["长短期记忆", "长短期记忆网络"]],
  "gru": [["GRU", "GRU门控循环单元"]],
  "gated recurrent unit": [["门控循环单元", "门控循环单元"]],
  "gan": [["甘", "GAN生成对抗网络"]],
  "generative adversarial network": [["生成对抗网络", "生成对抗网络"]],
  "vae": [["增值税", "VAE变分自编码器"]],
  "variational autoencoder": [["变分自编码器", "变分自编码器"]],
  "autoencoder": [["自动编码器", "自编码器"]],
  "diffusion": [["扩散", "扩散模型"]],
  "diffusion model": [["扩散模型", "扩散模型"]],
  "encoder": [["编码器", "编码器"]],
  "decoder": [["解码器", "解码器"]],
  "encoder-decoder": [["编码器-解码器", "编解码器"]],
  "autoregressive": [["自回归", "自回归"]],
  "feedforward": [["前馈", "前馈"]],
  "feed-forward": [["前馈", "前馈"]],
  "multi-head": [["多头", "多头"]],
  "multi-head attention": [["多头注意力", "多头注意力"]],
  "mha": [["MHA", "MHA多头注意力"]],
  "layer normalization": [["层归一化", "层归一化"]],
  "layer norm": [["层规范", "LayerNorm"]],
  "batch normalization": [["批归一化", "批归一化"]],
  "batch norm": [["批规范", "BatchNorm"]],
  "positional encoding": [["位置编码", "位置编码"]],
  "residual connection": [["残差连接", "残差连接"]],
  "skip connection": [["跳过连接", "跳跃连接"]],
  "mixture of experts": [["专家混合", "混合专家模型"]],
  "moe": [["萌", "MoE混合专家"]],

  // ========== LLM 大语言模型相关 ==========
  "hallucination": [["幻觉", "幻觉"], ["产生幻觉", "幻觉"]],
  "hallucinations": [["幻觉", "幻觉"]],
  "hallucinate": [["产生幻觉", "幻觉"], ["出现幻觉", "幻觉"]],
  "confabulation": [["虚构", "幻觉"], ["编造", "幻觉"]],
  "context window": [["上下文窗口", "上下文窗口"]],
  "context length": [["上下文长度", "上下文长度"]],
  "rag": [["抹布", "RAG检索增强生成"], ["破布", "RAG检索增强生成"]],
  "retrieval-augmented generation": [["检索增强生成", "检索增强生成(RAG)"]],
  "retrieval augmented generation": [["检索增强生成", "检索增强生成(RAG)"]],
  "rlhf": [["RLHF", "RLHF人类反馈强化学习"]],
  "reinforcement learning from human feedback": [["人类反馈强化学习", "人类反馈强化学习(RLHF)"]],
  "dpo": [["DPO", "DPO直接偏好优化"]],
  "direct preference optimization": [["直接偏好优化", "直接偏好优化"]],
  "ppo": [["PPO", "PPO近端策略优化"]],
  "in-context learning": [["上下文学习", "上下文学习"]],
  "in context learning": [["上下文学习", "上下文学习"]],
  "icl": [["ICL", "ICL上下文学习"]],
  "grounding": [["接地", "知识落地"], ["基础", "知识落地"]],
  "alignment": [["对齐", "对齐"], ["校准", "对齐"]],
  "instruction tuning": [["指令调优", "指令微调"]],
  "instruction-tuning": [["指令调优", "指令微调"]],
  "sft": [["SFT", "SFT监督微调"]],
  "supervised fine-tuning": [["监督微调", "监督微调"]],
  "system prompt": [["系统提示", "系统提示词"]],
  "temperature": [["温度", "温度参数"]],
  "top-p": [["顶部p", "Top-P采样"]],
  "top-k": [["顶部k", "Top-K采样"]],
  "nucleus sampling": [["核采样", "核采样"]],
  "sampling": [["采样", "采样"]],
  "beam search": [["光束搜索", "束搜索"]],
  "greedy decoding": [["贪婪解码", "贪心解码"]],
  "speculative decoding": [["投机解码", "推测解码"]],
  "kv cache": [["KV缓存", "KV缓存"]],
  "key-value cache": [["键值缓存", "KV缓存"]],
  "quantization": [["量化", "量化"]],
  "quantized": [["量化的", "量化"]],
  "distillation": [["蒸馏", "知识蒸馏"]],
  "knowledge distillation": [["知识蒸馏", "知识蒸馏"]],
  "pruning": [["修剪", "剪枝"]],
  "sparsity": [["稀疏性", "稀疏性"]],
  "sparse": [["稀疏", "稀疏"]],
  "dense": [["密集", "稠密"]],
  "scaling law": [["缩放定律", "缩放法则"]],
  "scaling laws": [["缩放定律", "缩放法则"]],
  "emergent": [["新兴", "涌现"]],
  "emergent abilities": [["新兴能力", "涌现能力"]],
  "emergence": [["出现", "涌现"]],

  // ========== 评估指标相关 ==========
  "benchmark": [["基准", "基准测试"], ["标杆", "基准测试"]],
  "benchmarks": [["基准", "基准测试"]],
  "baseline": [["基线", "基线"], ["底线", "基线"]],
  "accuracy": [["准确性", "准确率"], ["精度", "准确率"]],
  "precision": [["精度", "精确率"], ["精密度", "精确率"]],
  "recall": [["召回", "召回率"], ["回忆", "召回率"]],
  "f1 score": [["f1分数", "F1分数"]],
  "f1-score": [["f1分数", "F1分数"]],
  "perplexity": [["困惑", "困惑度"], ["复杂性", "困惑度"]],
  "bleu": [["蓝色", "BLEU分数"], ["蓝", "BLEU分数"]],
  "bleu score": [["蓝色分数", "BLEU分数"]],
  "rouge": [["胭脂", "ROUGE分数"], ["红色", "ROUGE分数"]],
  "rouge score": [["胭脂分数", "ROUGE分数"]],
  "mmlu": [["MMLU", "MMLU基准"]],
  "hellaswag": [["地狱沼泽", "HellaSwag基准"]],
  "truthfulqa": [["真实问答", "TruthfulQA基准"]],
  "humaneval": [["人类评估", "HumanEval基准"]],
  "gsm8k": [["GSM8K", "GSM8K数学基准"]],
  "arc": [["弧", "ARC推理基准"]],
  "winogrande": [["维诺格兰德", "WinoGrande基准"]],
  "aime": [["目标", "AIME数学竞赛"]],
  "math benchmark": [["数学基准", "数学基准测试"]],
  "leaderboard": [["排行榜", "排行榜"]],
  "sota": [["索塔", "SOTA最先进"]],
  "state of the art": [["最先进的", "最先进"]],
  "state-of-the-art": [["最先进的", "最先进"]],

  // ========== 数据处理相关 ==========
  "dataset": [["数据集", "数据集"]],
  "datasets": [["数据集", "数据集"]],
  "data augmentation": [["数据增强", "数据增强"]],
  "preprocessing": [["预处理", "预处理"]],
  "normalization": [["归一化", "归一化"], ["标准化", "归一化"]],
  "vectorization": [["矢量化", "向量化"]],
  "vector": [["向量", "向量"], ["矢量", "向量"]],
  "vectors": [["向量", "向量"], ["矢量", "向量"]],
  "dimensionality reduction": [["降维", "降维"]],
  "feature extraction": [["特征提取", "特征提取"]],
  "feature engineering": [["特征工程", "特征工程"]],
  "label": [["标签", "标签"]],
  "labels": [["标签", "标签"]],
  "annotation": [["注释", "标注"]],
  "annotations": [["注释", "标注"]],

  // ========== 应用场景相关 ==========
  "text generation": [["文本生成", "文本生成"]],
  "text summarization": [["文本摘要", "文本摘要"]],
  "summarization": [["总结", "摘要"]],
  "question answering": [["问答", "问答"]],
  "qa": [["质量保证", "问答"]],
  "sentiment analysis": [["情感分析", "情感分析"]],
  "named entity recognition": [["命名实体识别", "命名实体识别"]],
  "ner": [["NER", "NER命名实体识别"]],
  "machine translation": [["机器翻译", "机器翻译"]],
  "speech recognition": [["语音识别", "语音识别"]],
  "asr": [["ASR", "ASR语音识别"]],
  "text-to-speech": [["文本转语音", "文本转语音"]],
  "tts": [["TTS", "TTS文本转语音"]],
  "speech-to-text": [["语音转文本", "语音转文本"]],
  "stt": [["STT", "STT语音转文本"]],
  "image captioning": [["图像字幕", "图像描述"]],
  "object detection": [["目标检测", "目标检测"]],
  "semantic segmentation": [["语义分割", "语义分割"]],
  "image classification": [["图像分类", "图像分类"]],
  "ocr": [["光学字符识别", "OCR文字识别"]],
  "optical character recognition": [["光学字符识别", "光学字符识别"]],
  "recommendation system": [["推荐系统", "推荐系统"]],
  "recommender": [["推荐者", "推荐系统"]],
  "chatbot": [["聊天机器人", "聊天机器人"]],
  "conversational ai": [["对话式人工智能", "对话式AI"]],

  // ========== 常见缩写 ==========
  "llm": [["法学硕士", "大语言模型"], ["法律硕士", "大语言模型"]],
  "llms": [["法学硕士", "大语言模型"], ["法律硕士", "大语言模型"]],
  "vlm": [["VLM", "VLM视觉语言模型"]],
  "vlms": [["VLM", "视觉语言模型"]],
  "slm": [["SLM", "SLM小语言模型"]],
  "nlp": [["自然语言处理", "NLP自然语言处理"]],
  "nlg": [["自然语言生成", "NLG自然语言生成"]],
  "nlu": [["自然语言理解", "NLU自然语言理解"]],
  "cv": [["简历", "CV计算机视觉"]],
  "computer vision": [["计算机视觉", "计算机视觉"]],
  "gpt": [["通用技术", "GPT"]],
  "bert": [["伯特", "BERT"]],
  "api": [["应用程序接口", "API"]],
  "apis": [["应用程序接口", "APIs"]],
  "sdk": [["软件开发工具包", "SDK"]],
  "gpu": [["图形处理器", "GPU"]],
  "gpus": [["图形处理器", "GPUs"]],
  "tpu": [["张量处理器", "TPU"]],
  "tpus": [["张量处理器", "TPUs"]],
  "cpu": [["中央处理器", "CPU"]],
  "cpus": [["中央处理器", "CPUs"]],
  "cuda": [["酷达", "CUDA"]],
  "tensor": [["张量", "张量"]],
  "tensors": [["张量", "张量"]],
  "agi": [["人工通用智能", "AGI通用人工智能"]],
  "artificial general intelligence": [["人工通用智能", "通用人工智能"]],
  "asi": [["人工超级智能", "ASI超级人工智能"]],
  "artificial superintelligence": [["人工超级智能", "超级人工智能"]],

  // ========== 多模态相关 ==========
  "multimodal": [["多式联运", "多模态"], ["多模式", "多模态"]],
  "multi-modal": [["多式联运", "多模态"], ["多模式", "多模态"]],
  "vision-language": [["视觉语言", "视觉-语言"]],
  "vision language model": [["视觉语言模型", "视觉语言模型"]],
  "text-to-image": [["文本到图像", "文生图"]],
  "image-to-text": [["图像到文本", "图生文"]],
  "text-to-video": [["文本到视频", "文生视频"]],
  "video-to-text": [["视频到文本", "视频生文"]],
  "text-to-audio": [["文本到音频", "文生音频"]],
  "audio-to-text": [["音频到文本", "音频转文字"]],
  "text-to-3d": [["文本到3D", "文生3D"]],
  "image-to-image": [["图像到图像", "图生图"]],
  "inpainting": [["修复", "图像修复"]],
  "outpainting": [["外绘", "图像扩展"]],
  "img2img": [["图像到图像", "图生图"]],
  "txt2img": [["文本到图像", "文生图"]],

  // ========== 安全与伦理 ==========
  "jailbreak": [["越狱", "越狱攻击"]],
  "jailbreaking": [["越狱", "越狱攻击"]],
  "prompt injection": [["提示注入", "提示词注入"]],
  "prompt hacking": [["提示黑客", "提示词攻击"]],
  "adversarial": [["对抗性", "对抗性"]],
  "adversarial attack": [["对抗性攻击", "对抗攻击"]],
  "red teaming": [["红队", "红队测试"]],
  "red team": [["红队", "红队"]],
  "safety": [["安全", "安全性"]],
  "guardrails": [["护栏", "安全护栏"]],
  "content filter": [["内容过滤器", "内容过滤"]],
  "moderation": [["审核", "内容审核"], ["审核", "版务管理"]],
  "fairness": [["公平", "公平性"]],
  "interpretability": [["可解释性", "可解释性"]],
  "explainability": [["可解释性", "可解释性"]],
  "transparency": [["透明度", "透明性"]],
  "accountability": [["问责制", "可问责性"]],
  "ethical ai": [["道德人工智能", "AI伦理"]],
  "responsible ai": [["负责任的人工智能", "负责任AI"]],

  // ========== 工具与框架 ==========
  "pytorch": [["火炬", "PyTorch"]],
  "tensorflow": [["张量流", "TensorFlow"]],
  "keras": [["凯拉斯", "Keras"]],
  "jax": [["杰克斯", "JAX"]],
  "onnx": [["ONNX", "ONNX"]],
  "triton": [["海神", "Triton"]],
  "vllm": [["VLLM", "vLLM"]],
  "langchain": [["语言链", "LangChain"]],
  "llamaindex": [["羊驼索引", "LlamaIndex"]],
  "ollama": [["奥拉马", "Ollama"]],
  "lmstudio": [["LM工作室", "LM Studio"]],
  "openai": [["开放人工智能", "OpenAI"]],
  "anthropic": [["拟人化", "Anthropic"]],
  "deepmind": [["深度思维", "DeepMind"]],
  "nvidia": [["英伟达", "NVIDIA"]],
  "meta ai": [["元人工智能", "Meta AI"]],
  "google ai": [["谷歌人工智能", "Google AI"]],
  "microsoft ai": [["微软人工智能", "Microsoft AI"]],

  // ========== 其他重要术语 ==========
  "open source": [["开源", "开源"]],
  "open-source": [["开源", "开源"]],
  "closed source": [["闭源", "闭源"]],
  "proprietary": [["专有的", "闭源"]],
  "parameter": [["参数", "参数"]],
  "parameters": [["参数", "参数"]],
  "billion parameters": [["十亿参数", "B参数"]],
  "7b": [["7B", "70亿参数"]],
  "13b": [["13B", "130亿参数"]],
  "70b": [["70B", "700亿参数"]],
  "deployment": [["部署", "部署"]],
  "serving": [["服务", "模型服务"]],
  "model serving": [["模型服务", "模型服务"]],
  "api endpoint": [["API端点", "API端点"]],
  "latency": [["延迟", "延迟"]],
  "throughput": [["吞吐量", "吞吐量"]],
  "batch inference": [["批量推理", "批量推理"]],
  "real-time": [["实时", "实时"]],
  "streaming": [["流式传输", "流式输出"]],
  "async": [["异步", "异步"]],
  "synchronous": [["同步", "同步"]],
  "asynchronous": [["异步", "异步"]],

  // ========== GitHub / Git 平台术语 ==========
  "repository": [["存储库", "仓库"], ["仓库", "仓库"]],
  "repositories": [["存储库", "仓库"]],
  "repo": [["回购", "仓库"], ["存储库", "仓库"]],
  "repos": [["回购", "仓库"]],
  "fork": [["叉子", "Fork分支"], ["分叉", "Fork"]],
  "forked": [["分叉", "已Fork"]],
  "forks": [["叉子", "Forks"]],
  "pull request": [["拉取请求", "Pull Request"], ["拉请求", "PR"]],
  "pull requests": [["拉取请求", "Pull Requests"]],
  "pr": [["公关", "PR"]],
  "prs": [["公关", "PRs"]],
  "merge": [["合并", "合并"]],
  "merged": [["合并", "已合并"]],
  "merging": [["合并", "合并中"]],
  "commit": [["提交", "提交"], ["承诺", "提交"]],
  "commits": [["提交", "提交"]],
  "committed": [["承诺", "已提交"]],
  "committer": [["提交者", "提交者"]],
  "branch": [["分支", "分支"], ["树枝", "分支"]],
  "branches": [["分支", "分支"], ["树枝", "分支"]],
  "main branch": [["主分支", "主分支"]],
  "master branch": [["主分支", "主分支"]],
  "feature branch": [["功能分支", "功能分支"]],
  "checkout": [["结账", "检出"], ["退房", "检出"]],
  "clone": [["克隆", "克隆"]],
  "cloned": [["克隆", "已克隆"]],
  "push": [["推", "推送"], ["推动", "推送"]],
  "pushed": [["推", "已推送"]],
  "pull": [["拉", "拉取"], ["拉动", "拉取"]],
  "fetch": [["获取", "获取"], ["取", "拉取"]],
  "issue": [["问题", "Issue"]],
  "issues": [["问题", "Issues"]],
  "open issue": [["打开问题", "开放Issue"]],
  "closed issue": [["关闭问题", "已关闭Issue"]],
  "star": [["星星", "Star"], ["明星", "Star"]],
  "stars": [["星星", "Stars"], ["明星", "Stars"]],
  "starred": [["星标", "已Star"]],
  "stargazers": [["观星者", "Star用户"]],
  "watch": [["观看", "Watch关注"]],
  "watchers": [["观察者", "关注者"]],
  "readme": [["自述", "README"]],
  "readme.md": [["自述文件", "README.md"]],
  "license": [["许可证", "开源协议"]],
  "contributor": [["贡献者", "贡献者"]],
  "contributors": [["贡献者", "贡献者"]],
  "contribution": [["贡献", "贡献"]],
  "contributions": [["贡献", "贡献"]],
  "maintainer": [["维护者", "维护者"]],
  "maintainers": [["维护者", "维护者"]],
  "release": [["发布", "Release版本"]],
  "releases": [["发布", "Releases"]],
  "tag": [["标签", "标签"]],
  "tags": [["标签", "标签"]],
  "gist": [["要点", "Gist代码片段"]],
  "gists": [["要点", "Gists"]],
  "diff": [["差异", "差异对比"]],
  "rebase": [["变基", "Rebase"]],
  "rebased": [["变基", "已Rebase"]],
  "squash": [["压扁", "压缩提交"]],
  "cherry-pick": [["樱桃采摘", "Cherry-pick"]],
  "stash": [["藏匿", "暂存"]],
  "gitignore": [["git忽略", ".gitignore"]],
  "workflow": [["工作流程", "工作流"]],
  "workflows": [["工作流程", "工作流"]],
  "action": [["行动", "Action"]],
  "actions": [["行动", "Actions"]],
  "github actions": [["GitHub行动", "GitHub Actions"]],
  "ci/cd": [["CI/CD", "CI/CD持续集成"]],
  "continuous integration": [["持续集成", "持续集成"]],
  "continuous deployment": [["持续部署", "持续部署"]],
  "pipeline": [["管道", "流水线"]],
  "code review": [["代码审查", "代码审查"]],
  "review": [["审查", "审查"]],
  "reviewer": [["审稿人", "审查者"]],
  "reviewers": [["审稿人", "审查者"]],
  "approve": [["批准", "批准"]],
  "approved": [["批准", "已批准"]],
  "request changes": [["请求更改", "请求修改"]],
  "milestone": [["里程碑", "里程碑"]],
  "milestones": [["里程碑", "里程碑"]],
  "project board": [["项目板", "项目看板"]],
  "kanban": [["看板", "看板"]],
  "assignee": [["受让人", "指派人"]],
  "assignees": [["受让人", "指派人"]],
  "discussion": [["讨论", "讨论"]],
  "discussions": [["讨论", "讨论区"]],
  "wiki": [["维基", "Wiki文档"]],
  "sponsor": [["赞助商", "赞助者"]],
  "sponsors": [["赞助商", "赞助者"]],
  "sponsoring": [["赞助", "赞助"]],
  "dependabot": [["依赖机器人", "Dependabot"]],
  "codespace": [["代码空间", "Codespace"]],
  "codespaces": [["代码空间", "Codespaces"]],

  // ========== Twitter / X 平台术语 ==========
  "tweet": [["推文", "推文"], ["鸣叫", "推文"]],
  "tweets": [["推文", "推文"], ["鸣叫", "推文"]],
  "retweet": [["转推", "转推"]],
  "retweets": [["转推", "转推"]],
  "retweeted": [["转推", "已转推"]],
  "quote tweet": [["引用推文", "引用推文"]],
  "thread": [["线程", "推文串"], ["主题", "帖子串"], ["线程", "子区"]],
  "threads": [["线程", "推文串"], ["线程", "子区"]],
  "hashtag": [["标签", "话题标签"]],
  "hashtags": [["标签", "话题标签"]],
  "trending": [["趋势", "热门趋势"]],
  "trends": [["趋势", "热门"]],
  "follower": [["追随者", "粉丝"]],
  "followers": [["追随者", "粉丝"]],
  "following": [["关注", "关注中"]],
  "follow": [["关注", "关注"], ["跟随", "关注"]],
  "unfollow": [["取消关注", "取关"]],
  "timeline": [["时间线", "时间线"]],
  "feed": [["饲料", "动态"], ["提要", "信息流"]],
  "home feed": [["主页提要", "首页动态"]],
  "for you": [["为你", "为你推荐"]],
  "dm": [["DM", "私信"]],
  "dms": [["DM", "私信"]],
  "direct message": [["直接消息", "私信"]],
  "direct messages": [["直接消息", "私信"]],
  "mention": [["提及", "@提及"]],
  "mentions": [["提及", "@提及"]],
  "handle": [["句柄", "用户名"]],
  "username": [["用户名", "用户名"]],
  "verified": [["已验证", "已认证"]],
  "verification": [["验证", "认证"]],
  "blue check": [["蓝色勾号", "蓝V认证"]],
  "like": [["喜欢", "点赞"]],
  "likes": [["喜欢", "点赞"]],
  "liked": [["喜欢", "已点赞"]],
  "bookmark": [["书签", "收藏"]],
  "bookmarks": [["书签", "收藏"]],
  "bookmarked": [["已添加书签", "已收藏"]],
  "mute": [["静音", "静音"]],
  "muted": [["静音", "已静音"]],
  "block": [["阻止", "拉黑"]],
  "blocked": [["阻止", "已拉黑"]],
  "report": [["报告", "举报"]],
  "x premium": [["X高级版", "X Premium"]],
  "twitter blue": [["推特蓝", "Twitter Blue"]],
  "space": [["空间", "语音空间"], ["空间", "Space应用"]],
  "spaces": [["空间", "语音空间"], ["空间", "Spaces"]],
  "fleet": [["舰队", "限时动态"]],
  "fleets": [["舰队", "限时动态"]],
  "moment": [["时刻", "精选时刻"]],
  "moments": [["时刻", "精选时刻"]],
  "list": [["列表", "列表"]],
  "lists": [["列表", "列表"]],
  "community": [["社区", "社区"]],
  "communities": [["社区", "社区"]],
  "impression": [["印象", "曝光量"]],
  "impressions": [["印象", "曝光量"]],
  "engagement": [["参与", "互动量"]],
  "engagements": [["参与", "互动量"]],
  "analytics": [["分析", "数据分析"]],

  // ========== Reddit 平台术语 ==========
  "subreddit": [["子版块", "子版块"]],
  "subreddits": [["子版块", "子版块"]],
  "upvote": [["赞成票", "点赞"]],
  "upvotes": [["赞成票", "点赞数"]],
  "upvoted": [["赞成", "已点赞"]],
  "downvote": [["反对票", "踩"]],
  "downvotes": [["反对票", "踩数"]],
  "downvoted": [["反对", "已踩"]],
  "karma": [["业力", "Karma声望"], ["因果", "Karma"]],
  "post karma": [["帖子业力", "帖子Karma"]],
  "comment karma": [["评论业力", "评论Karma"]],
  "crosspost": [["交叉发布", "转发"]],
  "crossposts": [["交叉发布", "转发"]],
  "x-post": [["交叉帖子", "转发"]],
  "flair": [["天赋", "用户标签"], ["才华", "帖子分类"]],
  "flairs": [["天赋", "标签"]],
  "user flair": [["用户天赋", "用户标签"]],
  "post flair": [["帖子天赋", "帖子分类"]],
  "mod": [["模组", "版主"]],
  "mods": [["模组", "版主"]],
  "moderator": [["主持人", "版主"]],
  "moderators": [["主持人", "版主"]],
  "ama": [["AMA", "AMA问我任何事"]],
  "ask me anything": [["问我任何事", "AMA"]],
  "iama": [["我是一个", "IAMA"]],
  "op": [["运营", "楼主"], ["操作", "原帖作者"]],
  "original poster": [["原始发帖人", "楼主"]],
  "tl;dr": [["太长不看", "摘要"]],
  "tldr": [["太长不看", "摘要"]],
  "eli5": [["ELI5", "简单解释"]],
  "explain like i'm 5": [["像我5岁一样解释", "通俗解释"]],
  "lurker": [["潜伏者", "潜水用户"]],
  "lurking": [["潜伏", "潜水"]],
  "redditor": [["红迪用户", "Reddit用户"]],
  "redditors": [["红迪用户", "Reddit用户"]],
  "reddit gold": [["红迪金币", "Reddit Gold"]],
  "reddit premium": [["红迪高级版", "Reddit Premium"]],
  "award": [["奖项", "打赏"]],
  "awards": [["奖项", "打赏"]],
  "gilded": [["镀金", "获得打赏"]],
  "cake day": [["蛋糕日", "Reddit注册纪念日"]],
  "front page": [["首页", "首页热门"]],
  "hot": [["热", "热门"]],
  "new": [["新", "最新"]],
  "top": [["顶部", "最高赞"]],
  "rising": [["上升", "上升中"]],
  "controversial": [["有争议", "争议"]],
  "best": [["最佳", "最佳"]],
  "nsfw": [["不适合工作", "成人内容"]],
  "spoiler": [["剧透", "剧透"]],
  "oc": [["原创内容", "原创"]],
  "original content": [["原创内容", "原创"]],
  "repost": [["转帖", "转载"]],
  "reposts": [["转帖", "转载"]],
  "brigading": [["刷帖", "恶意刷帖"]],

  // ========== Discord 平台术语 ==========
  "server": [["服务器", "服务器"]],
  "servers": [["服务器", "服务器"]],
  "discord server": [["Discord服务器", "Discord服务器"]],
  "channel": [["频道", "频道"]],
  "channels": [["频道", "频道"]],
  "text channel": [["文字频道", "文字频道"]],
  "voice channel": [["语音频道", "语音频道"]],
  "stage channel": [["舞台频道", "舞台频道"]],
  "forum channel": [["论坛频道", "论坛频道"]],
  "category": [["类别", "分类"]],
  "categories": [["类别", "分类"]],
  "role": [["角色", "身份组"]],
  "roles": [["角色", "身份组"]],
  "permission": [["权限", "权限"]],
  "permissions": [["权限", "权限"]],
  "bot": [["机器人", "机器人"]],
  "bots": [["机器人", "机器人"]],
  "discord bot": [["Discord机器人", "Discord机器人"]],
  "webhook": [["网络钩子", "Webhook"]],
  "webhooks": [["网络钩子", "Webhooks"]],
  "ping": [["平", "提醒"], ["乒", "@提醒"]],
  "pinged": [["被ping", "被@"]],
  "pinging": [["ping", "@提醒"]],
  "nitro": [["硝基", "Nitro会员"]],
  "discord nitro": [["Discord硝基", "Discord Nitro"]],
  "boost": [["提升", "助力"]],
  "boosts": [["提升", "助力"]],
  "boosted": [["提升", "已助力"]],
  "server boost": [["服务器提升", "服务器助力"]],
  "level": [["级别", "等级"]],
  "levels": [["级别", "等级"]],
  "emoji": [["表情符号", "表情"]],
  "emojis": [["表情符号", "表情"]],
  "custom emoji": [["自定义表情", "自定义表情"]],
  "sticker": [["贴纸", "贴纸"]],
  "stickers": [["贴纸", "贴纸"]],
  "reaction": [["反应", "表情回应"]],
  "reactions": [["反应", "表情回应"]],
  "slash command": [["斜杠命令", "斜杠命令"]],
  "slash commands": [["斜杠命令", "斜杠命令"]],
  "invite": [["邀请", "邀请"]],
  "invites": [["邀请", "邀请"]],
  "invite link": [["邀请链接", "邀请链接"]],
  "ban": [["禁止", "封禁"]],
  "banned": [["禁止", "已封禁"]],
  "kick": [["踢", "踢出"]],
  "kicked": [["踢", "已踢出"]],
  "timeout": [["超时", "禁言"]],
  "timed out": [["超时", "已禁言"]],
  "afk": [["离开", "挂机"]],
  "afk channel": [["离开频道", "挂机频道"]],
  "slowmode": [["慢速模式", "慢速模式"]],
  "stage": [["阶段", "舞台"]],
  "stages": [["阶段", "舞台"]],
  "activity": [["活动", "活动状态"]],
  "activities": [["活动", "活动"]],
  "rich presence": [["丰富存在", "游戏状态"]],
  "status": [["状态", "状态"]],
  "online": [["在线", "在线"]],
  "idle": [["闲置", "离开"]],
  "do not disturb": [["请勿打扰", "勿扰"]],
  "dnd": [["DND", "勿扰"]],
  "invisible": [["隐身", "隐身"]],
  "offline": [["离线", "离线"]],

  // ========== Facebook / Instagram 平台术语 ==========
  "post": [["帖子", "帖子"], ["发布", "发帖"]],
  "posts": [["帖子", "帖子"]],
  "story": [["故事", "动态"], ["故事", "限时动态"]],
  "stories": [["故事", "限时动态"]],
  "reel": [["卷轴", "短视频"], ["卷", "Reels"]],
  "reels": [["卷轴", "短视频"]],
  "instagram reel": [["Instagram卷轴", "Instagram短视频"]],
  "facebook reel": [["Facebook卷轴", "Facebook短视频"]],
  "highlight": [["亮点", "精选动态"]],
  "highlights": [["亮点", "精选集"]],
  "caption": [["标题", "文案"], ["说明", "图片描述"]],
  "captions": [["标题", "文案"]],
  "filter": [["过滤器", "滤镜"]],
  "filters": [["过滤器", "滤镜"]],
  "live": [["直播", "直播"]],
  "go live": [["开始直播", "开播"]],
  "going live": [["正在直播", "正在直播"]],
  "comment": [["评论", "评论"]],
  "comments": [["评论", "评论"]],
  "share": [["分享", "分享"]],
  "shares": [["分享", "分享"]],
  "shared": [["分享", "已分享"]],
  "save": [["保存", "收藏"]],
  "saved": [["已保存", "已收藏"]],
  "saves": [["保存", "收藏"]],
  "explore": [["探索", "发现"]],
  "explore page": [["探索页面", "发现页"]],
  "suggested": [["建议", "推荐"]],
  "suggestions": [["建议", "推荐"]],
  "profile": [["个人资料", "主页"]],
  "profiles": [["个人资料", "主页"]],
  "bio": [["生物", "简介"], ["个人简介", "简介"]],
  "link in bio": [["简介中的链接", "简介链接"]],
  "grid": [["网格", "九宫格"]],
  "feed post": [["动态帖子", "信息流帖子"]],
  "news feed": [["新闻提要", "动态消息"]],
  "marketplace": [["市场", "二手市场"]],
  "group": [["群组", "群组"]],
  "groups": [["群组", "群组"]],
  "page": [["页面", "主页"]],
  "pages": [["页面", "主页"]],
  "event": [["事件", "活动"]],
  "events": [["事件", "活动"]],
  "messenger": [["信使", "Messenger"]],
  "instagram direct": [["Instagram直接", "Instagram私信"]],
  "reach": [["到达", "触达量"]],
  "engagement rate": [["参与率", "互动率"]],
  "influencer": [["影响者", "网红"]],
  "influencers": [["影响者", "网红"]],
  "creator": [["创作者", "创作者"]],
  "creators": [["创作者", "创作者"]],
  "content creator": [["内容创作者", "内容创作者"]],
  "ugc": [["用户生成内容", "UGC用户内容"]],
  "user generated content": [["用户生成内容", "用户原创内容"]],
  "collab": [["合作", "合拍"]],
  "collaboration": [["合作", "合作"]],
  "duet": [["二重唱", "合拍"]],
  "stitch": [["缝合", "拼接"]],

  // ========== Hugging Face 平台术语 ==========
  "hugging face hub": [["拥抱脸枢纽", "Hugging Face Hub"]],
  "model hub": [["模型中心", "模型库"]],
  "model card": [["模型卡", "模型卡片"]],
  "model cards": [["模型卡", "模型卡片"]],
  "dataset card": [["数据集卡", "数据集卡片"]],
  "huggingface spaces": [["拥抱脸空间", "Hugging Face Spaces"]],
  "gradio": [["格拉迪奥", "Gradio"]],
  "streamlit": [["流线型", "Streamlit"]],
  "inference api": [["推理API", "推理API"]],
  "inference endpoint": [["推理端点", "推理端点"]],
  "inference endpoints": [["推理端点", "推理端点"]],
  "autotrain": [["自动训练", "AutoTrain"]],
  "accelerate": [["加速", "Accelerate"]],
  "peft": [["PEFT", "PEFT参数高效微调"]],
  "lora": [["洛拉", "LoRA"]],
  "qlora": [["QLora", "QLoRA"]],
  "safetensors": [["安全张量", "SafeTensors"]],
  "gguf": [["GGUF", "GGUF格式"]],
  "ggml": [["GGML", "GGML格式"]],
  "awq": [["AWQ", "AWQ量化"]],
  "gptq": [["GPTQ", "GPTQ量化"]],
  "transformers library": [["变压器库", "Transformers库"]],
  "diffusers": [["扩散器", "Diffusers"]],
  "datasets library": [["数据集库", "Datasets库"]],
  "tokenizers library": [["分词器库", "Tokenizers库"]],
  "evaluate": [["评估", "Evaluate"]],
  "open llm leaderboard": [["开放LLM排行榜", "Open LLM排行榜"]],
  "trending models": [["趋势模型", "热门模型"]],
  "trending datasets": [["趋势数据集", "热门数据集"]],
  "trending spaces": [["趋势空间", "热门Spaces"]],

  // ========== 通用社交媒体术语 ==========
  "viral": [["病毒式", "爆款"]],
  "go viral": [["病毒式传播", "爆火"]],
  "viral content": [["病毒内容", "爆款内容"]],
  "algorithm": [["算法", "算法"]],
  "shadow ban": [["影子禁令", "限流"]],
  "shadowban": [["影子禁令", "限流"]],
  "shadowbanned": [["被影子禁令", "被限流"]],
  "content moderation": [["内容审核", "内容审核"]],
  "community guidelines": [["社区指南", "社区规范"]],
  "terms of service": [["服务条款", "服务条款"]],
  "privacy policy": [["隐私政策", "隐私政策"]],
  "two-factor authentication": [["双因素认证", "两步验证"]],
  "2fa": [["2FA", "两步验证"]],
  "login": [["登录", "登录"]],
  "logout": [["注销", "退出登录"]],
  "sign up": [["注册", "注册"]],
  "sign in": [["登录", "登录"]],
  "notification": [["通知", "通知"]],
  "notifications": [["通知", "通知"]],
  "push notification": [["推送通知", "推送通知"]],
  "email notification": [["电子邮件通知", "邮件通知"]],
  "settings": [["设置", "设置"]],
  "account settings": [["帐户设置", "账号设置"]],
  "privacy settings": [["隐私设置", "隐私设置"]],
  "dark mode": [["深色模式", "深色模式"]],
  "light mode": [["浅色模式", "浅色模式"]],
  "avatar": [["头像", "头像"]],
  "profile picture": [["个人资料图片", "头像"]],
  "banner": [["横幅", "背景图"]],
  "cover photo": [["封面照片", "封面图"]],
  "pinned": [["固定", "置顶"]],
  "pinned post": [["固定帖子", "置顶帖"]],
  "pinned tweet": [["固定推文", "置顶推文"]],
  "archive": [["存档", "存档"]],
  "archived": [["存档", "已存档"]]
};

// 备选翻译源：MyMemory API
async function translateFallback(text, targetLang = 'zh-CN') {
  try {
    // MyMemory 使用 'source|target' 格式的语言对，auto 表示自动检测源语言
    const langPair = `auto|${targetLang}`;
    const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=${encodeURIComponent(langPair)}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`MyMemory HTTP ${response.status}`);
    const data = await response.json();
    if (data && data.responseData && data.responseData.translatedText) {
      return data.responseData.translatedText;
    }
    return null;
  } catch (e) {
    console.warn('YX翻译: MyMemory 备选翻译失败', e.message);
    return null;
  }
}

// 翻译单条文本（带重试和退避机制）
async function translateSingle(text, retryCount = 0, targetLang = 'zh-CN') {
  if (!text || !text.trim()) return { original: text, translated: text };

  const MAX_RETRIES = 2;
  const RETRY_DELAY = 1000; // 基础延迟 1 秒

  try {
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=${encodeURIComponent(targetLang)}&dt=t&q=${encodeURIComponent(text)}`;
    const response = await fetch(url);

    // 处理 429 限流错误
    if (response.status === 429 && retryCount < MAX_RETRIES) {
      const delay = RETRY_DELAY * Math.pow(2, retryCount); // 指数退避
      console.warn(`YX翻译: API 限流，${delay}ms 后重试...`);
      await new Promise(resolve => setTimeout(resolve, delay));
      return translateSingle(text, retryCount + 1, targetLang);
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // 增强的响应验证
    if (data && Array.isArray(data[0])) {
      let translatedText = data[0]
        .filter(segment => segment && segment[0]) // 过滤空值
        .map(segment => segment[0])
        .join('');

      if (translatedText) {
        // 仅中文目标语言时执行术语校对
        if (targetLang.startsWith('zh')) {
          translatedText = await refineTranslation(text, translatedText);
        }
        return { original: text, translated: translatedText };
      }
    }
    return { original: text, translated: text };
  } catch (error) {
    console.warn(`YX翻译: Google 翻译失败 - ${error.message}，尝试备选翻译源...`);
    // Google 翻译失败时尝试 MyMemory
    const fallbackResult = await translateFallback(text, targetLang);
    if (fallbackResult) {
      let result = fallbackResult;
      if (targetLang.startsWith('zh')) {
        result = await refineTranslation(text, result);
      }
      return { original: text, translated: result };
    }
    return { original: text, translated: text };
  }
}

// 预编译术语替换表（只在首次调用时构建）
let compiledGlossary = null;

async function buildCompiledGlossary() {
  if (compiledGlossary) return compiledGlossary;

  // 构建：关键词 -> 替换规则映射
  const keywordMap = new Map();
  // 构建：错误译文 -> 正确译文 的直接映射（用于快速替换）
  const directReplacements = new Map();

  // 先加载用户自定义术语（优先级更高）
  let userGlossary = [];
  try {
    const data = await chrome.storage.sync.get(['user_glossary']);
    userGlossary = data.user_glossary || [];
  } catch (e) { /* 忽略错误 */ }

  // 将用户术语转为内置格式并合并（用户术语优先）
  for (const item of userGlossary) {
    if (item.keyword && item.badWord && item.goodWord) {
      const key = item.keyword.toLowerCase();
      if (!keywordMap.has(key)) {
        keywordMap.set(key, []);
      }
      // 插入到数组前端，确保用户术语优先
      keywordMap.get(key).unshift([item.badWord, item.goodWord]);
      if (!directReplacements.has(item.badWord)) {
        directReplacements.set(item.badWord, []);
      }
      directReplacements.get(item.badWord).unshift({ keyword: key, good: item.goodWord });
    }
  }

  // 加载内置术语
  for (const [keyword, replacements] of Object.entries(AI_GLOSSARY)) {
    if (!keywordMap.has(keyword)) {
      keywordMap.set(keyword, replacements);
    } else {
      // 用户已有同关键词，追加内置规则到后面
      keywordMap.get(keyword).push(...replacements);
    }
    for (const [bad, good] of replacements) {
      if (!directReplacements.has(bad)) {
        directReplacements.set(bad, []);
      }
      directReplacements.get(bad).push({ keyword, good });
    }
  }

  // 按错误译文长度降序排列（优先匹配长的）
  const sortedBadWords = Array.from(directReplacements.keys())
    .sort((a, b) => b.length - a.length);

  compiledGlossary = { keywordMap, directReplacements, sortedBadWords };
  return compiledGlossary;
}

// 监听用户术语变更，重置编译缓存
chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName === 'sync' && changes.user_glossary) {
    compiledGlossary = null;
  }
});

async function refineTranslation(source, target) {
  if (!target) return source;

  const { keywordMap, sortedBadWords, directReplacements } = await buildCompiledGlossary();
  const lowerSource = source.toLowerCase();

  // 找出原文中包含的关键词
  const matchedKeywords = new Set();
  for (const keyword of keywordMap.keys()) {
    if (lowerSource.includes(keyword)) {
      matchedKeywords.add(keyword);
    }
  }

  // 如果没有匹配的关键词，直接返回
  if (matchedKeywords.size === 0) return target;

  // 只替换与匹配关键词相关的错误译文
  let result = target;
  for (const badWord of sortedBadWords) {
    if (!result.includes(badWord)) continue;

    const replacementInfo = directReplacements.get(badWord);
    for (const { keyword, good } of replacementInfo) {
      if (matchedKeywords.has(keyword)) {
        result = result.split(badWord).join(good);
        break; // 一个错误译文只替换一次
      }
    }
  }

  return result;
}
