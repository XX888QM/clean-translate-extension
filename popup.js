function sendMessageToTab(message) {
    const statusDiv = document.getElementById('status');
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        // 检查是否有有效的标签页
        if (tabs.length === 0) {
            statusDiv.textContent = "无法获取当前标签页";
            statusDiv.style.color = "#d93025";
            return;
        }

        const tab = tabs[0];

        // 检查 Tab ID 是否有效
        if (!tab.id || tab.id === chrome.tabs.TAB_ID_NONE) {
            statusDiv.textContent = "当前页面不支持";
            statusDiv.style.color = "#d93025";
            return;
        }

        // 检查是否为特殊页面（chrome://、edge:// 等）
        if (tab.url && (tab.url.startsWith('chrome://') ||
            tab.url.startsWith('chrome-extension://') ||
            tab.url.startsWith('edge://') ||
            tab.url.startsWith('about:'))) {
            statusDiv.textContent = "系统页面不支持翻译";
            statusDiv.style.color = "#d93025";
            return;
        }

        chrome.tabs.sendMessage(tab.id, message, (response) => {
            if (chrome.runtime.lastError) {
                console.warn('YX翻译: 消息发送失败', chrome.runtime.lastError.message);
                statusDiv.textContent = "当前页面不支持";
                statusDiv.style.color = "#d93025";
            } else {
                if (message.type === 'START_TRANSLATE') {
                    statusDiv.textContent = "翻译中，请稍候...";
                    statusDiv.style.color = "#188038";
                } else if (message.type === 'RESTORE_ORIGINAL') {
                    statusDiv.textContent = "已还原原文";
                    statusDiv.style.color = "#5f6368";
                }
            }
        });
    });
}

// 翻译/还原按钮（手动翻译时记录网站偏好）
document.getElementById('translateBtn').addEventListener('click', () => {
    sendMessageToTab({ type: 'START_TRANSLATE', recordPreference: true });
});

document.getElementById('restoreBtn').addEventListener('click', () => {
    sendMessageToTab({ type: 'RESTORE_ORIGINAL' });
});

// ========== 目标语言选择 ==========
const targetLangSelect = document.getElementById('targetLangSelect');

chrome.storage.local.get(['target_lang'], (result) => {
    if (chrome.runtime.lastError) return;
    if (result.target_lang) {
        targetLangSelect.value = result.target_lang;
    }
});

targetLangSelect.addEventListener('change', () => {
    chrome.storage.local.set({ target_lang: targetLangSelect.value });
});

// ========== 翻译模式 + 网站偏好 ==========
const translateModeSelect = document.getElementById('translateModeSelect');
const sitePrefStatus = document.getElementById('sitePreferenceStatus');
const sitePrefAutoBtn = document.getElementById('sitePrefAutoBtn');
const sitePrefNeverBtn = document.getElementById('sitePrefNeverBtn');
const sitePrefClearBtn = document.getElementById('sitePrefClearBtn');
const bilingualToggle = document.getElementById('bilingualToggle');
let currentHostname = '';

// 更新网站偏好 UI 状态
function updateSitePrefUI(pref) {
    sitePrefStatus.textContent = pref === 'auto' ? '自动翻译'
        : pref === 'never' ? '从不翻译' : '跟随全局';
    sitePrefStatus.className = 'site-pref-status' + (pref ? ' ' + pref : '');
    sitePrefAutoBtn.classList.toggle('active', pref === 'auto');
    sitePrefNeverBtn.classList.toggle('active', pref === 'never');
}

// 保存网站偏好
function saveSitePreference(pref) {
    if (!currentHostname) return;
    chrome.storage.local.get(['site_preferences'], (result) => {
        if (chrome.runtime.lastError) return;
        const prefs = { ...(result.site_preferences || {}) };
        if (pref) {
            prefs[currentHostname] = pref;
        } else {
            delete prefs[currentHostname];
        }
        chrome.storage.local.set({ site_preferences: prefs });
        updateSitePrefUI(pref);
    });
}

// 加载当前标签页信息和设置
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs.length === 0) return;
    try {
        const url = new URL(tabs[0].url);
        currentHostname = url.hostname;

        chrome.storage.local.get([
            'translate_mode', 'auto_translate_enabled',
            'site_preferences', 'bilingual_mode'
        ], (result) => {
            if (chrome.runtime.lastError) return;

            // 翻译模式（兼容旧版）
            if (result.translate_mode) {
                translateModeSelect.value = result.translate_mode;
            } else {
                translateModeSelect.value = result.auto_translate_enabled === false ? 'manual' : 'auto_all';
            }

            // 网站偏好
            const sitePrefs = result.site_preferences || {};
            updateSitePrefUI(sitePrefs[currentHostname] || null);

            // 双语模式
            bilingualToggle.checked = result.bilingual_mode === true;
        });
    } catch (e) {
        // 特殊页面无法解析 URL
        sitePrefAutoBtn.disabled = true;
        sitePrefNeverBtn.disabled = true;
        sitePrefClearBtn.disabled = true;
    }
});

// 翻译模式切换
translateModeSelect.addEventListener('change', () => {
    chrome.storage.local.set({ translate_mode: translateModeSelect.value });
});

// 网站偏好按钮
sitePrefAutoBtn.addEventListener('click', () => saveSitePreference('auto'));
sitePrefNeverBtn.addEventListener('click', () => saveSitePreference('never'));
sitePrefClearBtn.addEventListener('click', () => saveSitePreference(null));

// 双语对照开关
bilingualToggle.addEventListener('change', () => {
    chrome.storage.local.set({ bilingual_mode: bilingualToggle.checked });
});

// ========== 翻译统计 ==========
function updateStats() {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs.length === 0 || !tabs[0].id) return;
        chrome.tabs.sendMessage(tabs[0].id, { type: 'GET_TRANSLATE_STATS' }, (response) => {
            if (chrome.runtime.lastError || !response) return;
            document.getElementById('statTotal').textContent = (response.totalTranslated || 0) + ' 条';
            document.getElementById('statCache').textContent = (response.cacheHits || 0) + ' 次';
            document.getElementById('statApi').textContent = (response.apiCalls || 0) + ' 次';
        });
    });
}
updateStats();

// ========== 缓存管理（通过 background IndexedDB） ==========
const cacheSizeEl = document.getElementById('cacheSize');
const clearCacheBtn = document.getElementById('clearCacheBtn');

function updateCacheSize() {
    chrome.runtime.sendMessage({ type: 'CACHE_COUNT' }, (response) => {
        if (chrome.runtime.lastError || !response?.success) {
            if (cacheSizeEl) cacheSizeEl.textContent = '读取失败';
            return;
        }
        if (cacheSizeEl) cacheSizeEl.textContent = `${response.count} 条`;
    });
}
updateCacheSize();

// 清除缓存
if (clearCacheBtn) {
    clearCacheBtn.addEventListener('click', () => {
        chrome.runtime.sendMessage({ type: 'CACHE_CLEAR' }, (response) => {
            if (chrome.runtime.lastError || !response?.success) {
                clearCacheBtn.textContent = '失败';
                setTimeout(() => { clearCacheBtn.textContent = '清除'; }, 2000);
                return;
            }
            if (cacheSizeEl) cacheSizeEl.textContent = '0 条';
            clearCacheBtn.textContent = '已清除';
            setTimeout(() => { clearCacheBtn.textContent = '清除'; }, 2000);
        });
    });
}

// 导出缓存
document.getElementById('exportCacheBtn').addEventListener('click', () => {
    chrome.runtime.sendMessage({ type: 'CACHE_GET_ALL' }, (response) => {
        if (chrome.runtime.lastError || !response?.success) return;
        const cache = response.results || {};
        const blob = new Blob([JSON.stringify(cache, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'yx-translate-cache.json';
        a.click();
        URL.revokeObjectURL(url);
    });
});

// 导入缓存
document.getElementById('importCacheBtnTrigger').addEventListener('click', () => {
    document.getElementById('importCacheFile').click();
});

document.getElementById('importCacheFile').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
        try {
            const imported = JSON.parse(evt.target.result);
            if (typeof imported !== 'object' || Array.isArray(imported)) {
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = '无效的缓存文件格式';
                statusDiv.style.color = '#d93025';
                return;
            }
            chrome.runtime.sendMessage({ type: 'CACHE_PUT_BATCH', entries: imported }, (response) => {
                if (chrome.runtime.lastError || !response?.success) return;
                updateCacheSize();
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = `已导入 ${Object.keys(imported).length} 条缓存`;
                statusDiv.style.color = '#188038';
                setTimeout(() => {
                    statusDiv.textContent = '准备就绪';
                    statusDiv.style.color = '#5f6368';
                }, 3000);
            });
        } catch (err) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = '缓存文件解析失败';
            statusDiv.style.color = '#d93025';
        }
    };
    reader.readAsText(file);
    // 重置文件输入，允许重复选择同一文件
    e.target.value = '';
});

// ========== 术语管理 ==========
const glossaryToggle = document.getElementById('glossaryToggle');
const glossaryArrow = document.getElementById('glossaryArrow');
const glossaryPanel = document.getElementById('glossaryPanel');
const glossaryList = document.getElementById('glossaryList');

glossaryToggle.addEventListener('click', () => {
    glossaryPanel.classList.toggle('show');
    glossaryArrow.classList.toggle('open');
    if (glossaryPanel.classList.contains('show')) {
        loadGlossary();
    }
});

function loadGlossary() {
    chrome.storage.sync.get(['user_glossary'], (result) => {
        if (chrome.runtime.lastError) return;
        const list = result.user_glossary || [];
        renderGlossaryList(list);
    });
}

function renderGlossaryList(list) {
    glossaryList.innerHTML = '';
    if (list.length === 0) {
        glossaryList.textContent = '暂无自定义术语';
        return;
    }
    list.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'glossary-item';
        const span = document.createElement('span');
        span.textContent = `${item.keyword}: ${item.badWord} → ${item.goodWord}`;
        const delBtn = document.createElement('button');
        delBtn.className = 'del-btn';
        delBtn.textContent = '\u2715';
        delBtn.addEventListener('click', () => deleteGlossaryItem(index));
        div.appendChild(span);
        div.appendChild(delBtn);
        glossaryList.appendChild(div);
    });
}

document.getElementById('glossaryAddBtn').addEventListener('click', () => {
    const keyword = document.getElementById('glossaryKeyword').value.trim();
    const badWord = document.getElementById('glossaryBad').value.trim();
    const goodWord = document.getElementById('glossaryGood').value.trim();
    if (!keyword || !badWord || !goodWord) return;

    chrome.storage.sync.get(['user_glossary'], (result) => {
        if (chrome.runtime.lastError) return;
        const list = result.user_glossary || [];
        list.push({ keyword, badWord, goodWord });
        chrome.storage.sync.set({ user_glossary: list }, () => {
            document.getElementById('glossaryKeyword').value = '';
            document.getElementById('glossaryBad').value = '';
            document.getElementById('glossaryGood').value = '';
            renderGlossaryList(list);
        });
    });
});

function deleteGlossaryItem(index) {
    chrome.storage.sync.get(['user_glossary'], (result) => {
        if (chrome.runtime.lastError) return;
        const list = result.user_glossary || [];
        list.splice(index, 1);
        chrome.storage.sync.set({ user_glossary: list }, () => {
            renderGlossaryList(list);
        });
    });
}

// ========== 翻译引擎配置 ==========
const engineSelect = document.getElementById('engineSelect');
const apiKeyPanel = document.getElementById('apiKeyPanel');
const apiKeyInput = document.getElementById('apiKeyInput');
const apiKeySaveBtn = document.getElementById('apiKeySaveBtn');
const baiduAppIdRow = document.getElementById('baiduAppIdRow');
const baiduAppIdInput = document.getElementById('baiduAppIdInput');
const apiKeyHint = document.getElementById('apiKeyHint');

// 各引擎对应的提示文字
const ENGINE_HINTS = {
  google_cloud: '请到 console.cloud.google.com 获取 API Key',
  deepl: '请到 deepl.com/pro-api 获取 API Key',
  baidu: '请到 fanyi-api.baidu.com 获取 APP ID 和密钥',
  openai: '请到 platform.openai.com 获取 API Key',
  claude: '请到 console.anthropic.com 获取 API Key',
  deepseek: '请到 platform.deepseek.com 获取 API Key',
  minimax: '请到 platform.minimaxi.com 获取 API Key',
  glm: '请到 open.bigmodel.cn 获取 API Key'
};

// 各引擎在 api_keys 中的存储键名
const ENGINE_KEY_MAP = {
  google_cloud: 'google_cloud',
  deepl: 'deepl',
  baidu: 'baidu_key',
  openai: 'openai',
  claude: 'claude',
  deepseek: 'deepseek',
  minimax: 'minimax',
  glm: 'glm'
};

// 更新 API Key 面板显示状态
function updateApiKeyPanel(engine) {
  if (engine === 'google_free') {
    apiKeyPanel.style.display = 'none';
    return;
  }
  apiKeyPanel.style.display = 'block';
  // 百度翻译需要额外的 APP ID
  baiduAppIdRow.style.display = engine === 'baidu' ? 'flex' : 'none';
  // 更新提示文字
  apiKeyHint.textContent = ENGINE_HINTS[engine] || '';
  // 从 storage 加载已保存的 key
  chrome.storage.local.get(['api_keys'], (result) => {
    if (chrome.runtime.lastError) return;
    const keys = result.api_keys || {};
    const keyName = ENGINE_KEY_MAP[engine];
    apiKeyInput.value = keys[keyName] || '';
    if (engine === 'baidu') {
      baiduAppIdInput.value = keys.baidu_appid || '';
    }
  });
}

// 初始化：读取已保存的引擎选择
chrome.storage.local.get(['translate_engine'], (result) => {
  if (chrome.runtime.lastError) return;
  const engine = result.translate_engine || 'google_free';
  engineSelect.value = engine;
  updateApiKeyPanel(engine);
});

// 切换引擎
engineSelect.addEventListener('change', () => {
  const engine = engineSelect.value;
  chrome.storage.local.set({ translate_engine: engine });
  updateApiKeyPanel(engine);
});

// 保存 API Key
apiKeySaveBtn.addEventListener('click', () => {
  const engine = engineSelect.value;
  if (engine === 'google_free') return;

  const keyName = ENGINE_KEY_MAP[engine];
  const keyValue = apiKeyInput.value.trim();

  chrome.storage.local.get(['api_keys'], (result) => {
    if (chrome.runtime.lastError) return;
    const keys = result.api_keys || {};
    keys[keyName] = keyValue;
    // 百度翻译同时保存 APP ID
    if (engine === 'baidu') {
      keys.baidu_appid = baiduAppIdInput.value.trim();
    }
    chrome.storage.local.set({ api_keys: keys }, () => {
      if (chrome.runtime.lastError) {
        apiKeySaveBtn.textContent = '失败';
        setTimeout(() => { apiKeySaveBtn.textContent = '保存'; }, 2000);
        return;
      }
      apiKeySaveBtn.textContent = '已保存';
      setTimeout(() => { apiKeySaveBtn.textContent = '保存'; }, 2000);
    });
  });
});

// ========== 翻译完成消息监听 ==========
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === 'TRANSLATION_DONE') {
        const statusDiv = document.getElementById('status');
        if (statusDiv) {
            statusDiv.textContent = "翻译完成";
            statusDiv.style.color = "#188038";
            setTimeout(() => {
                statusDiv.textContent = "准备就绪";
                statusDiv.style.color = "#5f6368";
            }, 3000);
        }
        updateCacheSize();
        updateStats();
    }
});
