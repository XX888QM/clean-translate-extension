// 基础配置
const TARGET_LANG = 'zh-CN';
const MIN_TEXT_LENGTH = 2;
const MAX_CACHE_SIZE = 10000; // 缓存最大条目数（约占用 3-5MB 存储空间）

// HTML 转义函数（防止 XSS）
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 划词翻译气泡样式
function injectSelectionPopupStyles() {
    if (document.getElementById('yx-selection-popup-style')) return;
    const style = document.createElement('style');
    style.id = 'yx-selection-popup-style';
    style.textContent = `
      #yx-selection-popup {
        position: absolute;
        z-index: 2147483647;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        padding: 10px 14px;
        max-width: 320px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 14px;
        line-height: 1.5;
        color: #333;
        opacity: 0;
        transform: translateY(8px);
        transition: opacity 0.2s, transform 0.2s;
        pointer-events: none;
      }
      #yx-selection-popup.show {
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
      }
      #yx-selection-popup .yx-popup-loading {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #666;
      }
      #yx-selection-popup .yx-popup-loading::before {
        content: '';
        width: 14px;
        height: 14px;
        border: 2px solid #e0e0e0;
        border-top-color: #1a73e8;
        border-radius: 50%;
        animation: yx-spin 0.8s linear infinite;
      }
      #yx-selection-popup .yx-popup-result {
        word-break: break-word;
      }
      #yx-selection-popup .yx-popup-original {
        font-size: 12px;
        color: #888;
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid #eee;
      }
    `;
    (document.head || document.documentElement).appendChild(style);
}

// 划词翻译气泡
let selectionPopup = null;
let selectionPopupTimeout = null;

function showSelectionPopup(text, x, y) {
    injectSelectionPopupStyles();

    if (!selectionPopup) {
        selectionPopup = document.createElement('div');
        selectionPopup.id = 'yx-selection-popup';
        document.body.appendChild(selectionPopup);
    }

    // 显示加载状态
    selectionPopup.innerHTML = '<div class="yx-popup-loading">翻译中...</div>';

    // 计算位置（避免超出屏幕）
    const popupWidth = 320;
    const viewportWidth = window.innerWidth;
    let posX = x;
    if (posX + popupWidth > viewportWidth - 20) {
        posX = viewportWidth - popupWidth - 20;
    }
    if (posX < 20) posX = 20;

    selectionPopup.style.left = posX + 'px';
    selectionPopup.style.top = (y + 10) + 'px';

    requestAnimationFrame(() => {
        selectionPopup.classList.add('show');
    });

    // 调用翻译
    translateSelectedText(text);
}

function hideSelectionPopup() {
    if (selectionPopup) {
        selectionPopup.classList.remove('show');
    }
}

async function translateSelectedText(text) {
    if (!chrome.runtime?.id) return;

    try {
        const response = await chrome.runtime.sendMessage({
            type: 'TRANSLATE_TEXT_BATCH',
            texts: [text]
        });

        if (response && response.success && selectionPopup) {
            const translated = response.results[text] || text;
            if (translated !== text) {
                selectionPopup.innerHTML = `
                    <div class="yx-popup-result">${escapeHtml(translated)}</div>
                    <div class="yx-popup-original">${escapeHtml(text)}</div>
                `;
            } else {
                selectionPopup.innerHTML = `<div class="yx-popup-result">${escapeHtml(translated)}</div>`;
            }
        }
    } catch (e) {
        if (selectionPopup) {
            selectionPopup.innerHTML = '<div class="yx-popup-result" style="color:#d93025;">翻译失败</div>';
        }
    }
}

// 监听选中事件
function initSelectionTranslate() {
    document.addEventListener('mouseup', (e) => {
        // 忽略点击在气泡上的情况
        if (selectionPopup && selectionPopup.contains(e.target)) return;

        if (selectionPopupTimeout) {
            clearTimeout(selectionPopupTimeout);
        }

        selectionPopupTimeout = setTimeout(() => {
            const selection = window.getSelection();
            const text = selection.toString().trim();

            if (text.length >= 2 && text.length <= 500) {
                // 检查是否为纯中文（已翻译过的内容跳过）
                if (/^[\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]+$/.test(text)) {
                    hideSelectionPopup();
                    return;
                }

                const range = selection.getRangeAt(0);
                const rect = range.getBoundingClientRect();
                showSelectionPopup(text, rect.left + window.scrollX, rect.bottom + window.scrollY);
            } else {
                hideSelectionPopup();
            }
        }, 200);
    });

    // 点击其他地方隐藏气泡
    document.addEventListener('mousedown', (e) => {
        if (selectionPopup && !selectionPopup.contains(e.target)) {
            hideSelectionPopup();
        }
    });

    // 滚动时隐藏气泡
    document.addEventListener('scroll', () => {
        hideSelectionPopup();
    }, true);
}

// Toast UI Implementation
function injectToastStyles() {
    if (document.getElementById('yx-clean-translate-toast-style')) return;
    const style = document.createElement('style');
    style.id = 'yx-clean-translate-toast-style';
    style.textContent = `
      #yx-toast-container {
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 2147483647;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        pointer-events: none;
        transition: opacity 0.3s ease, transform 0.3s ease;
        opacity: 0;
        transform: translateY(20px);
      }
      #yx-toast-container.show {
        opacity: 1;
        transform: translateY(0);
      }
      .yx-toast-message {
        background: rgba(32, 33, 36, 0.9);
        color: white;
        padding: 12px 18px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 13px;
        font-weight: 500;
        backdrop-filter: blur(4px);
        letter-spacing: 0.3px;
      }
      .yx-toast-icon {
        width: 14px;
        height: 14px;
        border: 2px solid rgba(255,255,255,0.3);
        border-top: 2px solid white;
        border-radius: 50%;
        animation: yx-spin 1s linear infinite;
      }
      .yx-toast-icon.success {
        border: none;
        animation: none;
        background: none;
        width: auto;
        height: auto;
        color: #81c995; 
        font-size: 16px;
      }
      @keyframes yx-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `;
    (document.head || document.documentElement).appendChild(style);
}

function showToast(message, type = 'loading') {
    if (!document.body) return;
    injectToastStyles();
    let container = document.getElementById('yx-toast-container');

    if (!container) {
        container = document.createElement('div');
        container.id = 'yx-toast-container';
        document.body.appendChild(container);
    }

    let iconHtml = '<div class="yx-toast-icon"></div>';
    if (type === 'success') {
        iconHtml = '<div class="yx-toast-icon success">✓</div>';
    } else if (type === 'restore') {
        iconHtml = '<div class="yx-toast-icon success">↺</div>';
    }

    container.innerHTML = `<div class="yx-toast-message">${iconHtml}<span>${message}</span></div>`;

    requestAnimationFrame(() => {
        container.classList.add('show');
    });

    if (type !== 'loading') {
        setTimeout(() => {
            container.classList.remove('show');
        }, 3000);
    }
}

// 状态管理
let isTranslated = false;
let isTranslating = false; // 翻译锁，防止并发执行
let observer = null;
let mutationDebounceTimer = null; // 防抖计时器
let cacheSaveTimer = null; // 缓存保存防抖计时器
let pendingCacheUpdates = {}; // 待保存的缓存更新
const originalTextMap = new WeakMap(); // 存储原文: Node -> String
const translationCache = new Map();    // 内存缓存: String -> String（使用 LRU 策略）
const MAX_PENDING_NODES = 100; // MutationObserver 最大待处理节点数

// LRU 缓存辅助函数：访问时移动到末尾
function cacheGet(key) {
    if (translationCache.has(key)) {
        const value = translationCache.get(key);
        // 移动到末尾以标记为最近使用
        translationCache.delete(key);
        translationCache.set(key, value);
        return value;
    }
    return undefined;
}

// LRU 缓存辅助函数：添加时检查大小限制
function cacheSet(key, value) {
    // 如果已存在，先删除（以便移动到末尾）
    if (translationCache.has(key)) {
        translationCache.delete(key);
    }
    translationCache.set(key, value);
    // 超出限制时删除最旧的条目
    if (translationCache.size > MAX_CACHE_SIZE) {
        const firstKey = translationCache.keys().next().value;
        translationCache.delete(firstKey);
    }
}

// Elements to ignore
const IGNORED_TAGS = new Set([
    'SCRIPT', 'STYLE', 'NOSCRIPT', 'TEXTAREA', 'INPUT', 'PRE', 'CODE',
    'KBD', 'SAMP', 'VAR', 'IFRAME', 'IMG', 'SVG', 'PATH', 'METADATA'
]);

function isTranslatable(node) {
    const parent = node.parentNode;
    if (!parent) return false;

    if (IGNORED_TAGS.has(parent.tagName)) return false;
    if (parent.isContentEditable) return false;

    // Check parent classes for icon indicators
    if (parent.className && typeof parent.className === 'string') {
        const cls = parent.className.toLowerCase();
        if (cls.includes('material-icons') || cls.includes('material-symbols') ||
            cls.includes('fa-') || cls.includes('icon') || cls.includes('glyph')) {
            return false;
        }
    }

    const text = node.nodeValue.trim();
    if (text.length < MIN_TEXT_LENGTH) return false;

    if (/^\d+$/.test(text)) return false;
    if (/^[^\p{L}]+$/u.test(text)) return false;
    // JSON-like or Caps-constants
    if (/^\{.*\}$/.test(text) || /^[A-Z0-9_]+$/.test(text)) return false;

    // Ignore snake_case strings often used for ligatures (e.g. keyboard_arrow_down)
    if (/^[a-z0-9]+(_[a-z0-9]+)+$/.test(text)) return false;

    return true;
}

function getTextNodes(root = document.body) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
    const nodes = [];
    let node;
    while (node = walker.nextNode()) {
        if (isTranslatable(node)) nodes.push(node);
    }
    return nodes;
}

async function loadCache() {
    if (!chrome.runtime?.id) return;
    try {
        const storage = await chrome.storage.local.get(['translation_cache']);
        if (storage.translation_cache) {
            const entries = Object.entries(storage.translation_cache);
            // 只加载最近的 MAX_CACHE_SIZE 条
            const toLoad = entries.slice(-MAX_CACHE_SIZE);
            for (const [key, value] of toLoad) {
                translationCache.set(key, value);
            }
        }
    } catch (e) {
        console.warn("缓存加载失败:", e);
    }
}

// 缓存保存（带防抖，合并多次调用）
function saveCache(newTranslations) {
    if (!chrome.runtime?.id) return;

    // 先更新内存缓存
    for (const [k, v] of Object.entries(newTranslations)) {
        cacheSet(k, v);
        pendingCacheUpdates[k] = v;
    }

    // 防抖：500ms 内的多次保存合并为一次
    if (cacheSaveTimer) {
        clearTimeout(cacheSaveTimer);
    }

    cacheSaveTimer = setTimeout(() => {
        if (!chrome.runtime?.id) return;
        try {
            // 只保存最近的 MAX_CACHE_SIZE 条
            const entries = Array.from(translationCache.entries()).slice(-MAX_CACHE_SIZE);
            chrome.storage.local.set({
                'translation_cache': Object.fromEntries(entries)
            });
            pendingCacheUpdates = {}; // 清空待保存队列
        } catch (e) {
            console.warn('YX翻译: 缓存保存失败', e);
        }
    }, 500);
}

// 清除缓存（供 popup 调用）
function clearCache() {
    translationCache.clear();
    if (chrome.runtime?.id) {
        chrome.storage.local.remove(['translation_cache']);
    }
}

function restoreOriginal() {
    showToast('正在还原原文...', 'loading');
    if (observer) observer.disconnect();
    observer = null;

    const nodes = getTextNodes();
    let count = 0;
    nodes.forEach(node => {
        if (originalTextMap.has(node)) {
            node.nodeValue = originalTextMap.get(node);
            count++;
        }
    });

    isTranslated = false;
    showToast('已还原原文', 'restore');
}

async function performTranslation(root = document.body) {
    // 防止并发执行（但允许子树翻译）
    if (isTranslating && root === document.body) {
        console.log('YX翻译: 翻译正在进行中，跳过重复请求');
        return;
    }

    if (root === document.body) {
        isTranslating = true;
    }

    try {
        await _doTranslation(root);
    } finally {
        if (root === document.body) {
            isTranslating = false;
        }
    }
}

async function _doTranslation(root = document.body) {
    await loadCache();

    const nodes = getTextNodes(root);
    if (nodes.length === 0) return;

    const textNodeMap = new Map();
    const missingTranslations = new Set();

    nodes.forEach(node => {
        if (!originalTextMap.has(node)) {
            originalTextMap.set(node, node.nodeValue);
        }
        const text = node.nodeValue.trim();
        const cached = cacheGet(text);
        if (cached !== undefined) {
            applyTextToNode(node, cached);
        } else {
            if (!textNodeMap.has(text)) textNodeMap.set(text, []);
            textNodeMap.get(text).push(node);
            missingTranslations.add(text);
        }
    });

    const textsToTranslate = Array.from(missingTranslations);
    if (textsToTranslate.length === 0) {
        isTranslated = true;
        return;
    }

    const totalCount = textsToTranslate.length;
    const showProgress = totalCount > 5;

    // 批量翻译时显示进度
    if (showProgress) {
        showToast(`翻译中 0/${totalCount}...`, 'loading');
    }

    // 增大 CHUNK_SIZE 以配合后台的 BATCH_SIZE
    const CHUNK_SIZE = 60;
    let translatedCount = 0;
    let errorCount = 0;
    const MAX_CONSECUTIVE_ERRORS = 3; // 连续错误阈值

    for (let i = 0; i < textsToTranslate.length; i += CHUNK_SIZE) {
        if (!chrome.runtime?.id) break;

        const chunk = textsToTranslate.slice(i, i + CHUNK_SIZE);
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TRANSLATE_TEXT_BATCH',
                texts: chunk
            });
            if (response && response.success) {
                saveCache(response.results);
                applyBatchTranslations(response.results, textNodeMap);
                translatedCount += chunk.length;
                errorCount = 0; // 重置错误计数

                // 更新进度显示
                if (showProgress) {
                    const percent = Math.round((translatedCount / totalCount) * 100);
                    showToast(`翻译中 ${translatedCount}/${totalCount} (${percent}%)`, 'loading');
                }
            } else if (response && !response.success) {
                errorCount++;
                console.warn('YX翻译: 批次翻译失败', response.error);
            }
        } catch (e) {
            errorCount++;
            console.warn('YX翻译: 翻译消息错误:', e);

            // 连续错误过多时提示用户
            if (errorCount >= MAX_CONSECUTIVE_ERRORS) {
                showToast('翻译遇到问题，部分内容可能未翻译', 'success');
                break;
            }
        }
    }
    isTranslated = true;
}

function applyTextToNode(node, translatedText) {
    const current = node.nodeValue;
    if (!current) return;
    const match = current.match(/^(\s*)([\s\S]*?)(\s*)$/);
    if (match) {
        const [_, prefix, content, suffix] = match;
        node.nodeValue = prefix + translatedText + suffix;
    } else {
        node.nodeValue = translatedText;
    }
}

function applyBatchTranslations(results, textNodeMap) {
    for (const [original, translated] of Object.entries(results)) {
        if (original === translated) continue;
        const nodes = textNodeMap.get(original);
        if (nodes) {
            nodes.forEach(node => applyTextToNode(node, translated));
        }
    }
}

function enableAutoTranslate() {
    if (observer) return;
    let pendingNodes = new Set(); // 待翻译节点去重

    observer = new MutationObserver((mutations) => {
        if (!isTranslated) return;

        mutations.forEach(m => {
            m.addedNodes.forEach(node => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    // 限制待处理节点数量，防止内存积压
                    if (pendingNodes.size < MAX_PENDING_NODES) {
                        pendingNodes.add(node);
                    }
                }
            });
        });

        // 防抖：200ms 内的变化合并处理
        if (mutationDebounceTimer) {
            clearTimeout(mutationDebounceTimer);
        }

        mutationDebounceTimer = setTimeout(() => {
            if (pendingNodes.size > 0) {
                const nodesToTranslate = Array.from(pendingNodes);
                pendingNodes.clear();

                // 过滤出仍在 DOM 中的节点
                const validNodes = nodesToTranslate.filter(node => document.body.contains(node));

                // 如果节点过多，只处理前 MAX_PENDING_NODES 个
                const toProcess = validNodes.slice(0, MAX_PENDING_NODES);

                // 批量处理，避免逐个调用
                toProcess.forEach(node => {
                    performTranslation(node);
                });
            }
        }, 200);
    });
    observer.observe(document.body, { childList: true, subtree: true });
}

// 检测页面内容是否主要为中文
function detectPageLanguage() {
    // 1. 先检查 html lang 属性
    const htmlLang = document.documentElement.lang || '';
    if (htmlLang.toLowerCase().includes('zh')) {
        return 'zh';
    }

    // 2. 抽样检测页面实际内容
    const sampleTexts = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    let node;
    let count = 0;

    while ((node = walker.nextNode()) && count < 50) {
        const text = node.nodeValue.trim();
        if (text.length >= 10) {
            sampleTexts.push(text);
            count++;
        }
    }

    if (sampleTexts.length === 0) return 'unknown';

    // 统计中文字符占比
    const allText = sampleTexts.join('');
    const chineseChars = allText.match(/[\u4e00-\u9fa5]/g) || [];
    const ratio = chineseChars.length / allText.length;

    // 中文字符超过 30% 视为中文页面
    return ratio > 0.3 ? 'zh' : 'other';
}

function checkAutoTranslate() {
    if (!chrome.runtime?.id) return;

    try {
        chrome.storage.local.get(['auto_translate_enabled', 'excluded_domains'], (result) => {
            if (chrome.runtime.lastError) return;

            // 检查域名是否被排除
            const currentHost = window.location.hostname;
            const excludedList = result.excluded_domains || [];
            if (excludedList.includes(currentHost)) {
                console.log(`YX翻译: 域名 ${currentHost} 已被用户排除`);
                return;
            }

            // 默认启用自动翻译
            const isEnabled = result.auto_translate_enabled !== false;

            if (isEnabled) {
                const detectedLang = detectPageLanguage();

                if (detectedLang !== 'zh') {
                    console.log('YX翻译: 检测到非中文内容，开始翻译...');
                    showToast('正在自动为您翻译...', 'loading');

                    performTranslation().then(() => {
                        if (chrome.runtime?.id) {
                            enableAutoTranslate();
                            showToast('翻译完成', 'success');
                            chrome.runtime.sendMessage({ type: 'TRANSLATION_DONE' }).catch(() => { });
                        }
                    });
                } else {
                    console.log('YX翻译: 检测到中文页面，跳过翻译');
                }
            }
        });
    } catch (e) {
        // 忽略错误
    }
}

// 初始化
function init() {
    checkAutoTranslate();
    initSelectionTranslate(); // 启用划词翻译
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (!chrome.runtime?.id) return;

    if (request.type === 'START_TRANSLATE') {
        showToast('开始分析页面...', 'loading');
        performTranslation().then(() => {
            if (chrome.runtime?.id) {
                enableAutoTranslate();
                showToast('翻译完成', 'success');
                chrome.runtime.sendMessage({ type: 'TRANSLATION_DONE' }).catch(() => { });
            }
        });
        sendResponse({ status: 'started' });
    } else if (request.type === 'RESTORE_ORIGINAL') {
        restoreOriginal();
        sendResponse({ status: 'restored' });
    } else if (request.type === 'TRANSLATE_SELECTION') {
        // 右键菜单触发的选中文本翻译
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const rect = range.getBoundingClientRect();
            showSelectionPopup(request.text, rect.left + window.scrollX, rect.bottom + window.scrollY);
        }
        sendResponse({ status: 'ok' });
    } else if (request.type === 'GET_CACHE_INFO') {
        // 获取缓存信息
        sendResponse({ size: translationCache.size });
    } else if (request.type === 'CLEAR_CACHE') {
        clearCache();
        sendResponse({ status: 'cleared' });
    }
});
