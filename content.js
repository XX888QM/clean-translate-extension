// Basic configuration
const TARGET_LANG = 'zh-CN';
const MIN_TEXT_LENGTH = 2;

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

// State management
let isTranslated = false;
let observer = null;
const originalTextMap = new WeakMap(); // Stores original text: Node -> String
const translationCache = new Map();    // Memory cache: String -> String

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
    if (!chrome.runtime?.id) return; // Safety check
    try {
        const storage = await chrome.storage.local.get(['translation_cache']);
        if (storage.translation_cache) {
            for (const [key, value] of Object.entries(storage.translation_cache)) {
                translationCache.set(key, value);
            }
        }
    } catch (e) {
        console.warn("Storage access failed (orphaned script?):", e);
    }
}

async function saveCache(newTranslations) {
    if (!chrome.runtime?.id) return;
    for (const [k, v] of Object.entries(newTranslations)) {
        translationCache.set(k, v);
    }
    try {
        chrome.storage.local.set({
            'translation_cache': Object.fromEntries(translationCache)
        });
    } catch (e) { }
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
        if (translationCache.has(text)) {
            applyTextToNode(node, translationCache.get(text));
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

    // Only show toast if we are doing a bulk translation (not just 1-2 dynamic nodes)
    if (textsToTranslate.length > 5) {
        showToast('YX翻译正在优化阅读体验...', 'loading');
    }

    const CHUNK_SIZE = 40;
    for (let i = 0; i < textsToTranslate.length; i += CHUNK_SIZE) {
        if (!chrome.runtime?.id) break; // Stop if invalid

        const chunk = textsToTranslate.slice(i, i + CHUNK_SIZE);
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TRANSLATE_TEXT_BATCH',
                texts: chunk
            });
            if (response && response.success) {
                saveCache(response.results);
                applyBatchTranslations(response.results, textNodeMap);
            }
        } catch (e) {
            console.warn('Translation/Message error:', e);
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
    observer = new MutationObserver((mutations) => {
        if (!isTranslated) return;
        let addedNodesList = [];
        mutations.forEach(m => {
            m.addedNodes.forEach(node => {
                if (node.nodeType === Node.ELEMENT_NODE) addedNodesList.push(node);
            });
        });
        if (addedNodesList.length > 0) {
            addedNodesList.forEach(node => performTranslation(node));
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
}

function checkAutoTranslate() {
    if (!chrome.runtime?.id) return;

    try {
        chrome.storage.local.get(['auto_translate_enabled', 'excluded_domains'], (result) => {
            if (chrome.runtime.lastError) return;

            // Check if domain is excluded
            const currentHost = window.location.hostname;
            const excludedList = result.excluded_domains || [];
            if (excludedList.includes(currentHost)) {
                console.log(`Clean Translator: Domain ${currentHost} is ignored by user preference.`);
                return;
            }

            // Default to TRUE if undefined (user hasn't toggled it yet)
            const isEnabled = result.auto_translate_enabled !== false;

            if (isEnabled) {
                const lang = document.documentElement.lang || '';
                const isChinese = lang.toLowerCase().includes('zh');
                if (!isChinese) {
                    console.log('Clean Translator: Auto-detecting non-Chinese content...');
                    // Show toast immediately
                    showToast('正在自动为您翻译...', 'loading');

                    performTranslation().then(() => {
                        if (chrome.runtime?.id) {
                            enableAutoTranslate();
                            showToast('翻译完成', 'success');
                            chrome.runtime.sendMessage({ type: 'TRANSLATION_DONE' }).catch(() => { });
                        }
                    });
                }
            }
        });
    } catch (e) {
        // Ignore
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', checkAutoTranslate);
} else {
    checkAutoTranslate();
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
    }
});
