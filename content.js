// 测试环境标记：Node require 时为真，浏览器中 module 未定义为假（不影响扩展运行）
const __TEST__ = (typeof module !== 'undefined' && module.exports);

// 基础配置
const MIN_TEXT_LENGTH = 2;
const MAX_CACHE_SIZE = 10000; // 缓存最大条目数（约占用 3-5MB 存储空间）
const MAX_TRANSLATION_TEXT_CHARS = 1500;
const MAX_TRANSLATION_BATCH_ITEMS = 200;
const MAX_TRANSLATION_BATCH_CHARS = 20000;
const DYNAMIC_TRANSLATION_BUDGET_WINDOW_MS = 60 * 1000;
const MAX_DYNAMIC_TRANSLATION_CHARS_PER_WINDOW = 50000;

// 可翻译的 HTML 属性列表
const TRANSLATABLE_ATTRS = ['placeholder', 'title', 'alt', 'aria-label'];

// 共享动画样式（只注入一次）
function injectSharedStyles() {
    if (document.getElementById('yx-shared-style')) return;
    const style = document.createElement('style');
    style.id = 'yx-shared-style';
    style.textContent = `
      @keyframes yx-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `;
    (document.head || document.documentElement).appendChild(style);
}

// ===== 功能1: 翻译进度条 =====
// 注入进度条样式
function injectProgressBarStyles() {
    if (document.getElementById('yx-progress-bar-style')) return;
    const style = document.createElement('style');
    style.id = 'yx-progress-bar-style';
    style.textContent = `
      #yx-progress-bar {
        position: fixed;
        top: 0;
        left: 0;
        width: 0%;
        height: 3px;
        background: linear-gradient(90deg, #1a73e8, #4fc3f7);
        z-index: 2147483647;
        transition: width 0.3s ease, opacity 0.4s ease;
        opacity: 1;
        pointer-events: none;
        box-shadow: 0 0 6px rgba(26, 115, 232, 0.5);
      }
      #yx-progress-bar.yx-progress-done {
        opacity: 0;
      }
    `;
    (document.head || document.documentElement).appendChild(style);
}

// 显示进度条
function showProgressBar() {
    injectProgressBarStyles();
    let bar = document.getElementById('yx-progress-bar');
    if (!bar) {
        bar = document.createElement('div');
        bar.id = 'yx-progress-bar';
        document.body.appendChild(bar);
    }
    bar.classList.remove('yx-progress-done');
    bar.style.width = '0%';
}

// 更新进度条百分比
function updateProgress(percent) {
    const bar = document.getElementById('yx-progress-bar');
    if (!bar) return;
    // 限制最大到 99%，完成时由 hideProgressBar 设为 100%
    bar.style.width = Math.min(percent, 99) + '%';
}

// 隐藏进度条（先到 100% 再渐隐）
function hideProgressBar() {
    const bar = document.getElementById('yx-progress-bar');
    if (!bar) return;
    bar.style.width = '100%';
    setTimeout(() => {
        bar.classList.add('yx-progress-done');
        // 渐隐动画完成后重置
        setTimeout(() => {
            bar.style.width = '0%';
            bar.classList.remove('yx-progress-done');
        }, 500);
    }, 300);
}

// ===== 功能3: 鼠标悬停显示原文 =====
let hoverTooltip = null;
let hoverTimer = null;

// 注入悬停提示样式
function injectHoverTooltipStyles() {
    if (document.getElementById('yx-hover-tooltip-style')) return;
    const style = document.createElement('style');
    style.id = 'yx-hover-tooltip-style';
    style.textContent = `
      #yx-hover-tooltip {
        position: absolute;
        z-index: 2147483647;
        background: #f8f9fa;
        border: 1px solid #dadce0;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        padding: 6px 10px;
        max-width: 280px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 12px;
        line-height: 1.4;
        color: #5f6368;
        opacity: 0;
        transform: translateY(4px);
        transition: opacity 0.15s ease, transform 0.15s ease;
        pointer-events: none;
        word-break: break-word;
      }
      #yx-hover-tooltip.show {
        opacity: 1;
        transform: translateY(0);
      }
      @media (prefers-color-scheme: dark) {
        #yx-hover-tooltip {
          background: #2d2e31;
          border-color: #5f6368;
          color: #e8eaed;
        }
      }
    `;
    (document.head || document.documentElement).appendChild(style);
}

// 显示原文提示浮层
function showOriginalTooltip(text, x, y) {
    injectHoverTooltipStyles();

    if (!hoverTooltip) {
        hoverTooltip = document.createElement('div');
        hoverTooltip.id = 'yx-hover-tooltip';
        document.body.appendChild(hoverTooltip);
    }

    hoverTooltip.textContent = text;

    // 计算位置，避免超出视口
    const tooltipWidth = 280;
    const viewportWidth = window.innerWidth;
    let posX = x;
    if (posX + tooltipWidth > viewportWidth - 10) {
        posX = viewportWidth - tooltipWidth - 10;
    }
    if (posX < 10) posX = 10;

    // 向上偏移，显示在鼠标上方
    let posY = y - 30;
    if (posY < 10) posY = y + 20; // 上方空间不足则显示在下方

    hoverTooltip.style.left = posX + 'px';
    hoverTooltip.style.top = posY + 'px';

    requestAnimationFrame(() => {
        hoverTooltip.classList.add('show');
    });
}

// 隐藏原文提示浮层
function hideOriginalTooltip() {
    if (hoverTimer) {
        clearTimeout(hoverTimer);
        hoverTimer = null;
    }
    if (hoverTooltip) {
        hoverTooltip.classList.remove('show');
    }
}

// 初始化鼠标悬停显示原文功能（事件委托）
function initHoverOriginal() {
    document.body.addEventListener('mouseenter', (e) => {
        // 只在已翻译状态下启用
        if (!isTranslated) return;

        const target = e.target;
        if (!target || target.nodeType !== Node.ELEMENT_NODE) return;

        // 忽略自身组件
        if (target.id && target.id.startsWith('yx-')) return;

        // 查找该元素下是否有被翻译的文本节点
        const walker = document.createTreeWalker(target, NodeFilter.SHOW_TEXT);
        let textNode = walker.nextNode();
        let originalText = null;

        while (textNode) {
            if (originalTextMap.has(textNode)) {
                const orig = originalTextMap.get(textNode);
                const current = textNode.nodeValue.trim();
                // 只有当前值和原文不同时才显示（说明已翻译）
                if (orig.trim() !== current) {
                    originalText = orig.trim();
                    break;
                }
            }
            textNode = walker.nextNode();
        }

        if (!originalText) return;

        // 延迟 1 秒后显示
        if (hoverTimer) clearTimeout(hoverTimer);
        const mouseX = e.clientX + window.scrollX;
        const mouseY = e.clientY + window.scrollY;
        hoverTimer = setTimeout(() => {
            showOriginalTooltip(originalText, mouseX, mouseY);
        }, 1000);
    }, true); // 捕获阶段，实现事件委托

    document.body.addEventListener('mouseleave', (e) => {
        const target = e.target;
        if (!target || target.nodeType !== Node.ELEMENT_NODE) return;
        hideOriginalTooltip();
    }, true);

    // 滚动时隐藏提示
    document.addEventListener('scroll', () => {
        hideOriginalTooltip();
    }, true);
}

// ===== 功能2: 视口内按需翻译 (IntersectionObserver) =====
let viewportObserver = null;

// 创建视口翻译观察器
function createViewportObserver() {
    if (viewportObserver) return viewportObserver;
    if (typeof IntersectionObserver === 'undefined') return null;

    viewportObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                // 进入视口时翻译该元素
                viewportObserver.unobserve(el);
                performTranslation(el, true);
            }
        });
    }, {
        // 提前 200px 开始翻译，提升用户感知速度
        rootMargin: '200px 0px'
    });

    return viewportObserver;
}

// 对新增节点使用按需翻译（只翻译进入视口的块级元素）
function performViewportTranslation(nodes) {
    const obs = createViewportObserver();
    if (!obs) {
        // 不支持 IntersectionObserver 时回退到直接翻译
        nodes.forEach(node => performTranslation(node, true));
        return;
    }

    nodes.forEach(node => {
        if (node.nodeType !== Node.ELEMENT_NODE) return;

        // 对块级容器元素直接观察
        const display = window.getComputedStyle(node).display;
        const isBlock = display === 'block' || display === 'flex' ||
                        display === 'grid' || display === 'list-item' ||
                        display === 'table' || display === 'table-row' ||
                        display === 'table-cell';

        if (isBlock) {
            obs.observe(node);
        } else {
            // 行内元素直接翻译（通常体积小）
            performTranslation(node, true);
        }
    });
}

// 划词翻译气泡样式
function injectSelectionPopupStyles() {
    if (document.getElementById('yx-selection-popup-style')) return;
    injectSharedStyles();
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
        max-width: 400px;
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
      #yx-selection-popup .yx-popup-engine-label {
        font-size: 11px;
        color: #1a73e8;
        font-weight: 500;
        margin-bottom: 2px;
      }
      #yx-selection-popup .yx-popup-divider {
        border-top: 1px solid #eee;
        margin: 8px 0;
      }
      #yx-selection-popup .yx-popup-original {
        font-size: 12px;
        color: #888;
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid #eee;
      }
      @media (prefers-color-scheme: dark) {
        #yx-selection-popup {
          background: #292a2d;
          color: #e8eaed;
          box-shadow: 0 4px 16px rgba(0,0,0,0.5);
        }
        #yx-selection-popup .yx-popup-loading { color: #9aa0a6; }
        #yx-selection-popup .yx-popup-divider { border-top-color: #3c4043; }
        #yx-selection-popup .yx-popup-original {
          color: #9aa0a6;
          border-top-color: #3c4043;
        }
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

    // 显示加载状态（DOM API 避免 XSS）
    selectionPopup.textContent = '';
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'yx-popup-loading';
    loadingDiv.textContent = '翻译中...';
    selectionPopup.appendChild(loadingDiv);

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

// 渲染单个引擎翻译结果
function appendEngineResult(container, engineName, translatedText, error) {
    const wrapper = document.createElement('div');
    wrapper.className = 'yx-popup-result';
    const label = document.createElement('div');
    label.className = 'yx-popup-engine-label';
    label.textContent = engineName;
    wrapper.appendChild(label);
    const content = document.createElement('div');
    if (error) {
        content.style.color = '#d93025';
        content.textContent = '翻译失败: ' + error;
    } else {
        content.textContent = translatedText;
    }
    wrapper.appendChild(content);
    container.appendChild(wrapper);
}

async function translateSelectedText(text) {
    if (!chrome.runtime?.id) return;

    try {
        // 使用引擎对比翻译
        const response = await sendMessageWithRetry({
            type: 'TRANSLATE_COMPARE',
            text: text
        });

        if (response?.success && selectionPopup) {
            selectionPopup.textContent = '';
            const { primary, secondary } = response.results;

            // 主引擎结果
            appendEngineResult(selectionPopup, primary.engine, primary.text, primary.error);

            // 对比引擎结果（当前引擎非 google_free 时存在）
            if (secondary) {
                const divider = document.createElement('div');
                divider.className = 'yx-popup-divider';
                selectionPopup.appendChild(divider);
                appendEngineResult(selectionPopup, secondary.engine, secondary.text, secondary.error);
            }

            // 原文
            if (primary.text !== text) {
                const origDiv = document.createElement('div');
                origDiv.className = 'yx-popup-original';
                origDiv.textContent = text;
                selectionPopup.appendChild(origDiv);
            }
        }
    } catch (e) {
        if (selectionPopup) {
            selectionPopup.textContent = '';
            const errDiv = document.createElement('div');
            errDiv.className = 'yx-popup-result';
            errDiv.style.color = '#d93025';
            errDiv.textContent = '翻译失败';
            selectionPopup.appendChild(errDiv);
        }
    }
}

// 监听选中事件
function initSelectionTranslate() {
    // 读取划词翻译开关；popup 修改后通过 onChanged 即时生效（无需刷新页面）
    try {
        chrome.storage.local.get(['selection_translate_enabled'], (r) => {
            if (!chrome.runtime.lastError) {
                selectionTranslateEnabled = r.selection_translate_enabled !== false;
            }
        });
        chrome.storage.onChanged.addListener((changes, area) => {
            if (area === 'local' && changes.selection_translate_enabled) {
                selectionTranslateEnabled = changes.selection_translate_enabled.newValue !== false;
            }
        });
    } catch (e) { /* 读取失败保持默认开启 */ }

    document.addEventListener('mouseup', (e) => {
        if (!shouldHandleSelectionMouseup(
            e,
            selectionTranslateEnabled,
            isSensitiveHost(window.location.hostname, window.location.protocol)
        )) return;
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
    injectSharedStyles();
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
      .yx-toast-icon.error {
        border: none;
        animation: none;
        background: none;
        width: auto;
        height: auto;
        color: #f28b82;
        font-size: 16px;
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

    // 使用 DOM API 构建，避免 XSS
    container.innerHTML = '';
    const msgDiv = document.createElement('div');
    msgDiv.className = 'yx-toast-message';
    if (type === 'success') {
        const icon = document.createElement('div');
        icon.className = 'yx-toast-icon success';
        icon.textContent = '\u2713';
        msgDiv.appendChild(icon);
    } else if (type === 'restore') {
        const icon = document.createElement('div');
        icon.className = 'yx-toast-icon success';
        icon.textContent = '\u21BA';
        msgDiv.appendChild(icon);
    } else if (type === 'error') {
        const icon = document.createElement('div');
        icon.className = 'yx-toast-icon error';
        icon.textContent = '\u2715';
        msgDiv.appendChild(icon);
    } else {
        const icon = document.createElement('div');
        icon.className = 'yx-toast-icon';
        msgDiv.appendChild(icon);
    }
    const span = document.createElement('span');
    span.textContent = message;
    msgDiv.appendChild(span);
    container.appendChild(msgDiv);

    requestAnimationFrame(() => {
        container.classList.add('show');
    });

    if (type !== 'loading') {
        setTimeout(() => {
            container.classList.remove('show');
        }, 3000);
    }
}

// ===== 消息重试（Service Worker 可能被终止后重启） =====
async function sendMessageWithRetry(message, maxRetries = 2) {
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            if (!chrome.runtime?.id) {
                throw new Error('Extension context invalidated');
            }
            const response = await chrome.runtime.sendMessage(message);
            return response;
        } catch (e) {
            const msg = e.message || '';
            const isDisconnected = msg.includes('Extension context invalidated') ||
                                   msg.includes('Could not establish connection') ||
                                   msg.includes('Receiving end does not exist') ||
                                   msg.includes('message port closed');
            if (isDisconnected && attempt < maxRetries) {
                // Service Worker 可能被终止了，等待后重试（发消息会唤醒 SW）
                await new Promise(r => setTimeout(r, 500 * (attempt + 1)));
                continue;
            }
            throw e;
        }
    }
}

// 状态管理
let isTranslated = false;
let isTranslating = false; // 翻译锁，防止并发执行
// 翻译代际令牌：还原 / 切换语言时自增，使在途的旧翻译响应回写失效（防 stale 结果覆盖已还原内容）
let translationGeneration = 0;
let observer = null;
let mutationDebounceTimer = null; // 防抖计时器
let cacheSaveTimer = null; // 缓存保存防抖计时器
let pendingCacheUpdates = {}; // 待保存的缓存更新
let dynamicTranslationBudgetStartedAt = 0;
let dynamicTranslationBudgetChars = 0;
const originalTextMap = new WeakMap(); // 存储原文: Node -> String
const translationCache = new Map();    // 内存缓存: String -> String（使用 LRU 策略）
const MAX_PENDING_NODES = 100; // MutationObserver 最大待处理节点数

function resolveTranslateMode(mode, legacyEnabled) {
    if (mode === 'auto_all' || mode === 'whitelist' || mode === 'manual') return mode;
    return legacyEnabled === true ? 'auto_all' : 'manual';
}

function shouldHandleSelectionMouseup(event, enabled, sensitive) {
    return event?.isTrusted === true && enabled === true && sensitive === false;
}

function resetDynamicTranslationBudget() {
    dynamicTranslationBudgetStartedAt = 0;
    dynamicTranslationBudgetChars = 0;
}

function consumeDynamicTranslationBudget(chars, now = Date.now()) {
    if (!Number.isFinite(chars) || chars < 0) return false;
    if (!dynamicTranslationBudgetStartedAt ||
        now - dynamicTranslationBudgetStartedAt >= DYNAMIC_TRANSLATION_BUDGET_WINDOW_MS) {
        dynamicTranslationBudgetStartedAt = now;
        dynamicTranslationBudgetChars = 0;
    }
    if (dynamicTranslationBudgetChars + chars > MAX_DYNAMIC_TRANSLATION_CHARS_PER_WINDOW) {
        return false;
    }
    dynamicTranslationBudgetChars += chars;
    return true;
}

function chunkTranslationTexts(texts) {
    const chunks = [];
    let chunk = [];
    let chars = 0;
    for (const text of texts) {
        if (typeof text !== 'string' || !text.trim() || text.length > MAX_TRANSLATION_TEXT_CHARS) continue;
        if (chunk.length >= MAX_TRANSLATION_BATCH_ITEMS ||
            (chunk.length > 0 && chars + text.length > MAX_TRANSLATION_BATCH_CHARS)) {
            chunks.push(chunk);
            chunk = [];
            chars = 0;
        }
        chunk.push(text);
        chars += text.length;
    }
    if (chunk.length > 0) chunks.push(chunk);
    return chunks;
}

// 翻译 document.title
let originalTitle = null;

// HTML 属性翻译：存储原始属性值 Element -> { attr: originalValue }
const originalAttrMap = new WeakMap();
// 跟踪已翻译属性的元素（使用 WeakRef 避免阻止 GC）
const translatedAttrRefs = [];
// 快速查重：同一元素只记录一次 WeakRef，避免 SPA 长跑数组无限增长
let translatedAttrElementSet = new WeakSet();
// 触发数组压缩的阈值：超过则清掉已失效（被 GC）的 WeakRef
const TRANSLATED_ATTR_REFS_COMPACT_THRESHOLD = 5000;

// 清理已失效（element 已被 GC）的 WeakRef
function compactTranslatedAttrRefs() {
    let writeIdx = 0;
    for (let readIdx = 0; readIdx < translatedAttrRefs.length; readIdx++) {
        if (translatedAttrRefs[readIdx].deref() !== undefined) {
            translatedAttrRefs[writeIdx++] = translatedAttrRefs[readIdx];
        }
    }
    translatedAttrRefs.length = writeIdx;
}

// 记录已翻译属性元素，去重 push
function recordTranslatedAttrElement(element) {
    if (translatedAttrElementSet.has(element)) return;
    translatedAttrElementSet.add(element);
    translatedAttrRefs.push(new WeakRef(element));
    // 长跑保护：超阈值时清掉已失效的 WeakRef，控制数组规模
    if (translatedAttrRefs.length > TRANSLATED_ATTR_REFS_COMPACT_THRESHOLD) {
        compactTranslatedAttrRefs();
    }
}

// 重置追踪器（还原原文 / 语言切换时调用）
function resetTranslatedAttrTracker() {
    translatedAttrRefs.length = 0;
    translatedAttrElementSet = new WeakSet();
}

// 翻译统计
const translateStats = { totalTranslated: 0, cacheHits: 0, apiCalls: 0 };

// 双语对照模式
let bilingualMode = false;

// 划词翻译开关（popup 可关闭；默认开启）
let selectionTranslateEnabled = true;

// 当前翻译目标语言（用于检测语言切换）
let currentTargetLang = null;

// 缓存是否已从 background IndexedDB 加载到内存：只在首次整页翻译时拉一次，
// 子树翻译（MutationObserver/IntersectionObserver 触发）直接复用内存缓存
let cacheLoaded = false;

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

// 纯文本可翻译判定（无 DOM 依赖，便于测试）。参数：文本、父标签名、父 class 字符串、是否可编辑
function isTextTranslatable(text, parentTagName, parentClassName, isEditable) {
    if (IGNORED_TAGS.has(parentTagName)) return false;
    if (isEditable) return false;

    // Check parent classes for icon indicators
    if (parentClassName && typeof parentClassName === 'string') {
        const cls = parentClassName.toLowerCase();
        if (cls.includes('material-icons') || cls.includes('material-symbols') ||
            cls.includes('fa-') || cls.includes('icon') || cls.includes('glyph')) {
            return false;
        }
    }

    const trimmed = (text || '').trim();
    if (trimmed.length < MIN_TEXT_LENGTH) return false;

    if (/^\d+$/.test(trimmed)) return false;
    if (/^[^\p{L}]+$/u.test(trimmed)) return false;
    // JSON-like or Caps-constants
    if (/^\{.*\}$/.test(trimmed) || /^[A-Z0-9_]+$/.test(trimmed)) return false;

    // Ignore snake_case strings often used for ligatures (e.g. keyboard_arrow_down)
    if (/^[a-z0-9]+(_[a-z0-9]+)+$/.test(trimmed)) return false;

    return true;
}

function isTranslatable(node) {
    const parent = node.parentNode;
    if (!parent) return false;
    return isTextTranslatable(node.nodeValue, parent.tagName, parent.className, parent.isContentEditable);
}

function getTextNodes(root = document.body) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    let node;
    while (node = walker.nextNode()) {
        if (isTranslatable(node)) nodes.push(node);
    }
    return nodes;
}

// 获取含有可翻译属性的元素
function getTranslatableAttrs(root = document.body) {
    const results = []; // { element, attr, text }
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
    let el = walker.currentNode;
    while (el) {
        if (!IGNORED_TAGS.has(el.tagName)) {
            for (const attr of TRANSLATABLE_ATTRS) {
                const val = el.getAttribute(attr);
                if (val && val.trim().length >= MIN_TEXT_LENGTH) {
                    const text = val.trim();
                    // 跳过纯数字、纯符号、纯中文
                    if (/^\d+$/.test(text) || /^[^\p{L}]+$/u.test(text)) continue;
                    if (/^[\u4e00-\u9fa5]+$/.test(text)) continue;
                    results.push({ element: el, attr, text });
                }
            }
        }
        el = walker.nextNode();
    }
    return results;
}

async function loadCache() {
    if (!chrome.runtime?.id) return;
    try {
        // 从 background 的 IndexedDB 加载缓存到内存 LRU
        const response = await sendMessageWithRetry({ type: 'CACHE_GET_ALL' });
        if (response?.success && response.results) {
            const entries = Object.entries(response.results);
            const toLoad = entries.slice(-MAX_CACHE_SIZE);
            for (const [key, value] of toLoad) {
                translationCache.set(key, value);
            }
        }
    } catch (e) {
        console.warn("YX翻译: 缓存加载失败:", e);
    }
}

// 仅在首次整页翻译时拉一次全量缓存到内存；子树翻译复用已加载的内存缓存
async function ensureCacheLoaded() {
    if (cacheLoaded) return;
    await loadCache();
    cacheLoaded = true;
}

// 缓存保存（带防抖，合并多次调用，写入 background IndexedDB）
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
        const updates = { ...pendingCacheUpdates };
        pendingCacheUpdates = {};
        // 发消息到 background 写入 IndexedDB
        sendMessageWithRetry({
            type: 'CACHE_PUT_BATCH',
            entries: updates
        }).catch(e => console.warn('YX翻译: 缓存保存失败', e));
    }, 500);
}

// 清除缓存（供 popup 调用）
function clearCache() {
    translationCache.clear();
    cacheLoaded = false;
    if (chrome.runtime?.id) {
        sendMessageWithRetry({ type: 'CACHE_CLEAR' }).catch(() => {});
    }
}

function restoreOriginal() {
    // 作废所有在途翻译：之后回来的 chunk 响应会因代际不符被丢弃，不会覆盖已还原内容
    translationGeneration++;
    showToast('正在还原原文...', 'loading');
    hideProgressBar();
    hideOriginalTooltip();
    if (observer) observer.disconnect();
    observer = null;
    // 清理视口翻译观察器
    if (viewportObserver) {
        viewportObserver.disconnect();
        viewportObserver = null;
    }
    resetDynamicTranslationBudget();

    const nodes = getTextNodes();
    let count = 0;
    nodes.forEach(node => {
        if (originalTextMap.has(node)) {
            node.nodeValue = originalTextMap.get(node);
            count++;
        }
    });

    // 还原 document.title
    if (originalTitle !== null) {
        document.title = originalTitle;
        originalTitle = null;
    }

    // 还原 HTML 属性
    for (const ref of translatedAttrRefs) {
        const element = ref.deref();
        if (!element) continue; // 元素已被 GC
        const stored = originalAttrMap.get(element);
        if (stored) {
            for (const [attr, originalValue] of Object.entries(stored)) {
                element.setAttribute(attr, originalValue);
            }
        }
    }
    resetTranslatedAttrTracker();

    isTranslated = false;
    showToast('已还原原文', 'restore');
}

async function performTranslation(root = document.body, isDynamic = false) {
    // 防止并发执行（但允许子树翻译）
    if (isTranslating && root === document.body) {
        console.log('YX翻译: 翻译正在进行中，跳过重复请求');
        return;
    }

    if (root === document.body) {
        isTranslating = true;
    }

    // 每次翻译用独立的命中 key 集合：多条翻译链（整页 + 视口子树）并发时互不清空
    const touchedKeys = new Set();
    try {
        await _doTranslation(root, touchedKeys, isDynamic);
    } finally {
        if (root === document.body) {
            isTranslating = false;
        }
        // 兜底：异常退出时也要把已收集的 touch keys 发出去（_doTranslation 内已 flush 则此处为空操作）
        flushCacheTouches(touchedKeys);
    }
}

// 把传入的 cache 命中 key 集合异步发到 background 更新 lastAccess；不 await，失败忽略
function flushCacheTouches(keysSet) {
    if (!keysSet || keysSet.size === 0 || !chrome.runtime?.id) return;
    const keys = Array.from(keysSet);
    keysSet.clear();
    sendMessageWithRetry({ type: 'CACHE_TOUCH', keys })
        .catch(() => { /* 忽略 touch 失败 */ });
}

async function _doTranslation(root = document.body, touchedKeys = new Set(), isDynamic = false) {
    // 记录本次翻译代际（在任何 await 之前捕获）：还原/切语言会自增代际，
    // 后续每个回写点（含缓存命中同步回写、异步 chunk 回写）比对代际，不符即丢弃，
    // 防止在 storage 读取 / 缓存加载等 await 间隙被还原后，stale 结果又覆盖已还原的 DOM
    let myGen = translationGeneration;

    // 读取目标语言和双语模式设置
    let targetLang = 'zh-CN';
    try {
        const settings = await chrome.storage.local.get(['target_lang', 'bilingual_mode']);
        if (settings.target_lang) targetLang = settings.target_lang;
        bilingualMode = settings.bilingual_mode === true;
    } catch (e) { /* 默认值 */ }

    // 检测目标语言是否切换：如果切换了，先静默还原原文并清空缓存
    if (currentTargetLang && currentTargetLang !== targetLang && isTranslated && root === document.body) {
        console.log(`YX翻译: 目标语言从 ${currentTargetLang} 切换为 ${targetLang}，重新翻译`);
        // 静默还原所有文本节点
        const oldNodes = getTextNodes();
        oldNodes.forEach(node => {
            if (originalTextMap.has(node)) {
                node.nodeValue = originalTextMap.get(node);
            }
        });
        // 还原标题
        if (originalTitle !== null) {
            document.title = originalTitle;
            originalTitle = null;
        }
        // 还原属性
        for (const ref of translatedAttrRefs) {
            const element = ref.deref();
            if (!element) continue;
            const stored = originalAttrMap.get(element);
            if (stored) {
                for (const [attr, val] of Object.entries(stored)) {
                    element.setAttribute(attr, val);
                }
            }
        }
        resetTranslatedAttrTracker();
        // 清空内存缓存（旧语言的翻译结果无法复用）
        translationCache.clear();
        cacheLoaded = false;
        isTranslated = false;
        // 切换语言：作废旧语言在途翻译，并把本次翻译归入新代际（避免自己被误判 stale）
        translationGeneration++;
        myGen = translationGeneration;
    }
    currentTargetLang = targetLang;

    // 仅首次拉一次全量缓存；子树翻译/MutationObserver 不再重复 dump 全库
    await ensureCacheLoaded();

    // 代际检查：storage 读取 / 缓存加载这些 await 间隙内若用户已还原 / 切语言，立即丢弃，
    // 避免下面命中缓存的【同步回写】把已还原的 DOM 又写回译文（此处进度条尚未显示，无需 hide）
    if (myGen !== translationGeneration) {
        flushCacheTouches(touchedKeys);
        return;
    }

    const nodes = getTextNodes(root);
    const textNodeMap = new Map();
    const missingTranslations = new Set();

    nodes.forEach(node => {
        if (!originalTextMap.has(node)) {
            originalTextMap.set(node, node.nodeValue);
        }
        // 始终使用原文作为翻译源，而非当前可能已翻译的文本
        const text = (originalTextMap.get(node) || node.nodeValue).trim();
        const cached = cacheGet(text);
        if (cached !== undefined) {
            translateStats.cacheHits++;
            translateStats.totalTranslated++;
            touchedKeys.add(text);
            applyTextToNode(node, cached);
        } else {
            if (!textNodeMap.has(text)) textNodeMap.set(text, []);
            textNodeMap.get(text).push(node);
            missingTranslations.add(text);
        }
    });

    // 整页翻译时：翻译 document.title 和 HTML 属性
    if (root === document.body || root === document.documentElement) {
        // 翻译 document.title（使用原始标题作为翻译源）
        const titleText = (originalTitle || document.title).trim();
        if (titleText.length >= MIN_TEXT_LENGTH && !/^[\u4e00-\u9fa5]+$/.test(titleText)) {
            if (originalTitle === null) {
                originalTitle = document.title;
            }
            const cachedTitle = cacheGet(titleText);
            if (cachedTitle !== undefined) {
                document.title = cachedTitle;
                translateStats.cacheHits++;
                touchedKeys.add(titleText);
            } else {
                missingTranslations.add(titleText);
                if (!textNodeMap.has(titleText)) textNodeMap.set(titleText, []);
                textNodeMap.get(titleText).push({ __isTitle: true });
            }
        }

        // 收集可翻译属性（使用原始属性值作为翻译源）
        const attrItems = getTranslatableAttrs(root);
        attrItems.forEach(({ element, attr }) => {
            if (!originalAttrMap.has(element)) {
                originalAttrMap.set(element, {});
            }
            const stored = originalAttrMap.get(element);
            if (!stored[attr]) {
                stored[attr] = element.getAttribute(attr);
            }
            // 始终使用原始属性值作为翻译源
            const text = stored[attr].trim();
            recordTranslatedAttrElement(element);
            const cached = cacheGet(text);
            if (cached !== undefined) {
                element.setAttribute(attr, cached);
                translateStats.cacheHits++;
                touchedKeys.add(text);
            } else {
                missingTranslations.add(text);
                if (!textNodeMap.has(text)) textNodeMap.set(text, []);
                textNodeMap.get(text).push({ __isAttr: true, element, attr });
            }
        });
    }

    const chunks = chunkTranslationTexts(Array.from(missingTranslations));
    const textsToTranslate = chunks.flat();
    if (textsToTranslate.length === 0) {
        isTranslated = true;
        // 全部命中缓存的情况：仍需 flush 已收集的命中 key，否则 lastAccess 永不更新
        flushCacheTouches(touchedKeys);
        return;
    }

    if (isDynamic) {
        const dynamicChars = textsToTranslate.reduce((sum, text) => sum + text.length, 0);
        if (!consumeDynamicTranslationBudget(dynamicChars)) return;
    }

    const totalCount = textsToTranslate.length;
    const isFullPage = root === document.body || root === document.documentElement;
    const showProgress = totalCount > 5;

    // 整页翻译时显示进度条和 Toast
    if (showProgress && isFullPage) {
        showProgressBar();
        showToast(`翻译中 0/${totalCount}...`, 'loading');
    }

    const MAX_CONCURRENT = 3; // 最多同时发送 3 个 chunk
    let translatedCount = 0;
    let consecutiveFailedBatches = 0; // 连续"整批全失败"的批次数，用于熔断

    // 并发发送 chunk（限制并发数）
    for (let i = 0; i < chunks.length; i += MAX_CONCURRENT) {
        if (!chrome.runtime?.id) break;

        const concurrentChunks = chunks.slice(i, i + MAX_CONCURRENT);
        const promises = concurrentChunks.map(chunk =>
            sendMessageWithRetry({
                type: 'TRANSLATE_TEXT_BATCH',
                texts: chunk
            }).then(response => ({ response, chunk }))
              .catch(e => ({ error: e, chunk }))
        );

        const results = await Promise.all(promises);

        // 代际检查：在途期间用户已还原 / 切换语言 → 丢弃 stale 结果，停止回写
        if (myGen !== translationGeneration) {
            if (isFullPage && showProgress) hideProgressBar();
            return;
        }

        let batchSuccess = 0;
        let batchFailure = 0;
        for (const { response, error, chunk } of results) {
            if (error) {
                batchFailure++;
                console.warn('YX翻译: 翻译消息错误:', error);
                continue;
            }
            if (response && response.success) {
                saveCache(response.results);
                applyBatchTranslations(response.results, textNodeMap);
                applyBatchSpecialTranslations(response.results, textNodeMap);
                translatedCount += chunk.length;
                translateStats.apiCalls += chunk.length;
                translateStats.totalTranslated += chunk.length;
                batchSuccess++;
            } else if (response && !response.success) {
                batchFailure++;
                console.warn('YX翻译: 批次翻译失败', response.error);
            }
        }

        // 熔断：只有"整批全部失败"才累加；本批有任一成功即清零。
        // 旧逻辑在单 chunk 成功时清零整体计数，导致间歇性失败永远攒不到阈值
        if (batchSuccess > 0) {
            consecutiveFailedBatches = 0;
        } else if (batchFailure > 0) {
            consecutiveFailedBatches++;
        }

        // 更新进度
        if (showProgress) {
            const percent = Math.round((translatedCount / totalCount) * 100);
            if (isFullPage) updateProgress(percent);
            showToast(`翻译中 ${translatedCount}/${totalCount} (${percent}%)`, 'loading');
        }

        // 连续两批整批失败 → 熔断中止
        if (consecutiveFailedBatches >= 2) {
            if (isFullPage) hideProgressBar();
            showToast('翻译遇到问题，部分内容可能未翻译', 'error');
            break;
        }
    }

    // 翻译完成，隐藏进度条
    if (showProgress && isFullPage) {
        hideProgressBar();
    }
    isTranslated = true;

    // 异步触摸 lastAccess，让活跃缓存数据不被 TTL 清掉（不 await，失败也不影响主流程）
    flushCacheTouches(touchedKeys);
}

function applyTextToNode(node, translatedText) {
    const current = node.nodeValue;
    if (!current) return;
    const originalText = originalTextMap.get(node) || current;
    const match = current.match(/^(\s*)([\s\S]*?)(\s*)$/);
    if (match) {
        const [_, prefix, content, suffix] = match;
        if (bilingualMode && translatedText !== content) {
            node.nodeValue = prefix + translatedText + '\n' + originalText.trim() + suffix;
        } else {
            node.nodeValue = prefix + translatedText + suffix;
        }
    } else {
        if (bilingualMode) {
            node.nodeValue = translatedText + '\n' + originalText.trim();
        } else {
            node.nodeValue = translatedText;
        }
    }
}

function applyBatchTranslations(results, textNodeMap) {
    for (const [original, translated] of Object.entries(results)) {
        if (original === translated) continue;
        const nodes = textNodeMap.get(original);
        if (nodes) {
            nodes.forEach(node => {
                // 跳过特殊标记（title/属性），只处理真正的文本节点
                if (node.__isTitle || node.__isAttr) return;
                applyTextToNode(node, translated);
            });
        }
    }
}

// 应用 title 和属性翻译
function applyBatchSpecialTranslations(results, textNodeMap) {
    for (const [original, translated] of Object.entries(results)) {
        if (original === translated) continue;
        const items = textNodeMap.get(original);
        if (!items) continue;
        items.forEach(item => {
            if (item.__isTitle) {
                document.title = translated;
            } else if (item.__isAttr) {
                item.element.setAttribute(item.attr, translated);
            }
        });
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

                // 使用 IntersectionObserver 按需翻译新增节点
                performViewportTranslation(toProcess);
            }
        }, 200);
    });
    observer.observe(document.body, { childList: true, subtree: true });
}

// 检测页面语言是否与目标语言相同（相同则跳过翻译）
function detectPageLanguage(targetLang = 'zh-CN') {
    const htmlLang = (document.documentElement.lang || '').toLowerCase();
    const targetPrefix = targetLang.split('-')[0].toLowerCase(); // 'zh-CN' -> 'zh'

    // 1. 先检查 html lang 属性
    if (htmlLang && htmlLang.startsWith(targetPrefix)) {
        return 'target'; // 页面语言与目标语言相同，应跳过
    }

    // 2. 对中文目标语言，抽样检测中文字符占比
    if (targetPrefix === 'zh') {
        const sampleTexts = [];
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
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

        const allText = sampleTexts.join('');
        const chineseChars = allText.match(/[\u4e00-\u9fa5]/g) || [];
        const ratio = chineseChars.length / allText.length;

        // 中文字符超过 30% 视为中文页面
        return ratio > 0.3 ? 'target' : 'other';
    }

    // 3. 非中文目标语言，仅依靠 html lang 判断
    return 'other';
}

let autoTranslateTriggered = false; // 防止重复触发

// 触发自动翻译
function triggerAutoTranslate(targetLang) {
    const detectedLang = detectPageLanguage(targetLang);
    if (detectedLang !== 'target') {
        autoTranslateTriggered = true;
        resetDynamicTranslationBudget();
        console.log('YX翻译: 检测到外语内容，开始翻译...');
        showToast('正在自动为您翻译...', 'loading');

        performTranslation()
            .then(() => {
                if (chrome.runtime?.id) {
                    enableAutoTranslate();
                    showToast('翻译完成', 'success');
                    sendMessageWithRetry({ type: 'TRANSLATION_DONE' }).catch(() => {});
                }
            })
            .catch(e => {
                console.error('YX翻译: 自动翻译失败', e);
                hideProgressBar();
                showToast('翻译失败，请重试', 'error');
                isTranslating = false;
                autoTranslateTriggered = false;
            });
    } else {
        console.log('YX翻译: 页面语言与目标语言相同，跳过翻译');
    }
}

// 敏感域名/协议判定（纯函数，便于测试）：本地文件、本机、私网、含敏感关键词的 host
// 默认不【自动】翻译，避免把网银/邮箱/内网后台等明文内容自动外发第三方接口；用户仍可手动翻译
function isSensitiveHost(hostname, protocol) {
    if (protocol === 'file:') return true;
    if (!hostname) return false;
    const host = hostname.toLowerCase();
    // 本机 / 环回
    if (host === 'localhost' || host.endsWith('.localhost')) return true;
    if (host === '0.0.0.0') return true;
    if (/^127\./.test(host)) return true;                  // 127.0.0.0/8 整段环回
    if (host === '::1' || host === '[::1]') return true;
    // 内网保留域名
    if (host.endsWith('.local') || host.endsWith('.internal') || host.endsWith('.lan')) return true;
    // 私有 / 链路本地 IPv4 段
    if (/^10\.\d+\.\d+\.\d+$/.test(host)) return true;                  // 10.0.0.0/8
    if (/^192\.168\.\d+\.\d+$/.test(host)) return true;                // 192.168.0.0/16
    if (/^172\.(1[6-9]|2\d|3[01])\.\d+\.\d+$/.test(host)) return true;  // 172.16.0.0/12
    if (/^169\.254\.\d+\.\d+$/.test(host)) return true;                // 169.254.0.0/16 链路本地
    // 私有 / 链路本地 IPv6（hostname 带方括号）：ULA fc00::/7、link-local fe80::/10
    if (/^\[f[cd]/.test(host)) return true;
    if (/^\[fe[89ab]/.test(host)) return true;
    // 含敏感关键词的域名（网银/登录/账户/后台/钱包/支付/邮箱）
    const SENSITIVE_KEYWORDS = [
        'bank', 'login', 'signin', 'account', 'admin', 'wallet', 'payment', 'paypal',
        'mail', 'email', 'webmail', 'inbox', 'outlook', 'proton'
    ];
    return SENSITIVE_KEYWORDS.some(kw => host.includes(kw));
}

function checkAutoTranslate() {
    if (autoTranslateTriggered || isTranslated || isTranslating) return;
    if (!chrome.runtime?.id) return;

    try {
        chrome.storage.local.get([
            'translate_mode', 'auto_translate_enabled',
            'whitelist_domains', 'site_preferences',
            'excluded_domains', 'target_lang'
        ], (result) => {
            if (chrome.runtime.lastError) return;
            if (autoTranslateTriggered || isTranslated) return;

            const currentHost = window.location.hostname;
            const targetLang = result.target_lang || 'zh-CN';
            const sitePrefs = result.site_preferences || {};
            const sitePref = sitePrefs[currentHost];

            // 1. "从不翻译"优先级最高
            if (sitePref === 'never') {
                console.log(`YX翻译: 域名 ${currentHost} 偏好为"从不翻译"`);
                return;
            }
            // 2. 敏感域名强制不自动翻译，避免历史 auto 偏好或白名单导致明文自动外发；
            // 用户仍可手动逐次翻译，但不会被静默记为长期自动翻译。
            if (isSensitiveHost(currentHost, window.location.protocol)) {
                console.log(`YX翻译: 域名 ${currentHost} 命中敏感黑名单，默认不自动翻译（可手动翻译）`);
                return;
            }
            if (sitePref === 'auto') {
                triggerAutoTranslate(targetLang);
                return;
            }

            // 3. 根据翻译模式决定（兼容旧版 auto_translate_enabled）
            const mode = resolveTranslateMode(result.translate_mode, result.auto_translate_enabled);

            switch (mode) {
                case 'auto_all': {
                    // 兼容旧版排除列表
                    const excludedList = result.excluded_domains || [];
                    if (excludedList.includes(currentHost)) {
                        console.log(`YX翻译: 域名 ${currentHost} 在排除列表中`);
                        return;
                    }
                    triggerAutoTranslate(targetLang);
                    break;
                }
                case 'whitelist': {
                    const whitelist = result.whitelist_domains || [];
                    if (whitelist.includes(currentHost)) {
                        triggerAutoTranslate(targetLang);
                    } else {
                        console.log(`YX翻译: 域名 ${currentHost} 不在白名单中`);
                    }
                    break;
                }
                case 'manual':
                    // 不自动翻译
                    break;
            }
        });
    } catch (e) {
        // 忽略错误
    }
}

// 初始化
function init() {
    initSelectionTranslate();
    initHoverOriginal();

    // 立即尝试一次
    checkAutoTranslate();

    // 延迟重试：SPA 页面内容可能异步加载，document_idle 时 body 可能还是空的
    setTimeout(() => checkAutoTranslate(), 1000);
    setTimeout(() => checkAutoTranslate(), 3000);
}

// 后退/前进缓存恢复时重新检测
window.addEventListener('pageshow', (e) => {
    if (e.persisted) {
        autoTranslateTriggered = false;
        checkAutoTranslate();
    }
});

// 后台标签页切到前台时重新检测
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && !isTranslated) {
        checkAutoTranslate();
    }
});

// 测试环境下不触发页面初始化（init 会操作 DOM / 注册观察器）
if (!__TEST__) {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (!chrome.runtime?.id) return;

    if (request.type === 'START_TRANSLATE') {
        resetDynamicTranslationBudget();
        showToast('开始分析页面...', 'loading');
        performTranslation()
            .then(() => {
                if (chrome.runtime?.id) {
                    enableAutoTranslate();
                    showToast('翻译完成', 'success');
                    sendMessageWithRetry({ type: 'TRANSLATION_DONE' }).catch(() => {});
                }
            })
            .catch(e => {
                console.error('YX翻译: 手动翻译失败', e);
                hideProgressBar();
                showToast('翻译失败，请重试', 'error');
                isTranslating = false;
            });
        sendResponse({ status: 'started' });
    } else if (request.type === 'RESTORE_ORIGINAL') {
        restoreOriginal();
        sendResponse({ status: 'restored' });
    } else if (request.type === 'TRANSLATE_SELECTION') {
        // 右键菜单触发的选中文本翻译
        const text = typeof request.text === 'string' ? request.text.trim() : '';
        const selection = window.getSelection();
        if (text.length >= 2 && text.length <= 500 && selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const rect = range.getBoundingClientRect();
            showSelectionPopup(text, rect.left + window.scrollX, rect.bottom + window.scrollY);
        }
        sendResponse({ status: 'ok' });
    } else if (request.type === 'GET_TRANSLATE_STATS') {
        sendResponse(translateStats);
    } else if (request.type === 'GET_SITE_STATUS') {
        // popup 用于显示"敏感站点，已禁用自动翻译"提示
        sendResponse({ sensitive: isSensitiveHost(window.location.hostname, window.location.protocol) });
    }
});

// ===== 测试导出（仅 Node 环境；浏览器中 module 未定义，不影响运行）=====
if (__TEST__) {
    module.exports = {
        isTextTranslatable,
        isSensitiveHost,
        cacheGet,
        cacheSet,
        translationCache,
        MAX_CACHE_SIZE,
        resolveTranslateMode,
        shouldHandleSelectionMouseup,
        chunkTranslationTexts,
        consumeDynamicTranslationBudget,
        resetDynamicTranslationBudget,
    };
}
