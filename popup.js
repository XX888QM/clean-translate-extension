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

// Handle buttons
document.getElementById('translateBtn').addEventListener('click', () => {
    sendMessageToTab({ type: 'START_TRANSLATE' });
});

document.getElementById('restoreBtn').addEventListener('click', () => {
    sendMessageToTab({ type: 'RESTORE_ORIGINAL' });
});

// Handle Toggles
const autoToggle = document.getElementById('autoTranslateToggle');
const excludeToggle = document.getElementById('excludeDomainToggle');
let currentHostname = '';

// Get current tab hostname and load saved state
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs.length === 0) return;
    try {
        const url = new URL(tabs[0].url);
        currentHostname = url.hostname;

        chrome.storage.local.get(['auto_translate_enabled', 'excluded_domains'], (result) => {
            // 检查 storage 错误
            if (chrome.runtime.lastError) {
                console.warn('YX翻译: 读取设置失败', chrome.runtime.lastError.message);
                return;
            }

            // Default to true if undefined
            autoToggle.checked = result.auto_translate_enabled !== false;

            // Check if current domain is excluded
            const excludedList = result.excluded_domains || [];
            if (excludeToggle) {
                excludeToggle.checked = excludedList.includes(currentHostname);
            }
        });
    } catch (e) {
        // Invalid URL (e.g. chrome:// pages)
        console.warn('YX翻译: URL 解析失败', e);
        if (excludeToggle) {
            excludeToggle.disabled = true;
            excludeToggle.parentElement.title = '此页面不支持域名排除';
        }
    }
});

// Save global state on change
autoToggle.addEventListener('change', () => {
    chrome.storage.local.set({ auto_translate_enabled: autoToggle.checked }, () => {
        if (chrome.runtime.lastError) {
            console.warn('YX翻译: 保存自动翻译设置失败', chrome.runtime.lastError.message);
        }
    });
});

// Save domain exclusion state on change
if (excludeToggle) {
    excludeToggle.addEventListener('change', () => {
        if (!currentHostname) return;

        chrome.storage.local.get(['excluded_domains'], (result) => {
            if (chrome.runtime.lastError) {
                console.warn('YX翻译: 读取排除域名列表失败', chrome.runtime.lastError.message);
                return;
            }

            let list = result.excluded_domains || [];
            if (excludeToggle.checked) {
                // Add to blocklist
                if (!list.includes(currentHostname)) list.push(currentHostname);
            } else {
                // Remove from blocklist
                list = list.filter(domain => domain !== currentHostname);
            }
            chrome.storage.local.set({ excluded_domains: list }, () => {
                if (chrome.runtime.lastError) {
                    console.warn('YX翻译: 保存排除域名列表失败', chrome.runtime.lastError.message);
                }
            });
        });
    });
}

// 监听翻译完成消息
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
        // 更新缓存大小显示
        updateCacheSize();
    }
});

// 缓存管理
const cacheSizeEl = document.getElementById('cacheSize');
const clearCacheBtn = document.getElementById('clearCacheBtn');

// 更新缓存大小显示
function updateCacheSize() {
    chrome.storage.local.get(['translation_cache'], (result) => {
        if (chrome.runtime.lastError) {
            console.warn('YX翻译: 读取缓存大小失败', chrome.runtime.lastError.message);
            if (cacheSizeEl) {
                cacheSizeEl.textContent = '读取失败';
            }
            return;
        }

        const cache = result.translation_cache || {};
        const size = Object.keys(cache).length;
        if (cacheSizeEl) {
            cacheSizeEl.textContent = `${size} 条`;
        }
    });
}

// 页面加载时获取缓存大小
updateCacheSize();

// 清除缓存按钮
if (clearCacheBtn) {
    clearCacheBtn.addEventListener('click', () => {
        chrome.storage.local.remove(['translation_cache'], () => {
            if (chrome.runtime.lastError) {
                console.warn('YX翻译: 清除缓存失败', chrome.runtime.lastError.message);
                clearCacheBtn.textContent = '清除失败';
                clearCacheBtn.style.color = '#d93025';
                setTimeout(() => {
                    clearCacheBtn.textContent = '清除缓存';
                    clearCacheBtn.style.color = '';
                }, 2000);
                return;
            }

            if (cacheSizeEl) {
                cacheSizeEl.textContent = '0 条';
            }
            clearCacheBtn.textContent = '已清除';
            setTimeout(() => {
                clearCacheBtn.textContent = '清除缓存';
            }, 2000);
        });
    });
}
