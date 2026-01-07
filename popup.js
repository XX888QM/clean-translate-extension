function sendMessageToTab(message) {
    const statusDiv = document.getElementById('status');
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs.length === 0) return;
        const tabId = tabs[0].id;

        chrome.tabs.sendMessage(tabId, message, (response) => {
            if (chrome.runtime.lastError) {
                statusDiv.textContent = "当前页面不支持";
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
        if (excludeToggle) excludeToggle.disabled = true;
    }
});

// Save global state on change
autoToggle.addEventListener('change', () => {
    chrome.storage.local.set({ auto_translate_enabled: autoToggle.checked });
});

// Save domain exclusion state on change
if (excludeToggle) {
    excludeToggle.addEventListener('change', () => {
        if (!currentHostname) return;

        chrome.storage.local.get(['excluded_domains'], (result) => {
            let list = result.excluded_domains || [];
            if (excludeToggle.checked) {
                // Add to blocklist
                if (!list.includes(currentHostname)) list.push(currentHostname);
            } else {
                // Remove from blocklist
                list = list.filter(domain => domain !== currentHostname);
            }
            chrome.storage.local.set({ excluded_domains: list });
        });
    });
}

// Listen for completion message from content script
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
    }
});
