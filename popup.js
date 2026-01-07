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

// Handle Auto-Translate Toggle
const autoToggle = document.getElementById('autoTranslateToggle');

// Load saved state
chrome.storage.local.get(['auto_translate_enabled'], (result) => {
    // Default to true if undefined
    autoToggle.checked = result.auto_translate_enabled !== false;
});

// Save state on change
autoToggle.addEventListener('change', () => {
    chrome.storage.local.set({ auto_translate_enabled: autoToggle.checked });
});

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
