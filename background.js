chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'TRANSLATE_TEXT_BATCH') {
    (async () => {
      try {
        const results = await handleBatchTranslation(request.texts);
        sendResponse({ success: true, results });
      } catch (error) {
        console.error("Batch translation failed:", error);
        sendResponse({ success: false, error: error.message });
      }
    })();
    return true;
  }
});

async function handleBatchTranslation(texts) {
  const results = {};
  const queue = [...texts];
  // 调整为 18：黄金平衡点
  // 既能通过并行请求大幅提升速度，又将触发 API 限制的风险降到最低
  const BATCH_SIZE = 18;

  while (queue.length > 0) {
    const batch = queue.splice(0, BATCH_SIZE);

    const promises = batch.map(text =>
      Promise.race([
        translateSingle(text),
        new Promise(resolve => setTimeout(() => resolve({ original: text, translated: text }), 6000))
      ])
    );

    const batchResults = await Promise.all(promises);

    batchResults.forEach(res => {
      if (res && res.original) {
        results[res.original] = res.translated || res.original;
      }
    });
  }
  return results;
}

const AI_GLOSSARY = {
  "agent": [["代理", "智能体"], ["经纪人", "智能体"]],
  "transformer": [["变压器", "Transformer"]],
  "token": [["代币", "Token"], ["令牌", "Token"], ["标记", "Token"]],
  "prompt": [["迅速", "提示词"], ["提示", "提示词"]],
  "zero-shot": [["零射", "零样本"], ["零镜头", "零样本"]],
  "few-shot": [["少射", "少样本"], ["几射", "少样本"], ["几次射击", "少样本"]],
  "chain of thought": [["思想链", "思维链"]],
  "robustness": [["稳健性", "鲁棒性"]],
  "corpus": [["尸体", "语料库"], ["全集", "语料库"]],
  "epoch": [["时代", "轮次"], ["纪元", "轮次"]]
};

async function translateSingle(text) {
  if (!text || !text.trim()) return { original: text, translated: text };

  try {
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=zh-CN&dt=t&q=${encodeURIComponent(text)}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data && data[0]) {
      let translatedText = data[0].map(segment => segment[0]).join('');
      translatedText = refineTranslation(text, translatedText);
      return { original: text, translated: translatedText };
    }
    return { original: text, translated: text };
  } catch (error) {
    return { original: text, translated: text };
  }
}

function refineTranslation(source, target) {
  if (!target) return source;
  let result = target;
  const lowerSource = source.toLowerCase();

  for (const [key, replacements] of Object.entries(AI_GLOSSARY)) {
    if (lowerSource.includes(key)) {
      replacements.forEach(([bad, good]) => {
        result = result.split(bad).join(good);
      });
    }
  }
  return result;
}
