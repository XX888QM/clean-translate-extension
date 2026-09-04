#!/usr/bin/env node
// YX纯净网页翻译 — 轻量测试运行器
// 用法: node tests/run-tests.js  (或 npm test)
//
// 设计：先 mock 浏览器全局环境（chrome / window / document 等），再 require 真源码，
//      使测试覆盖【真正的源码函数】而非复制粘贴的副本（历史上副本与源码漂移曾漏过崩溃 bug）。
//      源码用 __TEST__ / typeof module 守卫，在 Node 下跳过 DOM 初始化副作用、导出纯函数。

const assert = require('node:assert/strict');
const fs = require('node:fs');

// ===== 测试运行器（支持 async 测试）=====
let totalTests = 0;
let passedTests = 0;
let failedTests = 0;
const failures = [];

async function describe(name, fn) {
    console.log(`\n  ${name}`);
    await fn();
}

async function it(name, fn) {
    totalTests++;
    try {
        await fn();
        passedTests++;
        console.log(`    ✓ ${name}`);
    } catch (e) {
        failedTests++;
        console.log(`    ✗ ${name}`);
        console.log(`      ${e.message}`);
        failures.push({ name, error: e.message });
    }
}

// ===== 浏览器环境 mock：让 Node 能安全 require 真源码 =====
const noop = () => {};
const listener = { addListener: noop, removeListener: noop, hasListener: () => false };

// chrome.storage 区域：可被测试改写（用于 refineTranslation 的 user_glossary 场景）
function makeStorageArea() {
    return {
        _data: {},
        get(keys, cb) {
            const result = {};
            const want = Array.isArray(keys) ? keys : (typeof keys === 'string' ? [keys] : Object.keys(keys || {}));
            for (const k of want) if (k in this._data) result[k] = this._data[k];
            if (typeof cb === 'function') { cb(result); return; }
            return Promise.resolve(result);
        },
        set(items, cb) { Object.assign(this._data, items); if (cb) cb(); return Promise.resolve(); },
        remove(keys, cb) {
            for (const k of (Array.isArray(keys) ? keys : [keys])) delete this._data[k];
            if (cb) cb(); return Promise.resolve();
        },
    };
}

global.chrome = {
    runtime: {
        id: 'test-extension-id',
        lastError: null,
        onMessage: listener,
        onInstalled: listener,
        sendMessage: (msg, cb) => { if (typeof cb === 'function') cb(undefined); return Promise.resolve(undefined); },
        getPlatformInfo: (cb) => { const info = { os: 'mac' }; if (cb) cb(info); return Promise.resolve(info); },
    },
    storage: {
        sync: makeStorageArea(),
        local: makeStorageArea(),
        onChanged: listener,
    },
    alarms: { create: noop, get: (n, cb) => { if (cb) cb(undefined); }, onAlarm: listener, clear: noop },
    contextMenus: { create: noop, onClicked: listener, removeAll: (cb) => { if (cb) cb(); } },
    commands: { onCommand: listener },
    action: { setBadgeText: noop, setBadgeBackgroundColor: noop },
    tabs: { query: (q, cb) => { if (cb) cb([]); return Promise.resolve([]); }, sendMessage: noop },
    i18n: { getMessage: (k) => k },
};

global.window = {
    addEventListener: noop, removeEventListener: noop,
    location: { hostname: 'test.local', href: 'http://test.local/' },
    getSelection: () => ({ rangeCount: 0 }),
};
global.document = {
    addEventListener: noop, removeEventListener: noop,
    readyState: 'complete', visibilityState: 'visible',
    createElement: () => ({ style: {}, setAttribute: noop, appendChild: noop, addEventListener: noop, classList: { add: noop } }),
    head: { appendChild: noop }, body: null,
    getElementById: () => null, querySelector: () => null, querySelectorAll: () => [],
    createTreeWalker: () => ({ nextNode: () => null, currentNode: null }),
};
global.self = global;
global.MutationObserver = class { observe() {} disconnect() {} takeRecords() { return []; } };
global.IntersectionObserver = class { observe() {} disconnect() {} unobserve() {} };
global.NodeFilter = { SHOW_TEXT: 4, SHOW_ELEMENT: 1 };

// ===== require 真源码（__TEST__ 守卫已跳过 DOM 初始化）=====
const bg = require('../background.js');
const ct = require('../content.js');

// ===== 镜像逻辑（源码内联、尚未导出的部分，标注以待后续抽取）=====
// calculateChineseRatio：源码内联于 content.js detectPageLanguage，未导出，此处为镜像
function calculateChineseRatio(text) {
    if (!text || text.length === 0) return 0;
    const chineseChars = text.match(/[一-龥]/g) || [];
    return chineseChars.length / text.length;
}
// shouldAutoTranslate：源码内联于 content.js checkAutoTranslate，未导出，此处为镜像
function shouldAutoTranslate(mode, sitePref, isInWhitelist, isInExcludeList, isSensitive = false) {
    if (sitePref === 'never') return false;
    if (isSensitive) return false;
    if (sitePref === 'auto') return true;
    switch (ct.resolveTranslateMode(mode)) {
        case 'auto_all': return !isInExcludeList;
        case 'whitelist': return isInWhitelist;
        case 'manual': return false;
        default: return false;
    }
}
// LRU 容量淘汰镜像：真源码 MAX_CACHE_SIZE=10000，端到端测淘汰需插万级条目，故用小容量镜像验证淘汰逻辑
function createLRUCache(maxSize) {
    const cache = new Map();
    return {
        get(key) {
            if (cache.has(key)) { const v = cache.get(key); cache.delete(key); cache.set(key, v); return v; }
            return undefined;
        },
        set(key, value) {
            if (cache.has(key)) cache.delete(key);
            cache.set(key, value);
            if (cache.size > maxSize) cache.delete(cache.keys().next().value);
        },
        size() { return cache.size; },
    };
}

// ===== 测试用例 =====
async function main() {

await describe('isTextTranslatable [真源码 content.js]', async () => {
    const f = ct.isTextTranslatable;
    await it('应该翻译正常英文文本', () => assert.equal(f('Hello World', 'DIV', '', false), true));
    await it('应该跳过 SCRIPT 标签内文本', () => assert.equal(f('var x = 1', 'SCRIPT', '', false), false));
    await it('应该跳过 STYLE 标签内文本', () => assert.equal(f('.cls { color: red }', 'STYLE', '', false), false));
    await it('应该跳过 CODE 标签内文本', () => assert.equal(f('console.log()', 'CODE', '', false), false));
    await it('应该跳过 contentEditable 元素', () => assert.equal(f('editable text', 'DIV', '', true), false));
    await it('应该跳过 icon class 元素', () => {
        assert.equal(f('arrow_back', 'SPAN', 'material-icons', false), false);
        assert.equal(f('home', 'I', 'fa-icon', false), false);
    });
    await it('应该跳过太短的文本', () => assert.equal(f('a', 'DIV', '', false), false));
    await it('应该跳过纯数字', () => assert.equal(f('12345', 'DIV', '', false), false));
    await it('应该跳过纯符号', () => {
        assert.equal(f('---', 'DIV', '', false), false);
        assert.equal(f('***', 'DIV', '', false), false);
    });
    await it('应该跳过 JSON 格式文本', () => assert.equal(f('{"key":"value"}', 'DIV', '', false), false));
    await it('应该跳过全大写常量', () => {
        assert.equal(f('MAX_SIZE', 'DIV', '', false), false);
        assert.equal(f('API_KEY_NAME', 'DIV', '', false), false);
    });
    await it('应该跳过 snake_case 字符串', () => assert.equal(f('keyboard_arrow_down', 'DIV', '', false), false));
    await it('应该翻译混合文本', () => assert.equal(f('Hello 123 World', 'DIV', '', false), true));
    await it('nodeValue 为 null 不应抛错（健壮性）', () => assert.equal(f(null, 'DIV', '', false), false));
});

await describe('LRU 内存缓存 [真源码 content.js cacheGet/cacheSet]', async () => {
    const reset = () => ct.translationCache.clear();
    await it('应该正确存取值', () => { reset(); ct.cacheSet('a', '翻译A'); assert.equal(ct.cacheGet('a'), '翻译A'); });
    await it('不存在的键应返回 undefined', () => { reset(); assert.equal(ct.cacheGet('missing'), undefined); });
    await it('访问后应移到最近使用（末尾）', () => {
        reset(); ct.cacheSet('a', '1'); ct.cacheSet('b', '2'); ct.cacheSet('c', '3');
        ct.cacheGet('a'); // a 移到末尾
        assert.deepEqual([...ct.translationCache.keys()], ['b', 'c', 'a']);
    });
    await it('覆盖已有键应更新值且不增容量', () => {
        reset(); ct.cacheSet('a', '旧值'); ct.cacheSet('a', '新值');
        assert.equal(ct.cacheGet('a'), '新值');
        assert.equal(ct.translationCache.size, 1);
    });
    await it('clear 应清空缓存', () => {
        reset(); ct.cacheSet('a', '1'); ct.cacheSet('b', '2'); ct.translationCache.clear();
        assert.equal(ct.translationCache.size, 0);
    });
    // 淘汰逻辑用小容量镜像（真 MAX_CACHE_SIZE=10000）
    await it('[镜像] 超出容量时应淘汰最旧条目', () => {
        const c = createLRUCache(3);
        c.set('a', '1'); c.set('b', '2'); c.set('c', '3'); c.set('d', '4');
        assert.equal(c.get('a'), undefined);
        assert.equal(c.get('d'), '4');
    });
    await it('[镜像] 访问后最近使用项不被淘汰', () => {
        const c = createLRUCache(3);
        c.set('a', '1'); c.set('b', '2'); c.set('c', '3');
        c.get('a'); c.set('d', '4'); // 应淘汰 b
        assert.equal(c.get('a'), '1');
        assert.equal(c.get('b'), undefined);
    });
});

await describe('normalizeTargetLang / normalizeEngine [真源码 background.js]', async () => {
    await it('白名单语言原样透传', () => {
        assert.equal(bg.normalizeTargetLang('zh-CN'), 'zh-CN');
        assert.equal(bg.normalizeTargetLang('ja'), 'ja');
        assert.equal(bg.normalizeTargetLang('vi'), 'vi');
    });
    await it('非白名单语言回退 zh-CN', () => {
        assert.equal(bg.normalizeTargetLang('klingon'), 'zh-CN');
        assert.equal(bg.normalizeTargetLang('__proto__'), 'zh-CN');
    });
    await it('非字符串回退 zh-CN', () => {
        assert.equal(bg.normalizeTargetLang(null), 'zh-CN');
        assert.equal(bg.normalizeTargetLang(123), 'zh-CN');
        assert.equal(bg.normalizeTargetLang(undefined), 'zh-CN');
    });
    await it('白名单引擎原样透传', () => {
        assert.equal(bg.normalizeEngine('claude'), 'claude');
        assert.equal(bg.normalizeEngine('deepl'), 'deepl');
    });
    await it('非白名单/非字符串引擎回退 google_free', () => {
        assert.equal(bg.normalizeEngine('evil'), 'google_free');
        assert.equal(bg.normalizeEngine('__proto__'), 'google_free');
        assert.equal(bg.normalizeEngine(null), 'google_free');
        assert.equal(bg.normalizeEngine(42), 'google_free');
    });
});

await describe('安全边界 [真源码 background.js / content.js]', async () => {
    await it('手动翻译不得暗中开启当前网站自动翻译', () => {
        const popupSource = fs.readFileSync(require.resolve('../popup.js'), 'utf8');
        const contentSource = fs.readFileSync(require.resolve('../content.js'), 'utf8');
        assert.equal(popupSource.includes('recordPreference'), false);
        assert.equal(contentSource.includes('request.recordPreference'), false);
    });

    await it('缺少或脏翻译模式默认 manual，明确旧偏好仍保留', () => {
        for (const resolve of [bg.resolveTranslateMode, ct.resolveTranslateMode]) {
            assert.equal(resolve(undefined, undefined), 'manual');
            assert.equal(resolve('evil', undefined), 'manual');
            assert.equal(resolve(undefined, false), 'manual');
            assert.equal(resolve(undefined, true), 'auto_all');
            assert.equal(resolve('auto_all', false), 'auto_all');
            assert.equal(resolve('whitelist', undefined), 'whitelist');
        }
    });

    await it('批量翻译拒绝非法、超长和超总量输入', () => {
        assert.deepEqual(bg.validateTranslationTexts(['hello']), { texts: ['hello'], totalChars: 5 });
        assert.throws(() => bg.validateTranslationTexts('hello'));
        assert.throws(() => bg.validateTranslationTexts([]));
        assert.throws(() => bg.validateTranslationTexts(['ok', 1]));
        assert.throws(() => bg.validateTranslationTexts(['x'.repeat(1501)]));
        assert.throws(() => bg.validateTranslationTexts(Array(14).fill('x'.repeat(1500))));
        assert.throws(() => bg.validateTranslationTexts(Array(201).fill('ok')));
    });

    await it('content 按条数和总字符组包，并跳过超长单条', () => {
        const chunks = ct.chunkTranslationTexts([
            ...Array(14).fill('x'.repeat(1500)),
            'y'.repeat(1501),
            'tail',
        ]);
        assert.equal(chunks.flat().includes('y'.repeat(1501)), false);
        assert.equal(chunks.flat().at(-1), 'tail');
        for (const chunk of chunks) {
            assert.ok(chunk.length <= 200);
            assert.ok(chunk.reduce((sum, text) => sum + text.length, 0) <= 20000);
        }
    });

    await it('划词只接受可信用户事件', () => {
        assert.equal(ct.shouldHandleSelectionMouseup({ isTrusted: false }, true, false), false);
        assert.equal(ct.shouldHandleSelectionMouseup({ isTrusted: true }, false, false), false);
        assert.equal(ct.shouldHandleSelectionMouseup({ isTrusted: true }, true, true), false);
        assert.equal(ct.shouldHandleSelectionMouseup({ isTrusted: true }, true, false), true);
    });

    await it('后台每标签页预算限制请求并在时间窗后恢复', () => {
        bg.resetTranslationRequestBudgets();
        for (let i = 0; i < 60; i++) {
            assert.equal(bg.consumeTranslationRequestBudget(7, 500, 1000), true);
        }
        assert.equal(bg.consumeTranslationRequestBudget(7, 1, 1000), false);
        assert.equal(bg.consumeTranslationRequestBudget(7, 500, 61000), true);
    });

    await it('动态翻译字符预算限制持续 DOM 更新并在时间窗后恢复', () => {
        ct.resetDynamicTranslationBudget();
        assert.equal(ct.consumeDynamicTranslationBudget(50000, 1000), true);
        assert.equal(ct.consumeDynamicTranslationBudget(1, 1000), false);
        assert.equal(ct.consumeDynamicTranslationBudget(50000, 61000), true);
    });
});

await describe('unwrapCacheValue [真源码 background.js]', async () => {
    await it('v1 纯字符串原样返回', () => assert.equal(bg.unwrapCacheValue('译文'), '译文'));
    await it('v2 {v,t} 解包出 v', () => assert.equal(bg.unwrapCacheValue({ v: '译文', t: 123 }), '译文'));
    await it('缺 v 字段返回 null', () => assert.equal(bg.unwrapCacheValue({ t: 123 }), null));
    await it('null/undefined 返回 null', () => {
        assert.equal(bg.unwrapCacheValue(null), null);
        assert.equal(bg.unwrapCacheValue(undefined), null);
    });
    await it('v 非字符串返回 null', () => assert.equal(bg.unwrapCacheValue({ v: 123, t: 1 }), null));
});

await describe('estimateEntryBytes [真源码 background.js]', async () => {
    await it('字符串 value：(key+val)*2 + 24', () => assert.equal(bg.estimateEntryBytes('ab', 'cd'), (2 + 2) * 2 + 24));
    await it('{v} 对象 value：按 v 长度算', () => assert.equal(bg.estimateEntryBytes('ab', { v: 'cd', t: 1 }), (2 + 2) * 2 + 24));
    await it('空 key/非法 value 退化为 24', () => assert.equal(bg.estimateEntryBytes('', { t: 1 }), 0 + 24));
});

await describe('md5 [真源码 background.js — 百度签名依赖，错一位即签名全失败]', async () => {
    await it("md5('abc')", () => assert.equal(bg.md5('abc'), '900150983cd24fb0d6963f7d28e17f72'));
    await it("md5('')", () => assert.equal(bg.md5(''), 'd41d8cd98f00b204e9800998ecf8427e'));
    await it("md5('hello')", () => assert.equal(bg.md5('hello'), '5d41402abc4b2a76b9719d911017c592'));
    await it('md5(UTF-8 中文) 应与标准一致', () => {
        const crypto = require('node:crypto');
        const expected = crypto.createHash('md5').update('你好', 'utf8').digest('hex');
        assert.equal(bg.md5('你好'), expected);
    });
});

await describe('parseBulkResponse [真源码 background.js]', async () => {
    await it('正常多段拼接按 \\n 拆分、行数匹配', () => {
        const data = [[['你好\n世界', null]]];
        const r = bg.parseBulkResponse(data, 2);
        assert.deepEqual(r.translated, ['你好', '世界']);
        assert.equal(r.matched, true);
    });
    await it('行数不匹配 matched=false', () => {
        const r = bg.parseBulkResponse([[['只有一行', null]]], 3);
        assert.equal(r.matched, false);
    });
    await it('data[0] 含空段被过滤', () => {
        const data = [[['行一', null], [null], ['', null], ['行二', null]]];
        const r = bg.parseBulkResponse(data, 1);
        assert.deepEqual(r.translated, ['行一行二']);
    });
    await it('data[0] 非数组时返回空串单元素', () => {
        const r = bg.parseBulkResponse({}, 1);
        assert.deepEqual(r.translated, ['']);
        assert.equal(r.matched, true);
    });
});

await describe('buildNumberedPrompt [真源码 background.js]', async () => {
    await it('生成 [n] 编号列表', () => {
        assert.equal(bg.buildNumberedPrompt(['a', 'b', 'c']), '[1] a\n[2] b\n[3] c');
    });
    await it('单条', () => assert.equal(bg.buildNumberedPrompt(['只有一条']), '[1] 只有一条'));
    await it('空数组返回空串', () => assert.equal(bg.buildNumberedPrompt([]), ''));
});

await describe('parseLLMReply [真源码 background.js — 5 个 LLM 引擎共用]', async () => {
    const texts = ['Hello', 'World', 'Foo'];
    await it('正常映射', () => {
        const r = bg.parseLLMReply('[1] 你好\n[2] 世界\n[3] 福', texts);
        assert.deepEqual(r, { Hello: '你好', World: '世界', Foo: '福' });
    });
    await it('乱序编号也能映射', () => {
        const r = bg.parseLLMReply('[3] 福\n[1] 你好\n[2] 世界', texts);
        assert.deepEqual(r, { Hello: '你好', World: '世界', Foo: '福' });
    });
    await it('缺号的条目保留原文', () => {
        const r = bg.parseLLMReply('[1] 你好\n[3] 福', texts);
        assert.equal(r.Hello, '你好');
        assert.equal(r.World, 'World'); // 缺 [2]，保留原文
        assert.equal(r.Foo, '福');
    });
    await it('越界编号被忽略不抛错', () => {
        const r = bg.parseLLMReply('[1] 你好\n[9] 越界', texts);
        assert.equal(r.Hello, '你好');
        assert.equal(r.World, 'World');
        assert.equal(r.Foo, 'Foo');
    });
    await it('空回包全部保留原文', () => {
        const r = bg.parseLLMReply('', texts);
        assert.deepEqual(r, { Hello: 'Hello', World: 'World', Foo: 'Foo' });
    });
    await it('构造↔解析 round-trip 对称', () => {
        const src = ['Alpha', 'Beta'];
        const prompt = bg.buildNumberedPrompt(src); // [1] Alpha\n[2] Beta
        // 模拟 LLM 原样回显
        const r = bg.parseLLMReply(prompt, src);
        assert.deepEqual(r, { Alpha: 'Alpha', Beta: 'Beta' });
    });
});

await describe('refineTranslation / buildCompiledGlossary [真源码 background.js, async]', async () => {
    await it('buildCompiledGlossary 返回结构含短路正则', async () => {
        const g = await bg.buildCompiledGlossary();
        assert.ok(g.keywordMap instanceof Map);
        assert.ok(Array.isArray(g.sortedBadWords));
        assert.ok(g.keywordProbe instanceof RegExp);
        assert.ok(g.badWordProbe instanceof RegExp);
    });
    await it('target 为空返回 source', async () => {
        assert.equal(await bg.refineTranslation('anything', ''), 'anything');
    });
    await it('原文不含任何术语 → 短路返回原 target 不变', async () => {
        assert.equal(await bg.refineTranslation('the quick brown fox', '敏捷的棕色狐狸'), '敏捷的棕色狐狸');
    });
    await it('命中内置术语 chatgpt → 校正 聊天GPT 为 ChatGPT', async () => {
        const r = await bg.refineTranslation('I use ChatGPT daily', '我每天使用聊天GPT');
        assert.equal(r, '我每天使用ChatGPT');
    });
    await it('短路②：原文含术语但译文不含待校正词 → target 原样', async () => {
        const r = await bg.refineTranslation('I use ChatGPT', '我使用ChatGPT');
        assert.equal(r, '我使用ChatGPT');
    });
});

await describe('calculateChineseRatio [镜像：content.js 内联于 detectPageLanguage 未导出]', async () => {
    await it('纯中文应返回 1', () => assert.equal(calculateChineseRatio('你好世界'), 1));
    await it('纯英文应返回 0', () => assert.equal(calculateChineseRatio('Hello World'), 0));
    await it('混合文本占比正确', () => { const r = calculateChineseRatio('你好Hello'); assert.ok(r > 0.28 && r < 0.29); });
    await it('空文本返回 0', () => { assert.equal(calculateChineseRatio(''), 0); assert.equal(calculateChineseRatio(null), 0); });
});

await describe('shouldAutoTranslate [镜像：content.js 内联于 checkAutoTranslate 未导出]', async () => {
    await it('网站偏好 never 始终不翻译', () => {
        assert.equal(shouldAutoTranslate('auto_all', 'never', false, false), false);
        assert.equal(shouldAutoTranslate('manual', 'never', false, false), false);
    });
    await it('网站偏好 auto 始终翻译', () => {
        assert.equal(shouldAutoTranslate('manual', 'auto', false, false), true);
    });
    await it('敏感域名即使有 auto 偏好也不自动翻译', () => {
        assert.equal(shouldAutoTranslate('manual', 'auto', false, false, true), false);
        assert.equal(shouldAutoTranslate('auto_all', null, false, false, true), false);
        assert.equal(shouldAutoTranslate('whitelist', null, true, false, true), false);
    });
    await it('auto_all 翻译非排除域名', () => {
        assert.equal(shouldAutoTranslate('auto_all', null, false, false), true);
        assert.equal(shouldAutoTranslate('auto_all', null, false, true), false);
    });
    await it('whitelist 只翻白名单', () => {
        assert.equal(shouldAutoTranslate('whitelist', null, true, false), true);
        assert.equal(shouldAutoTranslate('whitelist', null, false, false), false);
    });
    await it('manual 不自动翻译', () => {
        assert.equal(shouldAutoTranslate('manual', null, true, false), false);
    });
});

await describe('isSensitiveHost [真源码 content.js — 第3批隐私黑名单]', async () => {
    const f = ct.isSensitiveHost;
    await it('file:// 协议视为敏感', () => assert.equal(f('', 'file:'), true));
    await it('localhost / 本机 / 环回视为敏感', () => {
        assert.equal(f('localhost', 'http:'), true);
        assert.equal(f('foo.localhost', 'http:'), true);
        assert.equal(f('127.0.0.1', 'http:'), true);
        assert.equal(f('127.0.0.2', 'http:'), true);
        assert.equal(f('0.0.0.0', 'http:'), true);
        assert.equal(f('[::1]', 'http:'), true);
    });
    await it('私网 / 链路本地 IPv4 段视为敏感', () => {
        assert.equal(f('192.168.1.1', 'http:'), true);
        assert.equal(f('10.0.0.5', 'http:'), true);
        assert.equal(f('172.16.0.1', 'http:'), true);
        assert.equal(f('172.31.255.255', 'http:'), true);
        assert.equal(f('169.254.1.1', 'http:'), true);
    });
    await it('公网 172.x（非私网段）不敏感', () => {
        assert.equal(f('172.15.0.1', 'http:'), false);
        assert.equal(f('172.32.0.1', 'http:'), false);
    });
    await it('私有/链路本地 IPv6（ULA/link-local）视为敏感', () => {
        assert.equal(f('[fd00::1]', 'http:'), true);
        assert.equal(f('[fc00::1]', 'http:'), true);
        assert.equal(f('[fe80::1]', 'http:'), true);
    });
    await it('.local / .internal 内网域敏感', () => {
        assert.equal(f('myserver.local', 'http:'), true);
        assert.equal(f('api.internal', 'http:'), true);
    });
    await it('含敏感关键词的域名敏感', () => {
        assert.equal(f('mybank.com', 'https:'), true);
        assert.equal(f('login.example.com', 'https:'), true);
        assert.equal(f('accounts.google.com', 'https:'), true);
        assert.equal(f('admin.site.com', 'https:'), true);
    });
    await it('邮箱站点视为敏感（注释承诺覆盖邮箱）', () => {
        assert.equal(f('mail.google.com', 'https:'), true);
        assert.equal(f('gmail.com', 'https:'), true);
        assert.equal(f('outlook.live.com', 'https:'), true);
        assert.equal(f('proton.me', 'https:'), true);
    });
    await it('普通网站不敏感', () => {
        assert.equal(f('example.com', 'https:'), false);
        assert.equal(f('news.ycombinator.com', 'https:'), false);
        assert.equal(f('github.com', 'https:'), false);
    });
    await it('空 hostname + 非 file 协议不敏感', () => assert.equal(f('', 'https:'), false));
});

await describe('语言表单一源派生等价 [真源码 background.js — 第5批, deepEqual 锁死原值]', async () => {
    await it('ALLOWED_TARGET_LANGS 含全部 17 种语言', () => {
        const expected = ['zh-CN','zh-TW','zh','en','ja','ko','fr','de','ru','es','pt','pt-BR','pt-PT','it','ar','th','vi'];
        for (const k of expected) assert.ok(bg.ALLOWED_TARGET_LANGS.has(k), `缺 ${k}`);
        assert.equal(bg.ALLOWED_TARGET_LANGS.size, expected.length);
    });
    await it('DEEPL_LANG_MAP 与原表逐键等价', () => {
        assert.deepEqual(bg.DEEPL_LANG_MAP, {
            'zh-CN': 'ZH-HANS', 'zh-TW': 'ZH-HANT', 'zh': 'ZH-HANS',
            'en': 'EN-US', 'pt': 'PT-BR', 'pt-BR': 'PT-BR', 'pt-PT': 'PT-PT'
        });
    });
    await it('BAIDU_LANG_MAP 与原表逐键等价', () => {
        assert.deepEqual(bg.BAIDU_LANG_MAP, {
            'zh-CN': 'zh', 'zh-TW': 'cht', 'zh': 'zh',
            'en': 'en', 'ja': 'jp', 'ko': 'kor',
            'fr': 'fra', 'de': 'de', 'ru': 'ru',
            'es': 'spa', 'pt': 'pt', 'it': 'it',
            'ar': 'ara', 'th': 'th', 'vi': 'vie'
        });
    });
    await it('LANG_NAMES 与原表逐键等价（含上批补的 ar/th/vi/pt-BR/pt-PT）', () => {
        assert.deepEqual(bg.LANG_NAMES, {
            'zh-CN': '简体中文', 'zh-TW': '繁体中文', 'zh': '中文',
            'en': '英文', 'ja': '日文', 'ko': '韩文',
            'fr': '法文', 'de': '德文', 'ru': '俄文',
            'es': '西班牙文', 'pt': '葡萄牙文', 'pt-BR': '巴西葡萄牙文', 'pt-PT': '葡萄牙文',
            'it': '意大利文', 'ar': '阿拉伯文', 'th': '泰文', 'vi': '越南文'
        });
    });
});

await describe('引擎注册表派生等价 [真源码 background.js — 第5批]', async () => {
    await it('ALLOWED_ENGINES 含全部 9 引擎', () => {
        const expected = ['google_free','google_cloud','deepl','baidu','openai','claude','deepseek','minimax','glm'];
        for (const e of expected) assert.ok(bg.ALLOWED_ENGINES.has(e), `缺 ${e}`);
        assert.equal(bg.ALLOWED_ENGINES.size, expected.length);
    });
    await it('ENGINE_NAMES 与原显示名逐键等价', () => {
        assert.deepEqual(bg.ENGINE_NAMES, {
            google_free: 'Google翻译(免费)', google_cloud: 'Google Cloud',
            deepl: 'DeepL', baidu: '百度翻译', openai: 'OpenAI GPT',
            claude: 'Claude', deepseek: 'DeepSeek', minimax: 'MiniMax', glm: '智谱GLM'
        });
    });
    await it('ENGINE_REGISTRY 的 isLLM 标记正确', () => {
        assert.equal(bg.ENGINE_REGISTRY.openai.isLLM, true);
        assert.equal(bg.ENGINE_REGISTRY.claude.isLLM, true);
        assert.equal(bg.ENGINE_REGISTRY.glm.isLLM, true);
        assert.equal(bg.ENGINE_REGISTRY.google_free.isLLM, false);
        assert.equal(bg.ENGINE_REGISTRY.deepl.isLLM, false);
    });
});

await describe('popup 与 background 语言/引擎选项一致性 [静态解析 popup.html]', async () => {
    const fs = require('node:fs');
    const path = require('node:path');
    const html = fs.readFileSync(path.join(__dirname, '..', 'popup.html'), 'utf8');
    // 抓每个 select 的 option value
    const selectOptions = (id) => {
        const m = html.match(new RegExp(`<select id="${id}"[\\s\\S]*?</select>`));
        assert.ok(m, `popup.html 缺 <select id="${id}">`);
        return [...m[0].matchAll(/<option value="([^"]+)"/g)].map(x => x[1]);
    };
    await it('目标语言下拉的每个 value 都在 ALLOWED_TARGET_LANGS 白名单内', () => {
        const opts = selectOptions('targetLangSelect');
        assert.ok(opts.length >= 15, `语言选项应 ≥15 个，实际 ${opts.length}`);
        for (const v of opts) assert.ok(bg.ALLOWED_TARGET_LANGS.has(v), `popup 语言 ${v} 不在白名单`);
    });
    await it('引擎下拉的每个 value 都在 ALLOWED_ENGINES 白名单内且数量一致', () => {
        const opts = selectOptions('engineSelect');
        for (const v of opts) assert.ok(bg.ALLOWED_ENGINES.has(v), `popup 引擎 ${v} 不在白名单`);
        assert.equal(opts.length, bg.ALLOWED_ENGINES.size, 'popup 引擎数量与白名单不一致');
    });
});

}

// ===== 运行 + 输出 =====
main().then(() => {
    console.log('\n' + '='.repeat(50));
    console.log(`  测试完成: ${passedTests}/${totalTests} 通过`);
    if (failedTests > 0) {
        console.log(`  失败: ${failedTests} 个`);
        failures.forEach(f => console.log(`    - ${f.name}: ${f.error}`));
        process.exit(1);
    } else {
        console.log('  全部通过！');
        process.exit(0);
    }
}).catch(e => {
    console.error('测试运行器异常:', e);
    process.exit(1);
});
