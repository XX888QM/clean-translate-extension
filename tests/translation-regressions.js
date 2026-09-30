// 真源码回归：网络是可控边界；翻译分组、失败判断、缓存键和渲染均运行项目实现。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

module.exports = async function runTranslationRegressions(it) {
    const listeners = [];
    const changed = [];
    const settings = { target_lang: 'zh-CN', translate_engine: 'google_free', api_keys: {} };
    const noop = () => {};
    const event = { addListener: noop };
    const context = vm.createContext({
        console: { log: noop, warn: noop, error: noop },
        module: { exports: {} },
        setTimeout, clearTimeout, setInterval: () => 0, clearInterval: noop,
        URLSearchParams, AbortController, DOMException,
        chrome: {
            runtime: { id: 'test', onMessage: { addListener: f => listeners.push(f) }, onInstalled: event },
            storage: { local: { get: async () => settings }, sync: { get: async () => ({}) },
                onChanged: { addListener: f => changed.push(f) } },
            alarms: { create: noop, onAlarm: event },
            commands: { onCommand: event }, contextMenus: { onClicked: event },
        },
        fetch: async () => { throw new Error('offline'); },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../background.js'), 'utf8'), context);
    const run = expression => vm.runInContext(expression, context);
    const json = value => JSON.parse(JSON.stringify(value));
    const respond = data => ({ ok: true, status: 200, json: async () => data });
    const batch = texts => new Promise(resolve => listeners[0](
        { type: 'TRANSLATE_TEXT_BATCH', texts, targetLang: 'zh-CN', engine: 'google_free' },
        { id: 'test', tab: { id: 1 } }, resolve));

    await it('断网时不能返回原文充当成功译文', async () => {
        const response = await batch(['Hello']);
        assert.equal(response.success, false);
        assert.equal(response.results?.Hello, undefined);
    });
    await it('合法的原文等于译文仍算成功，不能靠字符串相等判失败', async () => {
        context.fetch = async () => respond([[['Hello', 'Hello']]]);
        const response = await batch(['Hello']);
        assert.equal(response.success, true);
        assert.equal(response.results.Hello, 'Hello');
    });
    await it('MyMemory 的业务错误即使 HTTP 200 也不能充当译文', async () => {
        context.fetch = async url => String(url).includes('mymemory')
            ? respond({ responseStatus: 403, responseData: { translatedText: 'INVALID LANGUAGE PAIR' } })
            : { ok: false, status: 503 };
        assert.equal((await batch(['Hello'])).success, false);
    });
    await it('请求使用开始时的语言和引擎，不能被后续全局设置串改', async () => {
        settings.target_lang = 'ja';
        context.fetch = async url => {
            assert.equal(new URL(url).searchParams.get('tl'), 'zh-CN');
            return respond([[['你好', 'Hello']]]);
        };
        const response = await batch(['Hello']);
        assert.equal(response.results.Hello, '你好');
        assert.equal(response.targetLang, 'zh-CN');
        settings.target_lang = 'zh-CN';
    });
    await it('付费引擎降级结果可展示，但不能混入该引擎的缓存', async () => {
        settings.translate_engine = 'deepl';
        context.fetch = async () => respond([[['你好', 'Hello']]]);
        const response = await run('_handleBatchTranslation(["Hello"], {targetLang:"zh-CN",engine:"deepl"})');
        assert.equal(response.results?.Hello, '你好');
        assert.equal(response.cacheableResults?.Hello, undefined);
        assert.equal(response.fallbackUsed, true);
        settings.translate_engine = 'google_free';
    });
    await it('不同引擎缺项时仅返回实际译文，保留部分成功', async () => {
        for (const [call, payload] of [
            ['translateGoogleCloud(["Hello","World"],"en","test")', { data: { translations: [{ translatedText: 'Hi' }] } }],
            ['translateDeepL(["Hello","World"],"en","test")', { translations: [{ text: 'Hi' }] }],
            ['translateBaidu(["Hello","World"],"en","id","test")', { trans_result: [{ src: 'Hello', dst: 'Hi' }] }],
        ]) {
            context.fetch = async () => respond(payload);
            assert.deepEqual(json(await run(call)), { Hello: 'Hi' });
        }
    });
    await it('中断信号不得触发备用翻译请求', async () => {
        let calls = 0;
        context.fetch = async () => { calls++; throw new DOMException('aborted', 'AbortError'); };
        await assert.rejects(run('translateByEngine(["Hello"],"zh-CN","deepl",{deepl:"test"})'), { name: 'AbortError' });
        assert.equal(calls, 1);
    });

    const contentChanges = [];
    const ct = vm.createContext({
        module: { exports: {} }, console, setTimeout, clearTimeout,
        chrome: { runtime: { id: 'test', onMessage: event }, storage: { onChanged: { addListener: f => contentChanges.push(f) } } },
        window: { addEventListener: noop }, document: { addEventListener: noop, getElementById: () => null },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../content.js'), 'utf8'), ct);
    const content = expression => vm.runInContext(expression, ct);
    await it('长文本切分保留全部字符、句间空格和完整 emoji', () => {
        assert.equal(content('typeof splitTranslationText'), 'function');
        ct.source = 'Read this sentence. '.repeat(110) + '🙂'.repeat(800);
        const parts = json(content('splitTranslationText(source)'));
        assert.equal(parts.join(''), ct.source);
        assert.ok(parts.length > 1);
        for (const part of parts) {
            assert.ok(part.length <= 1500);
            assert.ok(!/[\uD800-\uDBFF]$/.test(part));
            assert.ok(!/^[\uDC00-\uDFFF]/.test(part));
        }
        ct.source = 'Sentence one is short. '.repeat(60) + 'word '.repeat(60);
        assert.ok(content('splitTranslationText(source)[0]').endsWith('. '), '优先在完整句子后切分');
    });
    await it('缓存键区分语言和引擎，原文中的分隔符不能碰撞', () => {
        assert.equal(content('typeof translationCacheKey'), 'function');
        const keys = content('[translationCacheKey("Hello", "zh-CN", "google_free"), translationCacheKey("Hello", "ja", "google_free"), translationCacheKey("Hello", "zh-CN", "deepl"), translationCacheKey("Hello|ja", "zh-CN", "deepl")]');
        assert.equal(new Set(keys).size, 4);
        assert.ok(keys.every(k => k !== 'Hello'));
    });
    await it('双语反复开关不累加原文，纯显示操作不调用翻译接口', () => {
        content('globalThis.node = {nodeValue:"  Hello  "}; originalTextMap.set(node,"  Hello  "); bilingualMode = true; applyTextToNode(node,"你好");');
        assert.equal(ct.node.nodeValue, '  你好\nHello  ');
        content('applyTextToNode(node,"你好")');
        assert.equal(ct.node.nodeValue, '  你好\nHello  ');
        content('bilingualMode = false; applyTextToNode(node,"你好")');
        assert.equal(ct.node.nodeValue, '  你好  ');
    });
    await it('划词成功结果提供复制按钮，复制内容只有译文', async () => {
        function element(tag) {
            return { tagName: tag, children: [], style: {}, dataset: {},
                appendChild(child) { this.children.push(child); },
                setAttribute() {}, addEventListener(name, handler) { this[name] = handler; } };
        }
        ct.document.createElement = element;
        ct.container = element('div');
        let copied;
        ct.navigator = { clipboard: { writeText: async text => { copied = text; } } };
        content('appendEngineResult(container,"测试引擎","你好",null)');
        const buttons = ct.container.children[0].children.filter(child => child.tagName === 'button');
        assert.equal(buttons.length, 1);
        await buttons[0].click({ isTrusted: true });
        assert.equal(copied, '你好');
    });
    await it('重新翻译返回与原文相同的结果时，要移除旧译文并保持可还原', () => {
        content('globalThis.item = {nodeValue:"Hello"}; originalTextMap.set(item,"Hello"); applyTextToNode(item,"你好"); globalThis.items = new Map([["Hello",[item,{__isTitle:true}]]]);');
        content('applyBatchTranslations({Hello:"Hello"}, items); applyBatchSpecialTranslations({Hello:"Hello"}, items)');
        assert.equal(ct.item.nodeValue, 'Hello');
        assert.equal(ct.document.title, 'Hello');
    });
    await it('气泡自身滚动不能关闭气泡，页面滚动仍关闭', () => {
        const events = {};
        ct.document.addEventListener = (name, handler) => { events[name] = handler; };
        content('initSelectionTranslate()');
        let closed = 0;
        ct.popup = { contains: target => target === ct.popup, classList: { remove: () => { closed++; } } };
        content('selectionPopup = popup');
        events.scroll({ target: ct.popup });
        assert.equal(closed, 0);
        events.scroll({ target: ct.document });
        assert.equal(closed, 1);
        assert.equal(ct.popup.hidden, true, '关闭后的气泡不能继续被键盘或读屏访问');
        content('selectionPopup = null');
    });
    // 单节点 DOM 边界替身：保留真实 _doTranslation、缓存、合并与回写，UI 单独在浏览器检查。
    ct.document.body = {};
    ct.document.title = '';
    ct.settings = { target_lang: 'zh-CN', translate_engine: 'google_free', bilingual_mode: false };
    ct.chrome.storage.local = { get: async () => ({ ...ct.settings }) };
    content('getTextNodes = () => [node]; getTranslatableAttrs = () => []; showToast = () => {}; showProgressBar = () => {}; updateProgress = () => {}; hideProgressBar = () => {};');
    await it('强制重译失败后的重试不能使用上一份旧缓存冒充完成', async () => {
        content('translationCache.clear(); node = {nodeValue:"Fresh sentence"}; originalTextMap.set(node,"Fresh sentence");');
        let calls = 0;
        ct.sendMessageWithRetry = async request => {
            if (request.type !== 'TRANSLATE_TEXT_BATCH') return { success: true, results: {} };
            calls++;
            return calls === 1
                ? { success: true, targetLang: 'zh-CN', engine: 'google_free', results: { 'Fresh sentence': '新句子' }, cacheableResults: { 'Fresh sentence': '新句子' } }
                : { success: false, results: {} };
        };
        assert.equal((await content('performTranslation()')).completed, 1);
        assert.equal((await content('performTranslation(document.body, false, true)')).failed, 1);
        assert.equal((await content('performTranslation()')).failed, 1);
        assert.equal(calls, 3);
    });
    await it('部分失败重试保留已成功的降级译文，但不缓存且强制重译仍重新请求', async () => {
        content('restoreOriginal(true); translationCache.clear(); globalThis.trialNodes = [{nodeValue:"First source"},{nodeValue:"Second source"}]; getTextNodes = () => trialNodes;');
        ct.settings.translate_engine = 'deepl';
        const requests = [];
        ct.sendMessageWithRetry = async request => {
            if (request.type !== 'TRANSLATE_TEXT_BATCH') return { success: true, results: {} };
            requests.push(json(request.texts));
            return { success: true, targetLang: 'zh-CN', engine: 'deepl', fallbackUsed: true,
                results: requests.length === 1 ? { 'First source': '第一条备用译文' } : { 'Second source': '第二条备用译文' },
                cacheableResults: {} };
        };
        try {
            assert.equal((await content('performTranslation()')).failed, 1);
            const retried = await content('performTranslation()');
            assert.deepEqual(requests, [['First source', 'Second source'], ['Second source']]);
            assert.equal(retried.completed, 2);
            assert.equal(retried.failed, 0);
            assert.equal(retried.fallbackUsed, true);
            assert.equal(content('translationCache.size'), 0);
            assert.deepEqual(json(ct.trialNodes.map(node => node.nodeValue)), ['第一条备用译文', '第二条备用译文']);
            await content('performTranslation(document.body, false, true)');
            assert.deepEqual(requests[2], ['First source', 'Second source']);
            content('restoreOriginal(true)');
            await content('performTranslation()');
            assert.deepEqual(requests[3], ['First source', 'Second source']);
        } finally {
            content('restoreOriginal(true); getTextNodes = () => [node];');
            ct.settings.translate_engine = 'google_free';
        }
    });
    await it('请求在途时清缓存，晚到结果不得恢复重试暂存', async () => {
        content('restoreOriginal(true); translationCache.clear(); trialNodes = [{nodeValue:"First source"},{nodeValue:"Second source"}]; getTextNodes = () => trialNodes;');
        ct.settings.translate_engine = 'deepl';
        const requests = [];
        let release, started;
        const waiting = new Promise(resolve => { release = resolve; });
        const start = new Promise(resolve => { started = resolve; });
        ct.sendMessageWithRetry = async request => {
            if (request.type !== 'TRANSLATE_TEXT_BATCH') return { success: true, results: {} };
            requests.push(json(request.texts));
            if (requests.length === 1) { started(); await waiting; }
            return { success: true, targetLang: 'zh-CN', engine: 'deepl', fallbackUsed: true,
                results: { 'First source': '第一条备用译文' }, cacheableResults: {} };
        };
        try {
            const first = content('performTranslation()');
            await start;
            content('clearCache()');
            release(); await first;
            await content('performTranslation()');
            assert.deepEqual(requests[1], ['First source', 'Second source']);
        } finally {
            content('restoreOriginal(true); getTextNodes = () => [node];');
            ct.settings.translate_engine = 'google_free';
        }
    });
    await it('切换语言时的旧请求结果不能回写页面或写入旧缓存', async () => {
        content('translationCache.clear(); node = {nodeValue:"Pending sentence"}; originalTextMap.set(node,"Pending sentence");');
        let release;
        const pending = new Promise(resolve => { release = resolve; });
        let started;
        const start = new Promise(resolve => { started = resolve; });
        ct.sendMessageWithRetry = async request => {
            if (request.type !== 'TRANSLATE_TEXT_BATCH') return { success: true, results: {} };
            started();
            return pending;
        };
        const task = content('performTranslation()');
        await start;
        contentChanges[0]({ target_lang: { oldValue: 'zh-CN', newValue: 'ja' } }, 'local');
        release({ success: true, targetLang: 'zh-CN', engine: 'google_free', results: { 'Pending sentence': '旧译文' }, cacheableResults: { 'Pending sentence': '旧译文' } });
        assert.equal((await task).cancelled, true);
        assert.equal(ct.node.nodeValue, 'Pending sentence');
        assert.equal(content('translationCache.size'), 0);
    });
    await it('设置切换取消翻译时，页面不能残留翻译中的提示', () => {
        let visible = true;
        ct.document.getElementById = id => id === 'yx-toast-container'
            ? { classList: { remove: () => { visible = false; } } } : null;
        content('restoreOriginal(true)');
        assert.equal(visible, false);
    });
    await it('popup 切换语言或引擎后不能残留旧完成状态和重试按钮', () => {
        const elements = new Map();
        const get = id => {
            if (!elements.has(id)) elements.set(id, { value: '', style: {}, hidden: false, classList: { toggle() {} },
                addEventListener(name, fn) { this[name] = fn; } });
            return elements.get(id);
        };
        const local = { get: (keys, cb) => cb({}), set: (items, cb) => cb?.() };
        vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../popup.js'), 'utf8'), {
            document: { getElementById: get }, console, setTimeout, clearTimeout,
            chrome: { runtime: { sendMessage: (msg, cb) => cb?.(), onMessage: event },
                tabs: { query: (q, cb) => cb([]) }, storage: { local } },
        });
        for (const id of ['targetLangSelect', 'engineSelect']) {
            get('status').textContent = '翻译完成';
            get('retryBtn').hidden = false;
            get(id).value = id === 'targetLangSelect' ? 'ja' : 'deepl';
            get(id).change();
            assert.notEqual(get('status').textContent, '翻译完成');
            assert.equal(get('retryBtn').hidden, true);
        }
    });
};
