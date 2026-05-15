#!/usr/bin/env node
// YX纯净网页翻译 — 轻量测试运行器
// 用法: node tests/run-tests.js

const assert = require('node:assert/strict');

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;
const failures = [];

function describe(name, fn) {
    console.log(`\n  ${name}`);
    fn();
}

function it(name, fn) {
    totalTests++;
    try {
        fn();
        passedTests++;
        console.log(`    ✓ ${name}`);
    } catch (e) {
        failedTests++;
        console.log(`    ✗ ${name}`);
        console.log(`      ${e.message}`);
        failures.push({ name, error: e.message });
    }
}

// ===== 从源码中提取的纯函数（用于测试） =====

const MIN_TEXT_LENGTH = 2;
const IGNORED_TAGS = new Set([
    'SCRIPT', 'STYLE', 'NOSCRIPT', 'TEXTAREA', 'INPUT', 'PRE', 'CODE',
    'KBD', 'SAMP', 'VAR', 'IFRAME', 'IMG', 'SVG', 'PATH', 'METADATA'
]);

// 模拟 isTranslatable 的核心判断逻辑（不含 DOM 依赖）
function isTextTranslatable(text, parentTagName, parentClassName, isEditable) {
    if (IGNORED_TAGS.has(parentTagName)) return false;
    if (isEditable) return false;

    if (parentClassName && typeof parentClassName === 'string') {
        const cls = parentClassName.toLowerCase();
        if (cls.includes('material-icons') || cls.includes('material-symbols') ||
            cls.includes('fa-') || cls.includes('icon') || cls.includes('glyph')) {
            return false;
        }
    }

    const trimmed = text.trim();
    if (trimmed.length < MIN_TEXT_LENGTH) return false;
    if (/^\d+$/.test(trimmed)) return false;
    if (/^[^\p{L}]+$/u.test(trimmed)) return false;
    if (/^\{.*\}$/.test(trimmed) || /^[A-Z0-9_]+$/.test(trimmed)) return false;
    if (/^[a-z0-9]+(_[a-z0-9]+)+$/.test(trimmed)) return false;

    return true;
}

// LRU 缓存
function createLRUCache(maxSize) {
    const cache = new Map();
    return {
        get(key) {
            if (cache.has(key)) {
                const value = cache.get(key);
                cache.delete(key);
                cache.set(key, value);
                return value;
            }
            return undefined;
        },
        set(key, value) {
            if (cache.has(key)) cache.delete(key);
            cache.set(key, value);
            if (cache.size > maxSize) {
                const firstKey = cache.keys().next().value;
                cache.delete(firstKey);
            }
        },
        size() { return cache.size; },
        clear() { cache.clear(); },
        entries() { return Array.from(cache.entries()); }
    };
}

// 中文字符占比计算
function calculateChineseRatio(text) {
    if (!text || text.length === 0) return 0;
    const chineseChars = text.match(/[\u4e00-\u9fa5]/g) || [];
    return chineseChars.length / text.length;
}

// 翻译模式决策逻辑
function shouldAutoTranslate(mode, sitePref, isInWhitelist, isInExcludeList) {
    // 网站偏好优先级最高
    if (sitePref === 'never') return false;
    if (sitePref === 'auto') return true;

    // 根据模式决定
    switch (mode) {
        case 'auto_all':
            return !isInExcludeList;
        case 'whitelist':
            return isInWhitelist;
        case 'manual':
            return false;
        default:
            return true;
    }
}

// ===== 测试用例 =====

describe('isTextTranslatable', () => {
    it('应该翻译正常英文文本', () => {
        assert.equal(isTextTranslatable('Hello World', 'DIV', '', false), true);
    });

    it('应该跳过 SCRIPT 标签内文本', () => {
        assert.equal(isTextTranslatable('var x = 1', 'SCRIPT', '', false), false);
    });

    it('应该跳过 STYLE 标签内文本', () => {
        assert.equal(isTextTranslatable('.cls { color: red }', 'STYLE', '', false), false);
    });

    it('应该跳过 CODE 标签内文本', () => {
        assert.equal(isTextTranslatable('console.log()', 'CODE', '', false), false);
    });

    it('应该跳过 contentEditable 元素', () => {
        assert.equal(isTextTranslatable('editable text', 'DIV', '', true), false);
    });

    it('应该跳过 icon class 元素', () => {
        assert.equal(isTextTranslatable('arrow_back', 'SPAN', 'material-icons', false), false);
        assert.equal(isTextTranslatable('home', 'I', 'fa-icon', false), false);
    });

    it('应该跳过太短的文本', () => {
        assert.equal(isTextTranslatable('a', 'DIV', '', false), false);
    });

    it('应该跳过纯数字', () => {
        assert.equal(isTextTranslatable('12345', 'DIV', '', false), false);
    });

    it('应该跳过纯符号', () => {
        assert.equal(isTextTranslatable('---', 'DIV', '', false), false);
        assert.equal(isTextTranslatable('***', 'DIV', '', false), false);
    });

    it('应该跳过 JSON 格式文本', () => {
        assert.equal(isTextTranslatable('{"key":"value"}', 'DIV', '', false), false);
    });

    it('应该跳过全大写常量', () => {
        assert.equal(isTextTranslatable('MAX_SIZE', 'DIV', '', false), false);
        assert.equal(isTextTranslatable('API_KEY_NAME', 'DIV', '', false), false);
    });

    it('应该跳过 snake_case 字符串', () => {
        assert.equal(isTextTranslatable('keyboard_arrow_down', 'DIV', '', false), false);
    });

    it('应该翻译混合文本', () => {
        assert.equal(isTextTranslatable('Hello 123 World', 'DIV', '', false), true);
    });
});

describe('LRU 缓存', () => {
    it('应该正确存取值', () => {
        const cache = createLRUCache(5);
        cache.set('a', '翻译A');
        assert.equal(cache.get('a'), '翻译A');
    });

    it('不存在的键应返回 undefined', () => {
        const cache = createLRUCache(5);
        assert.equal(cache.get('missing'), undefined);
    });

    it('超出容量时应淘汰最旧条目', () => {
        const cache = createLRUCache(3);
        cache.set('a', '1');
        cache.set('b', '2');
        cache.set('c', '3');
        cache.set('d', '4'); // 'a' 应被淘汰
        assert.equal(cache.get('a'), undefined);
        assert.equal(cache.get('b'), '2');
        assert.equal(cache.get('d'), '4');
    });

    it('访问后应移到最近使用（不被淘汰）', () => {
        const cache = createLRUCache(3);
        cache.set('a', '1');
        cache.set('b', '2');
        cache.set('c', '3');
        cache.get('a'); // 访问 'a'，使其变为最近使用
        cache.set('d', '4'); // 'b' 应被淘汰（而非 'a'）
        assert.equal(cache.get('a'), '1');
        assert.equal(cache.get('b'), undefined);
    });

    it('覆盖已有键应更新值', () => {
        const cache = createLRUCache(3);
        cache.set('a', '旧值');
        cache.set('a', '新值');
        assert.equal(cache.get('a'), '新值');
        assert.equal(cache.size(), 1);
    });

    it('clear 应清空缓存', () => {
        const cache = createLRUCache(5);
        cache.set('a', '1');
        cache.set('b', '2');
        cache.clear();
        assert.equal(cache.size(), 0);
        assert.equal(cache.get('a'), undefined);
    });
});

describe('calculateChineseRatio', () => {
    it('纯中文应返回 1', () => {
        assert.equal(calculateChineseRatio('你好世界'), 1);
    });

    it('纯英文应返回 0', () => {
        assert.equal(calculateChineseRatio('Hello World'), 0);
    });

    it('混合文本应返回正确占比', () => {
        const ratio = calculateChineseRatio('你好Hello');
        // 2个中文 / 7个总字符 ≈ 0.2857
        assert.ok(ratio > 0.28 && ratio < 0.29);
    });

    it('空文本应返回 0', () => {
        assert.equal(calculateChineseRatio(''), 0);
        assert.equal(calculateChineseRatio(null), 0);
    });

    it('中文占比 > 0.3 应判定为中文页面', () => {
        const ratio = calculateChineseRatio('这是一个中文页面的示例文本内容abc');
        assert.ok(ratio > 0.3);
    });
});

describe('shouldAutoTranslate（翻译模式决策）', () => {
    it('网站偏好 never 应始终不翻译', () => {
        assert.equal(shouldAutoTranslate('auto_all', 'never', false, false), false);
        assert.equal(shouldAutoTranslate('manual', 'never', false, false), false);
    });

    it('网站偏好 auto 应始终翻译', () => {
        assert.equal(shouldAutoTranslate('manual', 'auto', false, false), true);
        assert.equal(shouldAutoTranslate('whitelist', 'auto', false, false), true);
    });

    it('auto_all 模式应翻译非排除域名', () => {
        assert.equal(shouldAutoTranslate('auto_all', null, false, false), true);
        assert.equal(shouldAutoTranslate('auto_all', null, false, true), false);
    });

    it('whitelist 模式应只翻译白名单域名', () => {
        assert.equal(shouldAutoTranslate('whitelist', null, true, false), true);
        assert.equal(shouldAutoTranslate('whitelist', null, false, false), false);
    });

    it('manual 模式应不自动翻译', () => {
        assert.equal(shouldAutoTranslate('manual', null, false, false), false);
        assert.equal(shouldAutoTranslate('manual', null, true, false), false);
    });
});

// ===== 输出结果 =====
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
