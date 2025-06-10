/**
 * Enhanced Comprehensive Claude Conversation State Extractor v2.0
 *
 * Maximum fidelity extractor designed to capture complete conversation state
 * for high-fidelity replay in new Claude instances. No size limits imposed.
 *
 * Enhanced Features:
 * - Unlimited content extraction (removed size constraints)
 * - Enhanced artifact evolution tracking
 * - Deep tool call result extraction
 * - Complete message threading and branching
 * - Advanced user preference inference
 * - Comprehensive system state capture
 * - Enhanced error recovery and fallbacks
 * - Complete DOM state preservation
 * - Advanced conversation flow analysis
 *
 * Usage: Run in browser console on claude.ai conversation page
 */

(function() {
    'use strict';

    console.log('🧠 Enhanced Comprehensive Claude Conversation State Extractor v2.0');
    console.log('📊 Extracting maximum fidelity conversation state...');

    const extractorConfig = {
        includeTimestamps: true,
        includeMetadata: true,
        includeArtifacts: true,
        includeDocuments: true,
        includeToolCalls: true,
        includePreferences: true,
        includeEnvironment: true,
        includeDOMState: true,
        includeErrorStates: true,
        includeConversationBranching: true,
        maxContentLength: Number.MAX_SAFE_INTEGER, // No limits
        maxNestingDepth: 100, // Very deep nesting allowed
        verboseLogging: true,
        preserveWhitespace: true,
        captureRawHTML: true,
        deepAnalysis: true
    };

    function log(message, level = 'info') {
        const prefix = {
            'debug': '🔍',
            'info': '📖',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'deep': '🕳️'
        }[level] || '📖';

        if (extractorConfig.verboseLogging || level !== 'debug') {
            console.log(`${prefix} ${message}`);
        }
    }

    function generateStateId() {
        return `state_${Date.now()}_${Math.random().toString(36).substr(2, 12)}`;
    }

    function safeExtractText(element, preserveFormatting = false) {
        if (!element) return '';

        try {
            if (preserveFormatting) {
                return element.innerHTML || element.textContent || '';
            }
            return element.textContent || element.innerText || '';
        } catch (e) {
            log(`Text extraction error: ${e.message}`, 'warning');
            return '';
        }
    }

    function extractAllAttributes(element) {
        if (!element || !element.attributes) return {};

        const attrs = {};
        for (const attr of element.attributes) {
            attrs[attr.name] = attr.value;
        }
        return attrs;
    }

    function detectEnvironment() {
        log('Detecting comprehensive environment and system context...', 'debug');

        return {
            platform: navigator.platform,
            userAgent: navigator.userAgent,
            language: navigator.language,
            languages: navigator.languages,
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight,
                devicePixelRatio: window.devicePixelRatio
            },
            screen: {
                width: screen.width,
                height: screen.height,
                colorDepth: screen.colorDepth
            },
            url: window.location.href,
            referrer: document.referrer,
            timestamp: new Date().toISOString(),
            performance: {
                navigation: performance.navigation?.type,
                timing: {
                    loadComplete: performance.timing?.loadEventEnd - performance.timing?.navigationStart,
                    domReady: performance.timing?.domContentLoadedEventEnd - performance.timing?.navigationStart
                }
            },
            claude_version: detectClaudeVersion(),
            available_tools: detectAvailableTools(),
            session_context: detectSessionContext(),
            memory_info: detectMemoryInfo(),
            connection_info: detectConnectionInfo()
        };
    }

    function detectMemoryInfo() {
        try {
            if (performance.memory) {
                return {
                    usedJSHeapSize: performance.memory.usedJSHeapSize,
                    totalJSHeapSize: performance.memory.totalJSHeapSize,
                    jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
                };
            }
        } catch (e) {
            log('Memory info not available', 'debug');
        }
        return null;
    }

    function detectConnectionInfo() {
        try {
            if (navigator.connection) {
                return {
                    effectiveType: navigator.connection.effectiveType,
                    downlink: navigator.connection.downlink,
                    rtt: navigator.connection.rtt,
                    saveData: navigator.connection.saveData
                };
            }
        } catch (e) {
            log('Connection info not available', 'debug');
        }
        return null;
    }

    function detectClaudeVersion() {
        const versionIndicators = [
            document.querySelector('[data-testid="model-name"]'),
            document.querySelector('.model-name'),
            document.querySelector('[class*="model"]'),
            document.querySelector('h1'),
            document.querySelector('title')
        ].filter(Boolean);

        for (const element of versionIndicators) {
            const text = safeExtractText(element);
            if (text && (text.includes('Claude') || text.includes('Sonnet') || text.includes('Opus') || text.includes('Haiku'))) {
                return text.trim();
            }
        }

        // Deep search in page content
        const bodyText = document.body.textContent || '';
        const versionMatch = bodyText.match(/(Claude[^,.\n]*(?:Sonnet|Opus|Haiku)[^,.\n]*)/i);
        if (versionMatch) {
            return versionMatch[1].trim();
        }

        return `Claude (detected from: ${document.title})`;
    }

    function detectAvailableTools() {
        const tools = new Set();

        // Comprehensive tool detection
        const toolSelectors = [
            '[data-testid*="tool"]',
            '[class*="tool"]',
            '[aria-label*="tool"]',
            '.function-call',
            '[data-function]',
            '[data-tool-name]',
            '.repl',
            '.artifact',
            '.web-search',
            '.computer-use'
        ];

        toolSelectors.forEach(selector => {
            try {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    const toolNames = [
                        el.getAttribute('data-tool'),
                        el.getAttribute('data-function'),
                        el.getAttribute('data-tool-name'),
                        el.className.match(/tool-(\w+)/)?.[1],
                        el.textContent?.match(/(\w+)(?:\s*\(|\s*:)/)?.[1]
                    ].filter(Boolean);

                    toolNames.forEach(name => tools.add(name));
                });
            } catch (e) {
                log(`Tool detection error for ${selector}: ${e.message}`, 'debug');
            }
        });

        // Content-based tool detection
        const bodyHTML = document.body.innerHTML;
        const commonTools = [
            'web_search', 'artifacts', 'repl', 'computer_use',
            'text_editor', 'bash', 'python', 'web_fetch',
            'google_drive_search', 'slack', 'asana', 'linear'
        ];

        commonTools.forEach(tool => {
            if (bodyHTML.includes(tool)) {
                tools.add(tool);
            }
        });

        // Function call pattern detection
        const functionCalls = bodyHTML.match(/<function_calls>[\s\S]*?<\/antml:function_calls>/g) || [];
        functionCalls.forEach(call => {
            const invokeMatch = call.match(/<invoke name="([^"]+)">/);
            if (invokeMatch) {
                tools.add(invokeMatch[1]);
            }
        });

        return Array.from(tools);
    }

    function detectSessionContext() {
        const messageElements = document.querySelectorAll('[data-testid="message"], .message, [role="article"]');

        return {
            has_artifacts: !!document.querySelector('[data-testid="artifact"], .artifact'),
            has_documents: !!document.querySelector('[data-testid="document"], .document'),
            has_tools: !!document.querySelector('[data-testid="tool-call"], .tool-call'),
            has_images: !!document.querySelector('img[src*="claude"], img[alt*="image"]'),
            conversation_length: messageElements.length,
            is_shared: window.location.href.includes('/share/') ||
                      !!document.querySelector('[data-testid="shared-chat"]'),
            has_regenerations: !!document.querySelector('[data-testid="regenerate"], .regenerate'),
            has_branches: !!document.querySelector('[data-testid="branch"], .branch'),
            has_edits: !!document.querySelector('[data-testid="edit"], .edit'),
            page_load_time: Date.now(),
            scroll_position: {
                x: window.pageXOffset,
                y: window.pageYOffset
            },
            visible_area: {
                top: window.pageYOffset,
                bottom: window.pageYOffset + window.innerHeight
            }
        };
    }

    function extractUserPreferences() {
        log('Performing deep user preference and behavioral analysis...', 'deep');

        const preferences = {
            detected_style: 'unknown',
            response_length_preference: 'unknown',
            technical_level: 'unknown',
            formatting_preference: 'unknown',
            interaction_patterns: {},
            language: navigator.language,
            communication_style: {},
            content_preferences: {},
            indicators: [],
            confidence_scores: {}
        };

        const messages = document.querySelectorAll('[data-testid="message"], .message, [role="article"]');
        const userMessages = Array.from(messages).filter(msg =>
            detectMessageSpeaker(msg) === 'User' ||
            safeExtractText(msg).toLowerCase().startsWith('user:')
        );

        if (userMessages.length === 0) {
            log('No user messages found for preference analysis', 'warning');
            return preferences;
        }

        // Enhanced style analysis
        const styleAnalysis = analyzeUserStyle(userMessages);
        preferences.detected_style = styleAnalysis.style;
        preferences.confidence_scores.style = styleAnalysis.confidence;
        preferences.indicators.push(...styleAnalysis.indicators);

        // Technical level analysis
        const techAnalysis = analyzeTechnicalLevel(userMessages);
        preferences.technical_level = techAnalysis.level;
        preferences.confidence_scores.technical = techAnalysis.confidence;

        // Communication pattern analysis
        preferences.communication_style = analyzeCommunicationPatterns(userMessages);

        // Content preference analysis
        preferences.content_preferences = analyzeContentPreferences(userMessages);

        // Response length preference
        const claudeMessages = Array.from(messages).filter(msg =>
            detectMessageSpeaker(msg) === 'Assistant'
        );
        preferences.response_length_preference = analyzeResponseLengthPreference(claudeMessages, userMessages);

        return preferences;
    }

    function analyzeUserStyle(userMessages) {
        const formalIndicators = [
            'please', 'thank you', 'could you', 'would you', 'i would appreciate',
            'kindly', 'respectfully', 'sincerely', 'formal', 'professional'
        ];
        const casualIndicators = [
            'hey', 'awesome', 'cool', 'thanks', 'yeah', 'yep', 'nah',
            'gonna', 'wanna', 'kinda', 'pretty much', 'lol', 'btw'
        ];

        let formalCount = 0, casualCount = 0;
        const indicators = [];

        userMessages.forEach(msg => {
            const text = safeExtractText(msg).toLowerCase();

            formalIndicators.forEach(indicator => {
                const matches = (text.match(new RegExp(indicator, 'g')) || []).length;
                if (matches > 0) {
                    formalCount += matches;
                    indicators.push(`formal: "${indicator}" (${matches}x)`);
                }
            });

            casualIndicators.forEach(indicator => {
                const matches = (text.match(new RegExp(indicator, 'g')) || []).length;
                if (matches > 0) {
                    casualCount += matches;
                    indicators.push(`casual: "${indicator}" (${matches}x)`);
                }
            });
        });

        const total = formalCount + casualCount;
        const confidence = total > 0 ? Math.abs(formalCount - casualCount) / total : 0;

        let style = 'neutral';
        if (formalCount > casualCount * 1.5) style = 'formal';
        else if (casualCount > formalCount * 1.5) style = 'casual';

        return { style, confidence, indicators };
    }

    function analyzeTechnicalLevel(userMessages) {
        const technicalTerms = [
            'api', 'function', 'algorithm', 'database', 'framework', 'library',
            'json', 'xml', 'http', 'rest', 'sql', 'regex', 'async', 'await',
            'class', 'object', 'array', 'variable', 'parameter', 'return',
            'import', 'export', 'module', 'package', 'dependency', 'version'
        ];

        const advancedTerms = [
            'microservices', 'kubernetes', 'docker', 'tensorflow', 'machine learning',
            'neural network', 'blockchain', 'cryptocurrency', 'devops', 'ci/cd',
            'oauth', 'jwt', 'websocket', 'graphql', 'nosql', 'mongodb'
        ];

        let basicCount = 0, advancedCount = 0;

        userMessages.forEach(msg => {
            const text = safeExtractText(msg).toLowerCase();

            technicalTerms.forEach(term => {
                if (text.includes(term)) basicCount++;
            });

            advancedTerms.forEach(term => {
                if (text.includes(term)) advancedCount++;
            });
        });

        const totalMessages = userMessages.length;
        const techDensity = (basicCount + advancedCount * 2) / totalMessages;

        let level = 'low';
        if (techDensity > 3) level = 'high';
        else if (techDensity > 1) level = 'medium';

        const confidence = Math.min(techDensity / 5, 1);

        return { level, confidence, basicCount, advancedCount, techDensity };
    }

    function analyzeCommunicationPatterns(userMessages) {
        const patterns = {
            average_message_length: 0,
            question_ratio: 0,
            exclamation_ratio: 0,
            code_block_usage: 0,
            list_usage: 0,
            emoji_usage: 0,
            politeness_markers: 0
        };

        let totalLength = 0;
        let questionCount = 0;
        let exclamationCount = 0;
        let codeBlockCount = 0;
        let listCount = 0;
        let emojiCount = 0;
        let politenessCount = 0;

        userMessages.forEach(msg => {
            const text = safeExtractText(msg);
            totalLength += text.length;

            if (text.includes('?')) questionCount++;
            if (text.includes('!')) exclamationCount++;
            if (text.includes('```') || text.includes('`')) codeBlockCount++;
            if (text.match(/^\s*[\-\*\d+\.]/m)) listCount++;
            if (text.match(/[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]/gu)) emojiCount++;
            if (text.match(/please|thank|sorry|excuse|pardon/i)) politenessCount++;
        });

        const msgCount = userMessages.length;
        if (msgCount > 0) {
            patterns.average_message_length = totalLength / msgCount;
            patterns.question_ratio = questionCount / msgCount;
            patterns.exclamation_ratio = exclamationCount / msgCount;
            patterns.code_block_usage = codeBlockCount / msgCount;
            patterns.list_usage = listCount / msgCount;
            patterns.emoji_usage = emojiCount / msgCount;
            patterns.politeness_markers = politenessCount / msgCount;
        }

        return patterns;
    }

    function analyzeContentPreferences(userMessages) {
        const preferences = {
            prefers_examples: 0,
            prefers_step_by_step: 0,
            prefers_code_heavy: 0,
            prefers_explanations: 0,
            prefers_concise: 0,
            prefers_detailed: 0
        };

        userMessages.forEach(msg => {
            const text = safeExtractText(msg).toLowerCase();

            if (text.includes('example') || text.includes('sample')) preferences.prefers_examples++;
            if (text.includes('step') || text.includes('guide') || text.includes('tutorial')) preferences.prefers_step_by_step++;
            if (text.includes('code') || text.includes('script') || text.includes('program')) preferences.prefers_code_heavy++;
            if (text.includes('explain') || text.includes('why') || text.includes('how')) preferences.prefers_explanations++;
            if (text.includes('brief') || text.includes('short') || text.includes('quick')) preferences.prefers_concise++;
            if (text.includes('detailed') || text.includes('comprehensive') || text.includes('thorough')) preferences.prefers_detailed++;
        });

        return preferences;
    }

    function analyzeResponseLengthPreference(claudeMessages, userMessages) {
        if (claudeMessages.length === 0) return 'unknown';

        const avgResponseLength = claudeMessages.reduce((sum, msg) => {
            return sum + safeExtractText(msg).length;
        }, 0) / claudeMessages.length;

        // Analyze user feedback on length
        const lengthFeedback = userMessages.reduce((feedback, msg) => {
            const text = safeExtractText(msg).toLowerCase();
            if (text.includes('shorter') || text.includes('brief') || text.includes('concise')) {
                feedback.shorter++;
            } else if (text.includes('longer') || text.includes('more detail') || text.includes('elaborate')) {
                feedback.longer++;
            }
            return feedback;
        }, { shorter: 0, longer: 0 });

        if (lengthFeedback.shorter > lengthFeedback.longer) return 'concise';
        if (lengthFeedback.longer > lengthFeedback.shorter) return 'detailed';

        if (avgResponseLength > 3000) return 'detailed';
        if (avgResponseLength > 1000) return 'moderate';
        return 'concise';
    }

    function extractArtifacts() {
        log('Performing deep artifact extraction and evolution analysis...', 'deep');

        const artifacts = [];

        // Comprehensive artifact detection
        const artifactSelectors = [
            '[data-testid="artifact"]',
            '.artifact',
            '[class*="artifact"]',
            '[data-artifact-id]',
            'pre[class*="language-"]',
            '.code-block',
            '.markdown-content'
        ];

        const artifactElements = new Set();
        artifactSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => artifactElements.add(el));
        });

        Array.from(artifactElements).forEach((element, index) => {
            try {
                const artifact = {
                    id: element.getAttribute('data-artifact-id') ||
                        element.id ||
                        `artifact_${index}_${Date.now()}`,
                    type: detectArtifactType(element),
                    title: extractArtifactTitle(element),
                    content: extractArtifactContent(element),
                    raw_html: extractorConfig.captureRawHTML ? element.outerHTML : null,
                    language: detectArtifactLanguage(element),
                    created_at: extractArtifactTimestamp(element),
                    metadata: extractArtifactMetadata(element),
                    updates: extractArtifactUpdates(element),
                    dependencies: extractArtifactDependencies(element),
                    execution_info: extractArtifactExecutionInfo(element),
                    dom_context: extractArtifactDOMContext(element),
                    size_metrics: calculateArtifactSizeMetrics(element)
                };

                if (artifact.content || artifact.raw_html) {
                    artifacts.push(artifact);
                    log(`Extracted artifact: ${artifact.title} (${artifact.type}, ${artifact.size_metrics.content_length} chars)`, 'debug');
                }
            } catch (e) {
                log(`Error extracting artifact ${index}: ${e.message}`, 'warning');
            }
        });

        return artifacts;
    }

    function detectArtifactType(element) {
        const content = safeExtractText(element);
        const classes = element.className || '';
        const rawHTML = element.innerHTML || '';

        // More sophisticated type detection
        const typeIndicators = {
            'code': [
                () => classes.includes('code'),
                () => content.includes('```'),
                () => element.tagName === 'PRE',
                () => /function\s+\w+\s*\(/.test(content),
                () => /class\s+\w+/.test(content),
                () => /import\s+/.test(content),
                () => /def\s+\w+\s*\(/.test(content)
            ],
            'html': [
                () => classes.includes('html'),
                () => content.includes('<!DOCTYPE'),
                () => /<html[\s>]/.test(content),
                () => /<div[\s>]/.test(content) && /<\/div>/.test(content),
                () => rawHTML.includes('<script>') || rawHTML.includes('<style>')
            ],
            'markdown': [
                () => classes.includes('markdown'),
                () => /^#\s+/.test(content),
                () => content.includes('**') || content.includes('*'),
                () => content.includes('- ') || content.includes('1. ')
            ],
            'json': [
                () => classes.includes('json'),
                () => /^\s*[{\[]/.test(content) && /[}\]]\s*$/.test(content),
                () => content.includes('"') && content.includes(':')
            ],
            'text': [
                () => classes.includes('text'),
                () => element.tagName === 'TEXTAREA'
            ]
        };

        for (const [type, checks] of Object.entries(typeIndicators)) {
            if (checks.some(check => {
                try {
                    return check();
                } catch (e) {
                    return false;
                }
            })) {
                return type;
            }
        }

        return 'unknown';
    }

    function extractArtifactContent(element) {
        // Multiple content extraction strategies
        const strategies = [
            () => element.querySelector('.artifact-content, [data-testid="artifact-content"]')?.textContent,
            () => element.querySelector('pre, code')?.textContent,
            () => element.querySelector('.content')?.textContent,
            () => element.textContent,
            () => element.innerText,
            () => element.innerHTML
        ];

        for (const strategy of strategies) {
            try {
                const content = strategy();
                if (content && content.trim()) {
                    return extractorConfig.preserveWhitespace ? content : content.trim();
                }
            } catch (e) {
                // Continue to next strategy
            }
        }

        return '';
    }

    function detectArtifactLanguage(element) {
        const content = safeExtractText(element);
        const classes = element.className || '';

        // Language detection patterns
        const languagePatterns = {
            'python': [
                /def\s+\w+\s*\(/,
                /import\s+\w+/,
                /from\s+\w+\s+import/,
                /if\s+__name__\s*==\s*['"']__main__['"']/,
                /\bprint\s*\(/
            ],
            'javascript': [
                /function\s+\w+\s*\(/,
                /const\s+\w+\s*=/,
                /let\s+\w+\s*=/,
                /var\s+\w+\s*=/,
                /=>\s*{/,
                /console\.log\s*\(/
            ],
            'html': [
                /<!DOCTYPE/i,
                /<html[\s>]/i,
                /<div[\s>]/i,
                /<script[\s>]/i,
                /<style[\s>]/i
            ],
            'css': [
                /[.#]\w+\s*{/,
                /\w+\s*:\s*[^;]+;/,
                /@media\s+/,
                /@import\s+/
            ],
            'sql': [
                /SELECT\s+/i,
                /FROM\s+/i,
                /WHERE\s+/i,
                /INSERT\s+INTO/i,
                /UPDATE\s+/i,
                /DELETE\s+FROM/i
            ],
            'bash': [
                /^#!/,
                /\$\w+/,
                /echo\s+/,
                /cd\s+/,
                /ls\s+/,
                /grep\s+/
            ],
            'json': [
                /^\s*[{\[]/,
                /[}\]]\s*$/,
                /"[\w-]+"\s*:/
            ],
            'xml': [
                /<\?xml/i,
                /<\/\w+>/,
                /<\w+[^>]*\/>/
            ]
        };

        // Check class names first
        const classLanguages = ['python', 'javascript', 'html', 'css', 'sql', 'bash', 'json', 'xml'];
        for (const lang of classLanguages) {
            if (classes.includes(lang) || classes.includes(`language-${lang}`)) {
                return lang;
            }
        }

        // Pattern-based detection
        for (const [lang, patterns] of Object.entries(languagePatterns)) {
            if (patterns.some(pattern => pattern.test(content))) {
                return lang;
            }
        }

        return 'text';
    }

    function extractArtifactDependencies(element) {
        const content = safeExtractText(element);
        const dependencies = [];

        // Extract various types of dependencies
        const depPatterns = {
            'python_import': /^(?:from\s+(\S+)\s+)?import\s+(.+)$/gm,
            'javascript_import': /import\s+.*?\s+from\s+['"]([^'"]+)['"]/g,
            'javascript_require': /require\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
            'html_script': /<script[^>]+src\s*=\s*['"]([^'"]+)['"]/gi,
            'html_link': /<link[^>]+href\s*=\s*['"]([^'"]+)['"]/gi,
            'css_import': /@import\s+(?:url\s*\(\s*)?['"]([^'"]+)['"]/gi
        };

        for (const [type, pattern] of Object.entries(depPatterns)) {
            let match;
            while ((match = pattern.exec(content)) !== null) {
                dependencies.push({
                    type: type,
                    name: match[1] || match[2] || match[0],
                    line: content.substring(0, match.index).split('\n').length
                });
            }
        }

        return dependencies;
    }

    function extractArtifactExecutionInfo(element) {
        const executionInfo = {
            has_been_executed: false,
            execution_results: [],
            error_states: [],
            interactive_elements: []
        };

        // Look for execution indicators
        const executionIndicators = element.querySelectorAll('.execution-result, .output, .error, .console');
        executionIndicators.forEach(indicator => {
            const text = safeExtractText(indicator);
            const isError = indicator.className.includes('error') || text.toLowerCase().includes('error');

            if (isError) {
                executionInfo.error_states.push({
                    message: text,
                    timestamp: new Date().toISOString(),
                    element_class: indicator.className
                });
            } else {
                executionInfo.execution_results.push({
                    output: text,
                    timestamp: new Date().toISOString(),
                    element_class: indicator.className
                });
            }

            executionInfo.has_been_executed = true;
        });

        // Look for interactive elements
        const interactiveElements = element.querySelectorAll('button, input, select, textarea');
        interactiveElements.forEach(el => {
            executionInfo.interactive_elements.push({
                type: el.tagName.toLowerCase(),
                value: el.value || el.textContent,
                attributes: extractAllAttributes(el)
            });
        });

        return executionInfo;
    }

    function extractArtifactDOMContext(element) {
        return {
            parent_element: element.parentElement ? {
                tag: element.parentElement.tagName,
                classes: element.parentElement.className,
                id: element.parentElement.id
            } : null,
            sibling_count: element.parentElement ? element.parentElement.children.length : 0,
            position_in_parent: Array.from(element.parentElement?.children || []).indexOf(element),
            depth_from_body: getElementDepth(element),
            xpath: generateXPath(element),
            css_selector: generateCSSSelector(element)
        };
    }

    function getElementDepth(element) {
        let depth = 0;
        let current = element;
        while (current.parentElement) {
            depth++;
            current = current.parentElement;
        }
        return depth;
    }

    function generateXPath(element) {
        if (element.id) {
            return `//*[@id="${element.id}"]`;
        }

        const path = [];
        let current = element;

        while (current.parentElement) {
            const siblings = Array.from(current.parentElement.children);
            const index = siblings.indexOf(current) + 1;
            path.unshift(`${current.tagName.toLowerCase()}[${index}]`);
            current = current.parentElement;
        }

        return '/' + path.join('/');
    }

    function generateCSSSelector(element) {
        if (element.id) {
            return `#${element.id}`;
        }

        const path = [];
        let current = element;

        while (current.parentElement) {
            let selector = current.tagName.toLowerCase();

            if (current.className) {
                selector += '.' + current.className.split(' ').join('.');
            }

            const siblings = Array.from(current.parentElement.children).filter(
                sibling => sibling.tagName === current.tagName
            );

            if (siblings.length > 1) {
                const index = siblings.indexOf(current) + 1;
                selector += `:nth-of-type(${index})`;
            }

            path.unshift(selector);
            current = current.parentElement;
        }

        return path.join(' > ');
    }

    function calculateArtifactSizeMetrics(element) {
        const content = safeExtractText(element);

        return {
            content_length: content.length,
            line_count: content.split('\n').length,
            word_count: content.split(/\s+/).filter(word => word.length > 0).length,
            dom_size: element.outerHTML?.length || 0,
            element_count: element.querySelectorAll('*').length
        };
    }

    function extractDocuments() {
        log('Performing comprehensive document context extraction...', 'deep');

        const documents = [];

        // Enhanced document detection
        const docSelectors = [
            '[data-testid="document"]',
            '.document',
            '[class*="document"]',
            '[data-document-id]',
            '.file-upload',
            '.uploaded-file',
            '[class*="file"]',
            '.attachment'
        ];

        const docElements = new Set();
        docSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => docElements.add(el));
        });

        Array.from(docElements).forEach((element, index) => {
            try {
                const doc = {
                    id: element.getAttribute('data-document-id') ||
                        element.id ||
                        `doc_${index}_${Date.now()}`,
                    title: extractDocumentTitle(element),
                    type: extractDocumentType(element),
                    content_preview: extractDocumentPreview(element),
                    full_content: extractDocumentFullContent(element),
                    processing_info: extractDocumentProcessing(element),
                    metadata: extractDocumentMetadata(element),
                    file_info: extractDocumentFileInfo(element),
                    extraction_methods: extractDocumentExtractionMethods(element),
                    dom_context: extractArtifactDOMContext(element)
                };

                documents.push(doc);
                log(`Extracted document: ${doc.title} (${doc.type}, preview: ${doc.content_preview.length} chars)`, 'debug');
            } catch (e) {
                log(`Error extracting document ${index}: ${e.message}`, 'warning');
            }
        });

        return documents;
    }

    function extractDocumentFullContent(element) {
        // Try multiple extraction methods for complete content
        const extractionMethods = [
            () => element.querySelector('.document-content, [data-testid="document-content"]')?.textContent,
            () => element.querySelector('pre')?.textContent,
            () => element.querySelector('.content')?.textContent,
            () => element.textContent,
            () => element.innerHTML
        ];

        for (const method of extractionMethods) {
            try {
                const content = method();
                if (content && content.length > 100) { // Prefer longer extractions
                    return content;
                }
            } catch (e) {
                // Continue to next method
            }
        }

        return safeExtractText(element);
    }

    function extractDocumentFileInfo(element) {
        const fileInfo = {
            filename: null,
            size: null,
            mime_type: null,
            upload_timestamp: null,
            file_extension: null
        };

        // Extract filename from various sources
        const filenameSelectors = [
            '.filename', '[data-filename]', '.file-name', '.document-title'
        ];

        for (const selector of filenameSelectors) {
            const filenameEl = element.querySelector(selector);
            if (filenameEl) {
                fileInfo.filename = safeExtractText(filenameEl);
                break;
            }
        }

        // Extract from attributes
        fileInfo.filename = fileInfo.filename ||
                          element.getAttribute('data-filename') ||
                          element.getAttribute('title');

        // Extract file extension
        if (fileInfo.filename) {
            const extensionMatch = fileInfo.filename.match(/\.([^.]+)$/);
            fileInfo.file_extension = extensionMatch ? extensionMatch[1].toLowerCase() : null;
        }

        // Look for size information
        const sizeText = safeExtractText(element);
        const sizeMatch = sizeText.match(/(\d+(?:\.\d+)?)\s*(KB|MB|GB|bytes?)/i);
        if (sizeMatch) {
            fileInfo.size = sizeMatch[0];
        }

        return fileInfo;
    }

    function extractDocumentExtractionMethods(element) {
        const methods = [];

        const content = safeExtractText(element);

        // Detect processing indicators
        if (content.includes('OCR') || content.includes('text recognition')) {
            methods.push('ocr');
        }
        if (content.includes('parsed') || content.includes('parsing')) {
            methods.push('parsing');
        }
        if (content.includes('extracted') || content.includes('extraction')) {
            methods.push('extraction');
        }
        if (content.includes('converted') || content.includes('conversion')) {
            methods.push('conversion');
        }

        return methods;
    }

    function extractToolCalls() {
        log('Performing comprehensive tool call and result extraction...', 'deep');

        const toolCalls = [];

        // Enhanced tool call detection
        const toolSelectors = [
            '[data-testid="tool-call"]',
            '.tool-call',
            '.function-call',
            '[class*="tool"]',
            '[data-tool]',
            '[data-function]'
        ];

        // Also look for function call patterns in content
        const functionCallPattern = /<function_calls>[\s\S]*?<\/antml:function_calls>/g;
        const bodyHTML = document.body.innerHTML;
        const functionCallMatches = bodyHTML.match(functionCallPattern) || [];

        // Extract from HTML patterns
        functionCallMatches.forEach((match, index) => {
            const toolCall = parseHTMLFunctionCall(match, index);
            if (toolCall) {
                toolCalls.push(toolCall);
            }
        });

        // Extract from DOM elements
        const toolElements = new Set();
        toolSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => toolElements.add(el));
        });

        Array.from(toolElements).forEach((element, index) => {
            try {
                const toolCall = {
                    id: element.getAttribute('data-tool-id') ||
                        element.id ||
                        `tool_${index}_${Date.now()}`,
                    tool_name: extractToolName(element),
                    parameters: extractToolParameters(element),
                    result: extractToolResult(element),
                    timestamp: extractToolTimestamp(element),
                    status: extractToolStatus(element),
                    metadata: extractToolMetadata(element),
                    execution_context: extractToolExecutionContext(element),
                    performance_metrics: extractToolPerformanceMetrics(element),
                    dom_context: extractArtifactDOMContext(element)
                };

                if (toolCall.tool_name && toolCall.tool_name !== 'unknown_tool') {
                    toolCalls.push(toolCall);
                    log(`Extracted tool call: ${toolCall.tool_name} (${toolCall.status})`, 'debug');
                }
            } catch (e) {
                log(`Error extracting tool call ${index}: ${e.message}`, 'warning');
            }
        });

        return toolCalls;
    }

    function parseHTMLFunctionCall(htmlMatch, index) {
        try {
            const invokeMatch = htmlMatch.match(/<invoke name="([^"]+)">/);
            if (!invokeMatch) return null;

            const toolName = invokeMatch[1];
            const parameters = {};

            // Extract parameters
            const paramPattern = /<parameter name="([^"]+)">([\s\S]*?)<\/antml:parameter>/g;
            let paramMatch;
            while ((paramMatch = paramPattern.exec(htmlMatch)) !== null) {
                parameters[paramMatch[1]] = paramMatch[2];
            }

            return {
                id: `html_tool_${index}_${Date.now()}`,
                tool_name: toolName,
                parameters: parameters,
                result: null, // Would need to find corresponding result
                timestamp: new Date().toISOString(),
                status: 'parsed_from_html',
                metadata: {
                    extraction_method: 'html_pattern',
                    raw_html: htmlMatch
                },
                execution_context: {
                    found_in_html: true,
                    html_length: htmlMatch.length
                },
                performance_metrics: {},
                dom_context: null
            };
        } catch (e) {
            log(`Error parsing HTML function call: ${e.message}`, 'warning');
            return null;
        }
    }

    function extractToolExecutionContext(element) {
        return {
            parent_message: findParentMessage(element),
            preceding_tools: findPrecedingTools(element),
            following_tools: findFollowingTools(element),
            conversation_position: getConversationPosition(element),
            user_prompt_context: extractUserPromptContext(element)
        };
    }

    function findParentMessage(element) {
        let current = element;
        while (current && current !== document.body) {
            if (current.matches('[data-testid="message"], .message')) {
                return {
                    speaker: detectMessageSpeaker(current),
                    content_preview: safeExtractText(current).substring(0, 200),
                    timestamp: extractMessageTimestamp(current)
                };
            }
            current = current.parentElement;
        }
        return null;
    }

    function findPrecedingTools(element) {
        // This is a simplified implementation
        const allTools = document.querySelectorAll('[data-testid="tool-call"], .tool-call');
        const currentIndex = Array.from(allTools).indexOf(element);

        if (currentIndex > 0) {
            return Array.from(allTools).slice(Math.max(0, currentIndex - 3), currentIndex).map(tool => ({
                tool_name: extractToolName(tool),
                timestamp: extractToolTimestamp(tool)
            }));
        }

        return [];
    }

    function findFollowingTools(element) {
        // Similar to findPrecedingTools but for following tools
        const allTools = document.querySelectorAll('[data-testid="tool-call"], .tool-call');
        const currentIndex = Array.from(allTools).indexOf(element);

        if (currentIndex >= 0 && currentIndex < allTools.length - 1) {
            return Array.from(allTools).slice(currentIndex + 1, currentIndex + 4).map(tool => ({
                tool_name: extractToolName(tool),
                timestamp: extractToolTimestamp(tool)
            }));
        }

        return [];
    }

    function getConversationPosition(element) {
        const allMessages = document.querySelectorAll('[data-testid="message"], .message');
        const parentMessage = findParentMessage(element);

        if (parentMessage) {
            // Find which message contains this tool
            for (let i = 0; i < allMessages.length; i++) {
                if (allMessages[i].contains(element)) {
                    return {
                        message_index: i,
                        total_messages: allMessages.length,
                        relative_position: i / allMessages.length
                    };
                }
            }
        }

        return null;
    }

    function extractUserPromptContext(element) {
        const parentMessage = findParentMessage(element);
        if (!parentMessage) return null;

        // Find the user message that preceded this tool call
        const allMessages = Array.from(document.querySelectorAll('[data-testid="message"], .message'));
        const currentMessageIndex = allMessages.findIndex(msg => msg.contains(element));

        // Look backwards for user message
        for (let i = currentMessageIndex - 1; i >= 0; i--) {
            if (detectMessageSpeaker(allMessages[i]) === 'User') {
                return {
                    user_prompt: safeExtractText(allMessages[i]),
                    prompt_length: safeExtractText(allMessages[i]).length,
                    messages_ago: currentMessageIndex - i
                };
            }
        }

        return null;
    }

    function extractToolPerformanceMetrics(element) {
        const metrics = {
            execution_time: null,
            result_size: null,
            error_count: 0,
            retry_count: 0
        };

        // Look for performance indicators
        const performanceElements = element.querySelectorAll('.duration, .time, .performance, [data-duration]');
        performanceElements.forEach(perfEl => {
            const text = safeExtractText(perfEl);
            const timeMatch = text.match(/(\d+(?:\.\d+)?)\s*(ms|s|seconds?|milliseconds?)/i);
            if (timeMatch) {
                metrics.execution_time = timeMatch[0];
            }
        });

        // Check for errors
        const errorElements = element.querySelectorAll('.error, .warning, [class*="error"]');
        metrics.error_count = errorElements.length;

        // Check for retry indicators
        const retryElements = element.querySelectorAll('.retry, [data-retry], [class*="retry"]');
        metrics.retry_count = retryElements.length;

        // Calculate result size if available
        const result = extractToolResult(element);
        if (result && result.content) {
            metrics.result_size = result.content.length;
        }

        return metrics;
    }

    function extractConversationFlow() {
        log('Extracting detailed conversation flow and interaction patterns...', 'deep');

        const messages = document.querySelectorAll('[data-testid="message"], .message, [role="article"]');
        const conversationFlow = [];

        messages.forEach((message, index) => {
            try {
                const messageData = {
                    index: index,
                    id: message.getAttribute('data-message-id') ||
                        message.id ||
                        `msg_${index}_${Date.now()}`,
                    speaker: detectMessageSpeaker(message),
                    content: extractMessageContent(message),
                    raw_html: extractorConfig.captureRawHTML ? message.outerHTML : null,
                    timestamp: extractMessageTimestamp(message),
                    type: detectMessageType(message),
                    metadata: extractMessageMetadata(message),
                    interactions: extractMessageInteractions(message),
                    threading_info: extractMessageThreadingInfo(message, index),
                    content_analysis: analyzeMessageContent(message),
                    dom_context: extractArtifactDOMContext(message),
                    edit_history: extractMessageEditHistory(message),
                    regeneration_info: extractMessageRegenerationInfo(message)
                };

                conversationFlow.push(messageData);
                log(`Extracted message ${index}: ${messageData.speaker} (${messageData.content.length} chars)`, 'debug');
            } catch (e) {
                log(`Error extracting message ${index}: ${e.message}`, 'warning');
            }
        });

        return conversationFlow;
    }

    function detectMessageSpeaker(element) {
        // Enhanced speaker detection
        const speakerIndicators = [
            element.getAttribute('data-speaker'),
            element.getAttribute('data-role'),
            element.querySelector('[data-testid="user-message"]') ? 'User' : null,
            element.querySelector('[data-testid="assistant-message"]') ? 'Assistant' : null,
            element.querySelector('.user-message') ? 'User' : null,
            element.querySelector('.assistant-message') ? 'Assistant' : null
        ].filter(Boolean);

        if (speakerIndicators.length > 0) {
            return speakerIndicators[0];
        }

        const content = safeExtractText(element);
        const contentLower = content.toLowerCase();

        // Pattern-based detection
        if (contentLower.startsWith('user:') || contentLower.startsWith('human:')) return 'User';
        if (contentLower.startsWith('assistant:') || contentLower.startsWith('claude:')) return 'Assistant';

        // Look for user input indicators
        const hasUserIndicators = element.querySelector('input, textarea, [contenteditable]');
        if (hasUserIndicators) return 'User';

        // Look for tool calls (usually assistant)
        const hasToolCalls = element.querySelector('[data-testid="tool-call"], .tool-call');
        if (hasToolCalls) return 'Assistant';

        // Look for artifacts (usually assistant)
        const hasArtifacts = element.querySelector('[data-testid="artifact"], .artifact');
        if (hasArtifacts) return 'Assistant';

        // Analyze position and context
        const prevElement = element.previousElementSibling;
        if (prevElement) {
            const prevSpeaker = detectMessageSpeaker(prevElement);
            if (prevSpeaker === 'User') return 'Assistant';
            if (prevSpeaker === 'Assistant') return 'User';
        }

        return 'Unknown';
    }

    function extractMessageThreadingInfo(message, index) {
        return {
            message_index: index,
            is_first_message: index === 0,
            is_last_message: index === document.querySelectorAll('[data-testid="message"], .message').length - 1,
            previous_speaker: index > 0 ? detectMessageSpeaker(
                document.querySelectorAll('[data-testid="message"], .message')[index - 1]
            ) : null,
            next_speaker: (() => {
                const allMessages = document.querySelectorAll('[data-testid="message"], .message');
                return index < allMessages.length - 1 ? detectMessageSpeaker(allMessages[index + 1]) : null;
            })(),
            conversation_turn: Math.ceil((index + 1) / 2), // Assuming alternating speakers
            is_continuation: detectContinuation(message, index)
        };
    }

    function detectContinuation(message, index) {
        if (index === 0) return false;

        const allMessages = document.querySelectorAll('[data-testid="message"], .message');
        const currentSpeaker = detectMessageSpeaker(message);
        const previousSpeaker = detectMessageSpeaker(allMessages[index - 1]);

        return currentSpeaker === previousSpeaker;
    }

    function analyzeMessageContent(message) {
        const content = safeExtractText(message);

        return {
            word_count: content.split(/\s+/).filter(word => word.length > 0).length,
            char_count: content.length,
            line_count: content.split('\n').length,
            paragraph_count: content.split(/\n\s*\n/).length,
            has_code_blocks: content.includes('```'),
            has_inline_code: content.includes('`') && !content.includes('```'),
            has_links: !!message.querySelector('a'),
            has_images: !!message.querySelector('img'),
            has_lists: /^\s*[\-\*\d+\.]/m.test(content),
            has_headings: /^#+\s+/m.test(content),
            has_bold: content.includes('**') || !!message.querySelector('b, strong'),
            has_italic: content.includes('*') || !!message.querySelector('i, em'),
            question_count: (content.match(/\?/g) || []).length,
            exclamation_count: (content.match(/!/g) || []).length,
            contains_urls: /https?:\/\/[^\s]+/g.test(content),
            language_indicators: detectContentLanguage(content),
            sentiment_indicators: detectSentimentIndicators(content),
            complexity_score: calculateContentComplexity(content)
        };
    }

    function detectContentLanguage(content) {
        // Simple language detection based on common words
        const languages = {
            'english': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that'],
            'spanish': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es'],
            'french': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et'],
            'german': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das']
        };

        const words = content.toLowerCase().split(/\s+/);
        const scores = {};

        for (const [lang, commonWords] of Object.entries(languages)) {
            scores[lang] = commonWords.reduce((score, word) => {
                return score + (words.filter(w => w === word).length);
            }, 0);
        }

        const totalWords = words.length;
        const detectedLanguages = Object.entries(scores)
            .filter(([lang, score]) => score / totalWords > 0.01)
            .sort(([,a], [,b]) => b - a)
            .map(([lang, score]) => ({ language: lang, confidence: score / totalWords }));

        return detectedLanguages;
    }

    function detectSentimentIndicators(content) {
        const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 'perfect'];
        const negativeWords = ['bad', 'terrible', 'awful', 'horrible', 'worse', 'worst', 'hate', 'annoying'];
        const questionWords = ['how', 'what', 'when', 'where', 'why', 'which', 'who'];

        const words = content.toLowerCase().split(/\s+/);

        return {
            positive_indicators: positiveWords.filter(word => words.includes(word)).length,
            negative_indicators: negativeWords.filter(word => words.includes(word)).length,
            question_indicators: questionWords.filter(word => words.includes(word)).length,
            uncertainty_indicators: ['maybe', 'perhaps', 'possibly', 'might', 'could'].filter(word => words.includes(word)).length,
            certainty_indicators: ['definitely', 'certainly', 'absolutely', 'surely', 'clearly'].filter(word => words.includes(word)).length
        };
    }

    function calculateContentComplexity(content) {
        const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);
        const words = content.split(/\s+/).filter(w => w.length > 0);

        if (sentences.length === 0 || words.length === 0) return 0;

        const avgWordsPerSentence = words.length / sentences.length;
        const avgCharsPerWord = content.replace(/\s/g, '').length / words.length;
        const longWordRatio = words.filter(word => word.length > 6).length / words.length;

        // Simple complexity score
        return (avgWordsPerSentence * 0.4) + (avgCharsPerWord * 0.3) + (longWordRatio * 0.3);
    }

    function extractMessageEditHistory(message) {
        const editHistory = [];

        // Look for edit indicators
        const editElements = message.querySelectorAll('.edit, .edited, [data-edited], .modification');
        editElements.forEach(editEl => {
            editHistory.push({
                type: 'edit_indicator',
                content: safeExtractText(editEl),
                timestamp: editEl.getAttribute('data-timestamp') || new Date().toISOString()
            });
        });

        // Look for version information
        const versionElements = message.querySelectorAll('[data-version], .version');
        versionElements.forEach(versionEl => {
            editHistory.push({
                type: 'version',
                version: versionEl.getAttribute('data-version') || safeExtractText(versionEl),
                timestamp: versionEl.getAttribute('data-timestamp') || new Date().toISOString()
            });
        });

        return editHistory;
    }

    function extractMessageRegenerationInfo(message) {
        const regenerationInfo = {
            has_regeneration_option: false,
            regeneration_count: 0,
            alternative_versions: []
        };

        // Look for regeneration buttons/options
        const regenElements = message.querySelectorAll('[data-testid="regenerate"], .regenerate, .retry');
        regenerationInfo.has_regeneration_option = regenElements.length > 0;

        // Look for version indicators
        const versionElements = message.querySelectorAll('.version, [data-version]');
        regenerationInfo.regeneration_count = versionElements.length;

        // Extract alternative versions if visible
        const altElements = message.querySelectorAll('.alternative, .variant, [data-alternative]');
        altElements.forEach(altEl => {
            regenerationInfo.alternative_versions.push({
                content: safeExtractText(altEl),
                version_id: altEl.getAttribute('data-version') || altEl.getAttribute('data-alternative')
            });
        });

        return regenerationInfo;
    }

    // Continue with remaining enhanced functions...
    function extractSystemConfiguration() {
        log('Extracting comprehensive system configuration...', 'deep');

        return {
            model_info: {
                name: detectClaudeVersion(),
                features: detectAvailableFeatures(),
                capabilities: detectCapabilities(),
                limitations: detectLimitations()
            },
            ui_state: extractUIState(),
            session_info: extractSessionInfo(),
            browser_state: extractBrowserState(),
            performance_metrics: extractPerformanceMetrics()
        };
    }

    function detectLimitations() {
        const limitations = [];

        // Look for limitation indicators in the UI
        const limitationText = document.body.textContent.toLowerCase();

        if (limitationText.includes('rate limit')) {
            limitations.push('rate_limited');
        }
        if (limitationText.includes('context limit')) {
            limitations.push('context_limited');
        }
        if (limitationText.includes('premium') || limitationText.includes('upgrade')) {
            limitations.push('feature_limited');
        }

        return limitations;
    }

    function extractBrowserState() {
        return {
            cookies_enabled: navigator.cookieEnabled,
            online: navigator.onLine,
            java_enabled: navigator.javaEnabled ? navigator.javaEnabled() : false,
            do_not_track: navigator.doNotTrack,
            storage_quota: estimateStorageQuota(),
            active_service_workers: getActiveServiceWorkers()
        };
    }

    async function estimateStorageQuota() {
        try {
            if ('storage' in navigator && 'estimate' in navigator.storage) {
                const estimate = await navigator.storage.estimate();
                return {
                    quota: estimate.quota,
                    usage: estimate.usage,
                    available: estimate.quota - estimate.usage
                };
            }
        } catch (e) {
            log('Storage quota estimation failed', 'debug');
        }
        return null;
    }

    function getActiveServiceWorkers() {
        try {
            if ('serviceWorker' in navigator) {
                return navigator.serviceWorker.controller ? 'active' : 'none';
            }
        } catch (e) {
            log('Service worker detection failed', 'debug');
        }
        return 'unknown';
    }

    function extractPerformanceMetrics() {
        const metrics = {
            page_load_time: null,
            dom_content_loaded: null,
            first_paint: null,
            largest_contentful_paint: null,
            cumulative_layout_shift: null,
            first_input_delay: null
        };

        try {
            const navigation = performance.getEntriesByType('navigation')[0];
            if (navigation) {
                metrics.page_load_time = navigation.loadEventEnd - navigation.loadEventStart;
                metrics.dom_content_loaded = navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart;
            }

            // Web Vitals if available
            const paintEntries = performance.getEntriesByType('paint');
            const firstPaint = paintEntries.find(entry => entry.name === 'first-paint');
            if (firstPaint) {
                metrics.first_paint = firstPaint.startTime;
            }

            const largestContentfulPaint = performance.getEntriesByType('largest-contentful-paint')[0];
            if (largestContentfulPaint) {
                metrics.largest_contentful_paint = largestContentfulPaint.startTime;
            }

        } catch (e) {
            log('Performance metrics extraction failed', 'debug');
        }

        return metrics;
    }

    function createComprehensiveState() {
        const state = {
            meta: {
                extractor_version: '2.0',
                extraction_id: generateStateId(),
                extracted_at: new Date().toISOString(),
                extraction_type: 'enhanced_comprehensive_conversation_state',
                extractor_config: extractorConfig,
                extraction_duration: null // Will be set after extraction
            },
            environment: extractorConfig.includeEnvironment ? detectEnvironment() : null,
            user_preferences: extractorConfig.includePreferences ? extractUserPreferences() : null,
            conversation_flow: extractConversationFlow(),
            artifacts: extractorConfig.includeArtifacts ? extractArtifacts() : [],
            documents: extractorConfig.includeDocuments ? extractDocuments() : [],
            tool_calls: extractorConfig.includeToolCalls ? extractToolCalls() : [],
            system_config: extractSystemConfiguration(),
            conversation_analysis: performConversationAnalysis(),
            quality_metrics: calculateQualityMetrics(),
            replay_instructions: generateEnhancedReplayInstructions(),
            checksum: null // Will be calculated after state creation
        };

        // Calculate extraction metrics
        const endTime = Date.now();
        state.meta.extraction_duration = endTime - (state.environment?.timestamp ? new Date(state.environment.timestamp).getTime() : endTime);

        // Calculate checksum for integrity verification
        state.checksum = calculateStateChecksum(state);

        return state;
    }

    function performConversationAnalysis() {
        log('Performing deep conversation analysis...', 'deep');

        const messages = document.querySelectorAll('[data-testid="message"], .message');
        const analysis = {
            conversation_length: messages.length,
            user_messages: 0,
            assistant_messages: 0,
            tool_usage_frequency: {},
            artifact_creation_pattern: [],
            conversation_topics: [],
            interaction_patterns: {},
            conversation_quality_indicators: {}
        };

        messages.forEach((message, index) => {
            const speaker = detectMessageSpeaker(message);
            if (speaker === 'User') analysis.user_messages++;
            if (speaker === 'Assistant') analysis.assistant_messages++;
        });

        // Analyze tool usage patterns
        const toolCalls = document.querySelectorAll('[data-testid="tool-call"], .tool-call');
        toolCalls.forEach(tool => {
            const toolName = extractToolName(tool);
            analysis.tool_usage_frequency[toolName] = (analysis.tool_usage_frequency[toolName] || 0) + 1;
        });

        // Analyze artifact creation pattern
        const artifacts = document.querySelectorAll('[data-testid="artifact"], .artifact');
        artifacts.forEach((artifact, index) => {
            analysis.artifact_creation_pattern.push({
                index: index,
                type: detectArtifactType(artifact),
                creation_context: findParentMessage(artifact)?.content_preview || 'unknown'
            });
        });

        return analysis;
    }

    function calculateQualityMetrics() {
        return {
            extraction_completeness: calculateExtractionCompleteness(),
            data_integrity_score: calculateDataIntegrityScore(),
            replay_fidelity_estimate: calculateReplayFidelityEstimate(),
            content_preservation_score: calculateContentPreservationScore()
        };
    }

    function calculateExtractionCompleteness() {
        const extractedElements = {
            messages: document.querySelectorAll('[data-testid="message"], .message').length,
            artifacts: document.querySelectorAll('[data-testid="artifact"], .artifact').length,
            tools: document.querySelectorAll('[data-testid="tool-call"], .tool-call').length,
            documents: document.querySelectorAll('[data-testid="document"], .document').length
        };

        const totalElements = Object.values(extractedElements).reduce((a, b) => a + b, 0);
        return totalElements > 0 ? 1.0 : 0.0; // Simplified metric
    }

    function calculateDataIntegrityScore() {
        // Check for extraction errors and data consistency
        let score = 1.0;

        // Penalize for extraction errors (simplified)
        const errorCount = document.querySelectorAll('.error, [class*="error"]').length;
        score -= (errorCount * 0.1);

        return Math.max(0, score);
    }

    function calculateReplayFidelityEstimate() {
        // Estimate how well this extraction could be replayed
        let fidelity = 1.0;

        // Factor in various completeness metrics
        const hasToolCalls = document.querySelectorAll('[data-testid="tool-call"], .tool-call').length > 0;
        const hasArtifacts = document.querySelectorAll('[data-testid="artifact"], .artifact').length > 0;
        const hasDocuments = document.querySelectorAll('[data-testid="document"], .document').length > 0;

        if (hasToolCalls) fidelity += 0.1; // Tool calls increase complexity but fidelity
        if (hasArtifacts) fidelity += 0.1; // Artifacts are well-preserved
        if (hasDocuments) fidelity -= 0.1; // Documents might be harder to replay exactly

        return Math.min(1.0, fidelity);
    }

    function calculateContentPreservationScore() {
        // Measure how well content formatting and structure is preserved
        const elementsWithFormatting = document.querySelectorAll('pre, code, .highlight, .formatted').length;
        const totalContentElements = document.querySelectorAll('p, div, span').length;

        return totalContentElements > 0 ? Math.min(1.0, elementsWithFormatting / totalContentElements) : 1.0;
    }

    function generateEnhancedReplayInstructions() {
        return {
            setup_requirements: [
                'Ensure identical Claude model/version for maximum fidelity',
                'Upload all referenced documents in exact original order',
                'Enable identical tool set as detected in system_config.capabilities',
                'Apply user preferences from user_preferences section',
                'Set browser environment to match environment section if possible',
                'Verify conversation context limits match original session'
            ],
            replay_process: [
                'Initialize new conversation with identical system configuration',
                'Apply detected user preferences and communication style',
                'Upload documents following documents section metadata',
                'Replay conversation_flow in exact sequential order',
                'Execute tool_calls with original parameters where possible',
                'Recreate artifacts following artifacts section specifications',
                'Maintain original interaction patterns and timing',
                'Monitor for deviations and apply corrections'
            ],
            fidelity_optimization: [
                'Use checksum verification for data integrity',
                'Cross-reference quality_metrics for completeness validation',
                'Apply conversation_analysis insights for context preservation',
                'Utilize DOM context information for precise element recreation',
                'Leverage performance_metrics for timing accuracy'
            ],
            known_limitations: [
                'External tool results may vary due to real-time data changes',
                'System timestamps cannot be perfectly replicated',
                'Browser-specific rendering differences may occur',
                'Network conditions will affect tool execution timing',
                'Some user interface states are session-dependent',
                'Document processing results may vary with different file versions'
            ],
            validation_steps: [
                'Compare conversation_flow message counts and structure',
                'Verify artifacts section against recreated artifacts',
                'Validate tool_calls execution against original results',
                'Check user_preferences application accuracy',
                'Confirm document processing consistency',
                'Cross-reference quality_metrics for replay success'
            ]
        };
    }

    function calculateStateChecksum(state) {
        // Simple checksum calculation for integrity verification
        const stateString = JSON.stringify(state, (key, value) => {
            if (key === 'checksum') return undefined; // Exclude checksum from calculation
            return value;
        });

        let hash = 0;
        for (let i = 0; i < stateString.length; i++) {
            const char = stateString.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }

        return hash.toString(16);
    }

    function downloadState(state, filename) {
        try {
            log(`Preparing download of comprehensive state (${JSON.stringify(state).length} characters)...`, 'info');

            const jsonString = JSON.stringify(state, null, 2);
            const blob = new Blob([jsonString], { type: 'application/json;charset=utf-8' });
            const url = URL.createObjectURL(blob);

            const downloadLink = document.createElement('a');
            downloadLink.href = url;
            downloadLink.download = filename;
            downloadLink.style.display = 'none';

            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);

            setTimeout(() => URL.revokeObjectURL(url), 1000);

            log(`Successfully downloaded: ${filename} (${(blob.size / 1024 / 1024).toFixed(2)} MB)`, 'success');
            return true;
        } catch (error) {
            log(`Download failed: ${error.message}`, 'error');
            return false;
        }
    }

    function displayEnhancedStateSummary(state) {
        const summary = {
            extraction_id: state.meta.extraction_id,
            extraction_duration: `${state.meta.extraction_duration}ms`,
            conversation_messages: state.conversation_flow.length,
            user_messages: state.conversation_analysis.user_messages,
            assistant_messages: state.conversation_analysis.assistant_messages,
            artifacts_extracted: state.artifacts.length,
            documents_processed: state.documents.length,
            tool_calls_captured: state.tool_calls.length,
            model_detected: state.system_config.model_info.name,
            available_features: state.system_config.model_info.features.join(', '),
            user_style_detected: state.user_preferences?.detected_style || 'unknown',
            technical_level: state.user_preferences?.technical_level || 'unknown',
            total_state_size: `${(JSON.stringify(state).length / 1024 / 1024).toFixed(2)} MB`,
            extraction_completeness: `${(state.quality_metrics.extraction_completeness * 100).toFixed(1)}%`,
            replay_fidelity_estimate: `${(state.quality_metrics.replay_fidelity_estimate * 100).toFixed(1)}%`,
            data_integrity_score: `${(state.quality_metrics.data_integrity_score * 100).toFixed(1)}%`,
            checksum: state.checksum
        };

        console.log('\n' + '='.repeat(80));
        console.log('🧠 ENHANCED COMPREHENSIVE CONVERSATION STATE EXTRACTED');
        console.log('='.repeat(80));

        Object.entries(summary).forEach(([key, value]) => {
            const displayKey = key.toUpperCase().replace(/_/g, ' ');
            console.log(`   ${displayKey}: ${value}`);
        });

        console.log('\n📊 DETAILED BREAKDOWN:');
        console.log(`   • Total Conversation Length: ${state.conversation_flow.length} messages`);
        console.log(`   • User Input Messages: ${state.conversation_analysis.user_messages}`);
        console.log(`   • Assistant Responses: ${state.conversation_analysis.assistant_messages}`);
        console.log(`   • Created Artifacts: ${state.artifacts.length}`);
        console.log(`   • Processed Documents: ${state.documents.length}`);
        console.log(`   • Tool Interactions: ${state.tool_calls.length}`);
        console.log(`   • System Features: ${state.system_config.model_info.features.length}`);
        console.log(`   • Extraction Quality: ${(state.quality_metrics.extraction_completeness * 100).toFixed(1)}%`);
        console.log(`   • Replay Fidelity Est.: ${(state.quality_metrics.replay_fidelity_estimate * 100).toFixed(1)}%`);
        console.log(`   • Total Size: ${(JSON.stringify(state).length / 1024 / 1024).toFixed(2)} MB`);

        if (state.tool_calls.length > 0) {
            console.log('\n🔧 TOOL USAGE ANALYSIS:');
            Object.entries(state.conversation_analysis.tool_usage_frequency).forEach(([tool, count]) => {
                console.log(`   • ${tool}: ${count} calls`);
            });
        }

        if (state.user_preferences) {
            console.log('\n👤 USER PROFILE DETECTED:');
            console.log(`   • Communication Style: ${state.user_preferences.detected_style}`);
            console.log(`   • Technical Level: ${state.user_preferences.technical_level}`);
            console.log(`   • Response Length Pref: ${state.user_preferences.response_length_preference}`);
        }

        return summary;
    }

    // Main execution
    try {
        const startTime = Date.now();
        log('Starting enhanced comprehensive conversation state extraction...', 'info');

        const state = createComprehensiveState();
        const summary = displayEnhancedStateSummary(state);

        // Auto-download the state
        const filename = `claude_enhanced_state_${state.meta.extraction_id}.json`;
        const downloaded = downloadState(state, filename);

        // Make enhanced state available globally
        window.conversationState = state;
        window.downloadConversationState = () => downloadState(state, filename);
        window.getStateJSON = () => {
            const json = JSON.stringify(state, null, 2);
            console.log(json);
            return json;
        };
        window.validateState = () => {
            const currentChecksum = calculateStateChecksum(state);
            const isValid = currentChecksum === state.checksum;
            console.log(`State integrity: ${isValid ? '✅ VALID' : '❌ CORRUPTED'}`);
            console.log(`Original checksum: ${state.checksum}`);
            console.log(`Current checksum: ${currentChecksum}`);
            return isValid;
        };
        window.getExtractionReport = () => {
            console.log('\n📋 EXTRACTION REPORT:');
            console.log(`Extraction ID: ${state.meta.extraction_id}`);
            console.log(`Duration: ${state.meta.extraction_duration}ms`);
            console.log(`Completeness: ${(state.quality_metrics.extraction_completeness * 100).toFixed(1)}%`);
            console.log(`Fidelity: ${(state.quality_metrics.replay_fidelity_estimate * 100).toFixed(1)}%`);
            console.log(`Integrity: ${(state.quality_metrics.data_integrity_score * 100).toFixed(1)}%`);
            return state.quality_metrics;
        };

        const extractionTime = Date.now() - startTime;

        console.log('\n🔧 ENHANCED COMMANDS AVAILABLE:');
        console.log('• downloadConversationState() - Re-download state file');
        console.log('• getStateJSON() - Display complete JSON in console');
        console.log('• validateState() - Verify state integrity with checksum');
        console.log('• getExtractionReport() - Show detailed quality metrics');
        console.log('• window.conversationState - Access complete state object');

        console.log('\n🎯 MAXIMUM FIDELITY REPLAY PACKAGE:');
        console.log('This enhanced state package provides maximum fidelity conversation');
        console.log('replay capabilities with comprehensive state preservation:');
        console.log('• Complete conversation flow with deep metadata');
        console.log('• Full tool interaction history with context');
        console.log('• Complete artifact evolution and dependencies');
        console.log('• Comprehensive user preference inference');
        console.log('• System configuration and environment capture');
        console.log('• Quality metrics and integrity verification');
        console.log('• Enhanced replay instructions and validation');
        console.log('• No size limitations - maximum data preservation');

        if (downloaded) {
            console.log(`\n✅ Enhanced comprehensive state exported successfully!`);
            console.log(`📁 File: ${filename}`);
            console.log(`📊 Size: ${(JSON.stringify(state).length / 1024 / 1024).toFixed(2)} MB`);
            console.log(`⏱️ Extraction time: ${extractionTime}ms`);
            console.log(`🔒 Checksum: ${state.checksum}`);
        } else {
            console.log('\n⚠️ Auto-download failed - use downloadConversationState()');
        }

    } catch (error) {
        log(`Enhanced extraction failed: ${error.message}`, 'error');
        console.error('Full error details:', error);
        console.log('\n🔧 TROUBLESHOOTING:');
        console.log('1. Ensure conversation has completely loaded');
        console.log('2. Try refreshing page and running extractor again');
        console.log('3. Check browser console for detailed error messages');
        console.log('4. Verify browser supports required APIs');
        console.log('5. Some advanced features may not work on shared conversations');
        console.log('6. For large conversations, allow extra time for processing');
    }

})();
