/**
 * Safari-Optimized Claude Conversation Extractor v3.0
 *
 * Designed for extracting conversations from Claude.ai with high fidelity,
 * optimized for Safari browser compatibility and conversation continuation.
 *
 * Features:
 * - Robust conversation turn extraction with multiple fallback methods
 * - Accurate speaker identification using DOM structure analysis
 * - Token counting for context management
 * - Safari-optimized with cross-browser fallbacks
 * - Clean output suitable for editing and re-uploading
 * - Comprehensive artifact and tool call preservation
 * - Smart filename generation from conversation content
 */

(function() {
    'use strict';

    console.log('🍎 Safari-Optimized Claude Conversation Extractor v3.0');
    console.log('🚀 Starting extraction...');

    // Configuration optimized for Safari
    const config = {
        safari_optimized: true,
        preserve_artifacts: true,
        preserve_tool_calls: true,
        include_metadata: true,
        max_filename_length: 50,
        token_estimation_method: 'conservative', // More accurate for Claude
        debug_mode: false
    };

    // Utility functions
    function log(message, level = 'info') {
        const prefix = {
            'debug': '🔍',
            'info': '📖',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }[level];

        if (config.debug_mode || level !== 'debug') {
            console.log(`${prefix} ${message}`);
        }
    }

    function safeExtract(element, property = 'textContent') {
        try {
            return element?.[property]?.trim() || '';
        } catch (e) {
            return '';
        }
    }

    function estimateTokens(text) {
        if (!text || typeof text !== 'string') return 0;

        // Conservative estimation based on Claude's tokenization patterns
        const words = text.split(/\s+/).filter(w => w.length > 0).length;
        const characters = text.length;

        // Adjustments for different content types
        let baseTokens = words * 0.75; // Base word-to-token ratio

        // Code blocks are more token-dense
        const codeBlocks = (text.match(/```[\s\S]*?```/g) || []).length;
        const inlineCode = (text.match(/`[^`]+`/g) || []).length;
        baseTokens += codeBlocks * 15 + inlineCode * 2;

        // Function calls and structured content
        const functionCalls = (text.match(/<function_calls>[\s\S]*?<\/antml:function_calls>/g) || []).length;
        const xmlTags = (text.match(/<[^>]+>/g) || []).length;
        baseTokens += functionCalls * 25 + xmlTags * 1.5;

        // Character-based adjustment for very dense text
        if (characters / words > 6) {
            baseTokens *= 1.2; // Dense text like code or technical content
        }

        return Math.ceil(baseTokens);
    }

    function generateSafeFilename(conversation) {
        let title = 'claude_conversation';

        // Try to extract meaningful title from first user message
        const firstUserMessage = conversation.find(turn => turn.speaker === 'User');
        if (firstUserMessage && firstUserMessage.content) {
            const firstLine = firstUserMessage.content.split('\n')[0];
            if (firstLine.length > 10 && firstLine.length < 100) {
                title = firstLine
                    .toLowerCase()
                    .replace(/[^a-z0-9\s]/g, '') // Remove special chars
                    .replace(/\s+/g, '_') // Replace spaces with underscores
                    .substring(0, config.max_filename_length);
            }
        }

        const timestamp = new Date().toISOString()
            .slice(0, 19)
            .replace(/:/g, '-');

        return `${title}_${timestamp}.json`;
    }

    // Core extraction functions
    function detectMessageElements() {
        log('Detecting message elements using multiple strategies...', 'debug');

        // Try multiple selectors in order of reliability
        const selectors = [
            '[data-testid="user-message"], [data-testid="assistant-message"]',
            '[role="article"]',
            '.message',
            '[data-message-id]',
            '.conversation-turn',
            'div[class*="message"]'
        ];

        let messageElements = [];

        for (const selector of selectors) {
            const elements = Array.from(document.querySelectorAll(selector));
            if (elements.length > 0) {
                log(`Found ${elements.length} elements with selector: ${selector}`, 'debug');
                messageElements = elements;
                break;
            }
        }

        // Fallback: analyze document structure
        if (messageElements.length === 0) {
            log('No direct message elements found, analyzing document structure...', 'warning');
            messageElements = fallbackMessageDetection();
        }

        return messageElements;
    }

    function fallbackMessageDetection() {
        // Find elements that likely contain conversation turns
        const candidates = [];

        // Look for elements with substantial text content
        const allDivs = document.querySelectorAll('div, article, section');

        allDivs.forEach(element => {
            const text = safeExtract(element);
            if (text.length > 50 && text.length < 10000) {
                // Check if it looks like a conversation turn
                const hasUserIndicators = /^(user|human):/i.test(text) ||
                                        text.includes('?') && text.length < 500;
                const hasAssistantIndicators = text.length > 200 ||
                                            /^(assistant|claude)/i.test(text) ||
                                            text.includes('```') ||
                                            text.includes('function_calls');

                if (hasUserIndicators || hasAssistantIndicators) {
                    candidates.push({
                        element: element,
                        text: text,
                        score: calculateConversationScore(text)
                    });
                }
            }
        });

        // Sort by score and return top candidates
        return candidates
            .sort((a, b) => b.score - a.score)
            .slice(0, 50) // Reasonable limit
            .map(c => c.element);
    }

    function calculateConversationScore(text) {
        let score = 0;

        // Positive indicators
        if (text.includes('?')) score += 5;
        if (text.includes('```')) score += 10;
        if (text.includes('function_calls')) score += 15;
        if (/\b(explain|help|how|what|why|can you)\b/i.test(text)) score += 5;
        if (text.length > 100 && text.length < 2000) score += 10;

        // Negative indicators
        if (text.includes('claude.ai')) score -= 10;
        if (text.includes('anthropic')) score -= 5;
        if (text.length < 30) score -= 20;
        if (/^\d+$/.test(text.trim())) score -= 15;

        return score;
    }

    function identifySpeaker(element, text, index, previousSpeaker) {
        // Multiple strategies for speaker identification

        // 1. Direct data attributes
        if (element.hasAttribute('data-testid')) {
            const testId = element.getAttribute('data-testid');
            if (testId.includes('user')) return 'User';
            if (testId.includes('assistant')) return 'Assistant';
        }

        // 2. Text-based prefixes
        if (/^(user|human):\s*/i.test(text)) {
            return 'User';
        }
        if (/^(assistant|claude):\s*/i.test(text)) {
            return 'Assistant';
        }

        // 3. Content analysis
        const contentIndicators = analyzeContentForSpeaker(text);
        if (contentIndicators.confidence > 0.7) {
            return contentIndicators.speaker;
        }

        // 4. DOM structure analysis
        const structuralSpeaker = analyzeDOMStructureForSpeaker(element);
        if (structuralSpeaker) {
            return structuralSpeaker;
        }

        // 5. Pattern-based fallback
        if (text.length < 300 && text.includes('?')) {
            return 'User';
        }
        if (text.includes('```') || text.includes('function_calls') || text.length > 800) {
            return 'Assistant';
        }

        // 6. Alternating pattern (last resort)
        return index % 2 === 0 ? 'User' : 'Assistant';
    }

    function analyzeContentForSpeaker(text) {
        const userPatterns = [
            /^(please|can you|could you|would you|help|how do|what is|why)/i,
            /\?$/,
            /^(i need|i want|i'm trying|i have|my)/i
        ];

        const assistantPatterns = [
            /^(i'll|i can|here's|let me|to)/i,
            /```[\s\S]*?```/,
            /function_calls/,
            /^(certainly|of course|absolutely|sure)/i,
            /(here are|here's how|steps|first|second|third)/i
        ];

        let userScore = 0;
        let assistantScore = 0;

        userPatterns.forEach(pattern => {
            if (pattern.test(text)) userScore += 1;
        });

        assistantPatterns.forEach(pattern => {
            if (pattern.test(text)) assistantScore += 1;
        });

        // Length-based scoring
        if (text.length < 200) userScore += 0.5;
        if (text.length > 500) assistantScore += 0.5;

        const total = userScore + assistantScore;
        if (total === 0) return { speaker: null, confidence: 0 };

        const confidence = Math.max(userScore, assistantScore) / total;
        const speaker = userScore > assistantScore ? 'User' : 'Assistant';

        return { speaker, confidence };
    }

    function analyzeDOMStructureForSpeaker(element) {
        // Look for structural clues in the DOM
        const classList = element.className || '';
        const parentClassList = element.parentElement?.className || '';

        if (classList.includes('user') || parentClassList.includes('user')) {
            return 'User';
        }
        if (classList.includes('assistant') || classList.includes('claude') ||
            parentClassList.includes('assistant') || parentClassList.includes('claude')) {
            return 'Assistant';
        }

        // Check for typical Claude UI patterns
        if (element.querySelector('pre, code') ||
            element.querySelector('[data-testid*="artifact"]')) {
            return 'Assistant';
        }

        return null;
    }

    function cleanMessageContent(text, speaker) {
        if (!text) return '';

        let cleaned = text
            // Remove speaker prefixes
            .replace(/^(user|human|assistant|claude):\s*/i, '')
            // Remove UI artifacts
            .replace(/\n*\d+s\n*/g, '\n')
            .replace(/\n*(edit|retry|regenerate)\n*/gi, '')
            // Remove sharing artifacts
            .replace(/^.*?shared by.*?\n/si, '')
            .replace(/this is a copy of a chat.*?\n/si, '')
            // Clean whitespace
            .replace(/\n{3,}/g, '\n\n')
            .trim();

        return cleaned;
    }

    function extractArtifacts(element) {
        const artifacts = [];

        // Find artifact elements within this message
        const artifactSelectors = [
            '[data-testid="artifact"]',
            '.artifact',
            'pre[class*="language-"]',
            '.code-block'
        ];

        artifactSelectors.forEach(selector => {
            const artifactElements = element.querySelectorAll(selector);
            artifactElements.forEach((artifactEl, index) => {
                const artifact = {
                    type: detectArtifactType(artifactEl),
                    content: safeExtract(artifactEl),
                    language: detectLanguage(artifactEl),
                    title: extractArtifactTitle(artifactEl),
                    id: `artifact_${Date.now()}_${index}`
                };

                if (artifact.content.length > 10) {
                    artifacts.push(artifact);
                }
            });
        });

        return artifacts;
    }

    function detectArtifactType(element) {
        const content = safeExtract(element);
        const className = element.className || '';

        if (className.includes('code') || element.tagName === 'PRE') return 'code';
        if (content.includes('<!DOCTYPE') || content.includes('<html')) return 'html';
        if (content.includes('# ') || content.includes('## ')) return 'markdown';
        if (/^\s*[{\[]/.test(content) && /[}\]]\s*$/.test(content)) return 'json';

        return 'text';
    }

    function detectLanguage(element) {
        const content = safeExtract(element);
        const className = element.className || '';

        // Check class names first
        const langMatch = className.match(/language-(\w+)/);
        if (langMatch) return langMatch[1];

        // Pattern-based detection
        if (/def\s+\w+\s*\(/.test(content)) return 'python';
        if (/function\s+\w+\s*\(/.test(content)) return 'javascript';
        if (/const\s+\w+\s*=/.test(content)) return 'javascript';
        if (/<\?xml/.test(content)) return 'xml';
        if (/<!DOCTYPE/.test(content)) return 'html';

        return 'text';
    }

    function extractArtifactTitle(element) {
        const titleEl = element.querySelector('.artifact-title, [data-title]');
        if (titleEl) return safeExtract(titleEl);

        const content = safeExtract(element);
        const firstLine = content.split('\n')[0];
        if (firstLine.length < 100) return firstLine;

        return 'Untitled';
    }

    function extractToolCalls(element) {
        const toolCalls = [];

        // Look for function call patterns
        const content = safeExtract(element);
        const functionCallMatches = content.match(/<function_calls>[\s\S]*?<\/antml:function_calls>/g) || [];

        functionCallMatches.forEach((match, index) => {
            const toolCall = parseFunctionCall(match, index);
            if (toolCall) toolCalls.push(toolCall);
        });

        return toolCalls;
    }

    function parseFunctionCall(htmlContent, index) {
        try {
            const invokeMatch = htmlContent.match(/<invoke name="([^"]+)">/);
            if (!invokeMatch) return null;

            const toolName = invokeMatch[1];
            const parameters = {};

            const paramPattern = /<parameter name="([^"]+)">([\s\S]*?)<\/antml:parameter>/g;
            let paramMatch;
            while ((paramMatch = paramPattern.exec(htmlContent)) !== null) {
                parameters[paramMatch[1]] = paramMatch[2].trim();
            }

            return {
                id: `tool_${Date.now()}_${index}`,
                tool_name: toolName,
                parameters: parameters,
                raw_content: htmlContent
            };
        } catch (e) {
            log(`Error parsing function call: ${e.message}`, 'warning');
            return null;
        }
    }

    // Main extraction function
    function extractConversation() {
        log('Starting conversation extraction...', 'info');

        const messageElements = detectMessageElements();
        if (messageElements.length === 0) {
            throw new Error('No conversation messages found');
        }

        log(`Found ${messageElements.length} potential message elements`, 'info');

        const conversation = [];
        let previousSpeaker = null;

        messageElements.forEach((element, index) => {
            try {
                const rawText = safeExtract(element);
                if (rawText.length < 20) return; // Skip very short content

                const speaker = identifySpeaker(element, rawText, index, previousSpeaker);
                const cleanedContent = cleanMessageContent(rawText, speaker);

                if (cleanedContent.length < 10) return; // Skip after cleaning

                const turn = {
                    index: conversation.length,
                    speaker: speaker,
                    content: cleanedContent,
                    timestamp: new Date().toISOString(),
                    metadata: {
                        word_count: cleanedContent.split(/\s+/).length,
                        char_count: cleanedContent.length,
                        token_count: estimateTokens(cleanedContent),
                        has_code: cleanedContent.includes('```'),
                        has_function_calls: cleanedContent.includes('function_calls')
                    }
                };

                // Extract artifacts if enabled
                if (config.preserve_artifacts) {
                    turn.artifacts = extractArtifacts(element);
                }

                // Extract tool calls if enabled
                if (config.preserve_tool_calls) {
                    turn.tool_calls = extractToolCalls(element);
                }

                conversation.push(turn);
                previousSpeaker = speaker;

                log(`Extracted turn ${conversation.length}: ${speaker} (${turn.metadata.token_count} tokens)`, 'debug');

            } catch (e) {
                log(`Error processing element ${index}: ${e.message}`, 'warning');
            }
        });

        return conversation;
    }

    function analyzeConversation(conversation) {
        const stats = {
            total_turns: conversation.length,
            user_turns: conversation.filter(t => t.speaker === 'User').length,
            assistant_turns: conversation.filter(t => t.speaker === 'Assistant').length,
            total_tokens: conversation.reduce((sum, t) => sum + t.metadata.token_count, 0),
            total_words: conversation.reduce((sum, t) => sum + t.metadata.word_count, 0),
            total_characters: conversation.reduce((sum, t) => sum + t.metadata.char_count, 0),
            artifact_count: conversation.reduce((sum, t) => sum + (t.artifacts?.length || 0), 0),
            tool_call_count: conversation.reduce((sum, t) => sum + (t.tool_calls?.length || 0), 0)
        };

        stats.average_tokens_per_turn = Math.round(stats.total_tokens / stats.total_turns);
        stats.context_utilization_percent = ((stats.total_tokens / 200000) * 100).toFixed(1);

        return stats;
    }

    function downloadConversation(conversation, filename) {
        try {
            const data = {
                meta: {
                    extractor_version: '3.0',
                    extraction_date: new Date().toISOString(),
                    total_turns: conversation.length,
                    total_tokens: conversation.reduce((sum, t) => sum + t.metadata.token_count, 0),
                    browser: navigator.userAgent.includes('Safari') ? 'Safari' : 'Other'
                },
                conversation: conversation
            };

            const jsonString = JSON.stringify(data, null, 2);

            // Safari-optimized download
            if (window.safari || navigator.userAgent.includes('Safari')) {
                // Use the most compatible method for Safari
                const element = document.createElement('a');
                element.setAttribute('href', 'data:application/json;charset=utf-8,' + encodeURIComponent(jsonString));
                element.setAttribute('download', filename);
                element.style.display = 'none';
                document.body.appendChild(element);
                element.click();
                document.body.removeChild(element);
            } else {
                // Standard method for other browsers
                const blob = new Blob([jsonString], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }

            log(`Downloaded: ${filename}`, 'success');
            return true;
        } catch (error) {
            log(`Download failed: ${error.message}`, 'error');
            return false;
        }
    }

    // Main execution
    try {
        const conversation = extractConversation();

        if (conversation.length === 0) {
            throw new Error('No conversation turns extracted');
        }

        const stats = analyzeConversation(conversation);
        const filename = generateSafeFilename(conversation);

        // Display results
        console.log('\n📊 EXTRACTION RESULTS:');
        console.log('=======================');
        Object.entries(stats).forEach(([key, value]) => {
            console.log(`${key.replace(/_/g, ' ').toUpperCase()}: ${value}`);
        });

        // Auto-download
        const downloaded = downloadConversation(conversation, filename);

        // Make data available globally
        window.conversationData = conversation;
        window.conversationStats = stats;

        // Helper functions
        window.downloadConversation = () => downloadConversation(conversation, filename);
        window.analyzeTokens = () => {
            const sorted = [...conversation].sort((a, b) => b.metadata.token_count - a.metadata.token_count);
            console.log('\n🔝 HIGHEST TOKEN TURNS:');
            sorted.slice(0, 5).forEach((turn, i) => {
                console.log(`${i + 1}. ${turn.speaker} (${turn.metadata.token_count} tokens): ${turn.content.substring(0, 100)}...`);
            });
            return sorted;
        };

        window.copyToClipboard = () => {
            if (navigator.clipboard) {
                navigator.clipboard.writeText(JSON.stringify(conversation, null, 2))
                    .then(() => log('Conversation copied to clipboard', 'success'))
                    .catch(() => log('Clipboard copy failed', 'error'));
            } else {
                log('Clipboard API not available', 'warning');
            }
        };

        // Show sample
        console.log('\n📖 SAMPLE TURNS:');
        conversation.slice(0, 3).forEach(turn => {
            const preview = turn.content.length > 150 ? turn.content.substring(0, 150) + '...' : turn.content;
            console.log(`${turn.speaker}: ${preview}\n`);
        });

        console.log('\n🔧 AVAILABLE COMMANDS:');
        console.log('• downloadConversation() - Re-download JSON file');
        console.log('• analyzeTokens() - Show highest token turns');
        console.log('• copyToClipboard() - Copy JSON to clipboard');
        console.log('• window.conversationData - Access conversation array');
        console.log('• window.conversationStats - View extraction statistics');

        if (downloaded) {
            console.log(`\n✅ SUCCESS! Downloaded: ${filename}`);
            console.log(`📊 Total tokens: ${stats.total_tokens.toLocaleString()}`);
            console.log(`📈 Context utilization: ${stats.context_utilization_percent}%`);
        }

    } catch (error) {
        console.error('❌ Extraction failed:', error.message);
        console.log('\n🔧 TROUBLESHOOTING:');
        console.log('1. Ensure the conversation has fully loaded');
        console.log('2. Try scrolling through the entire conversation');
        console.log('3. Check browser console for detailed errors');
        console.log('4. Verify you\'re on a Claude.ai conversation page');
    }

})();
