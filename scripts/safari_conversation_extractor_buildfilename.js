/**
 * JSON-Focused Claude Conversation Extractor
 * Optimized for clean JSON output and reliable downloads
 */

(function () {
    'use strict';

    console.log('📋 JSON Conversation Extractor v1.1 - Starting...');

    function estimateTokens(text) {
        // Simple estimation: ~0.75 tokens per word + adjustments for special content
        const words = text.split(/\s+/).length;
        const codeBlocks = (text.match(/```[\s\S]*?```/g) || []).length;
        const functionCalls = (text.match(/<function_calls>[\s\S]*?<\/antml:function_calls>/g) || []).length;
        return Math.ceil(words * 0.75) + (codeBlocks * 10) + (functionCalls * 20);
    }

    function buildConversationFilename(fallbackName = 'claude_conversation') {
        // Try multiple selectors to find conversation title
        const titleSelectors = [
            // Claude interface title selectors
            '[data-testid="conversation-title"]',
            '[data-testid="chat-title"]',
            'h1',
            'title',
            '.conversation-title',
            '.chat-title',
            // Meta tags
            'meta[property="og:title"]',
            'meta[name="title"]'
        ];

        let conversationTitle = null;

        // Try each selector until we find a title
        for (const selector of titleSelectors) {
            try {
                const element = document.querySelector(selector);
                if (element) {
                    let titleText = '';

                    if (selector.startsWith('meta')) {
                        titleText = element.getAttribute('content') || '';
                    } else if (element.tagName === 'TITLE') {
                        titleText = element.textContent || '';
                    } else {
                        titleText = element.textContent || element.innerText || '';
                    }

                    // Clean and validate the title
                    titleText = titleText.trim();

                    // Skip generic/empty titles
                    if (titleText &&
                        titleText.length > 2 &&
                        !titleText.toLowerCase().includes('claude.ai') &&
                        !titleText.toLowerCase().includes('anthropic') &&
                        titleText !== 'Claude') {

                        conversationTitle = titleText;
                        console.log(`✅ Found title with selector "${selector}": "${titleText}"`);
                        break;
                    }
                }
            } catch (e) {
                console.log(`❌ Selector "${selector}" failed:`, e.message);
            }
        }

        // Fallback: try to extract from page content
        if (!conversationTitle) {
            console.log('🔍 Trying content-based title extraction...');

            // Look for conversation starter patterns
            const firstUserMessage = document.body.textContent.match(/(?:Human|User):\s*(.{10,100}?)(?:\n|$)/);
            if (firstUserMessage) {
                conversationTitle = firstUserMessage[1].trim();
                console.log(`💡 Extracted from first message: "${conversationTitle}"`);
            }
        }

        // Sanitize title for filename
        let sanitizedTitle = fallbackName;
        if (conversationTitle) {
            sanitizedTitle = conversationTitle
                // Remove or replace invalid filename characters
                .replace(/[<>:"/\\|?*]/g, '-')
                .replace(/\s+/g, '_')
                .replace(/_{2,}/g, '_')
                .replace(/^_+|_+$/g, '')
                .substring(0, 50) // Limit length
                .trim();

            // Ensure we have something valid
            if (sanitizedTitle.length < 2) {
                sanitizedTitle = fallbackName;
            }
        }

        // Generate timestamp
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');

        // Build final filename
        const filename = `${sanitizedTitle}_${timestamp}.json`;

        console.log('📁 Generated filename:', filename);

        return {
            filename: filename,
            title: conversationTitle,
            sanitizedTitle: sanitizedTitle,
            timestamp: timestamp
        };
    }


    function cleanContent(text) {
        if (!text) return '';

        return text
            // Remove sharing metadata
            .replace(/^.*?Shared by.*?\n/s, '')
            .replace(/This is a copy of a chat.*?Report\s*/s, '')
            .replace(/Files hidden in shared chats\s*/g, '')

            // Remove Claude's internal reasoning (the "thinking" parts)
            .replace(/The user has attached.*?I should maintain.*?assessment\./s, '')
            .replace(/Given their preference.*?thorough in my assessment\./s, '')
            .replace(/^Analyzed [^.]*\.\s*/gm, '')
            .replace(/^Pondered [^.]*\.\s*/gm, '')

            // Remove UI artifacts
            .replace(/\n*\d+s\n*/g, '\n')
            .replace(/\n*Edit\n*/g, '')
            .replace(/\n*Retry\n*/g, '')

            // Clean whitespace
            .replace(/\n{3,}/g, '\n\n')
            .trim();
    }

    function extractToJSON() {
        console.log('🔍 Extracting conversation...');

        // Get all text content
        const bodyText = document.body.innerText || document.body.textContent || '';
        const cleanedText = cleanContent(bodyText);

        // Split into potential conversation segments
        const segments = cleanedText.split(/\n\n+/).filter(segment =>
            segment.trim().length > 20
        );

        console.log(`📝 Found ${segments.length} text segments`);

        const conversation = [];
        let currentSpeaker = 'User'; // Start with User

        for (let i = 0; i < segments.length; i++) {
            const segment = segments[i].trim();

            // Skip very short or metadata-like segments
            if (segment.length < 30 ||
                segment.includes('claude.ai') ||
                segment.includes('Anthropic') ||
                segment.match(/^\d+\/\d+$/)) {
                continue;
            }

            // Try to detect speaker from content
            let speaker = currentSpeaker;
            let content = segment;

            // Clean speaker prefixes if they exist
            if (content.match(/^(User|Human):/i)) {
                speaker = 'User';
                content = content.replace(/^(User|Human):\s*/i, '');
            } else if (content.match(/^(Assistant|Claude):/i)) {
                speaker = 'Assistant';
                content = content.replace(/^(Assistant|Claude[^:]*?):\s*/i, '');
            } else {
                // Heuristics for speaker detection
                if (content.length < 200 && content.includes('?')) {
                    speaker = 'User';
                } else if (content.length > 500 || content.includes('##') || content.includes('1.')) {
                    speaker = 'Assistant';
                } else {
                    // Alternate speakers
                    speaker = (conversation.length % 2 === 0) ? 'User' : 'Assistant';
                }
            }

            // Only add if we have substantial content
            if (content.length > 20) {
                const cleanContent = content.trim();
                conversation.push({
                    speaker: speaker,
                    content: cleanContent,
                    index: conversation.length,
                    word_count: cleanContent.split(/\s+/).length,
                    char_count: cleanContent.length,
                    token_count: estimateTokens(cleanContent)
                });

                currentSpeaker = speaker === 'User' ? 'Assistant' : 'User';
            }
        }

        console.log(`✅ Extracted ${conversation.length} conversation turns`);
        return conversation;
    }

    function downloadJSON(data, filename = null) {
        try {

            if (!filename) {
                const filenameData = buildConversationFilename('claude_conversation');
                filename = filenameData.filename;
            }


            const jsonString = JSON.stringify(data, null, 2);
            const blob = new Blob([jsonString], { type: 'application/json;charset=utf-8' });
            const url = URL.createObjectURL(blob);

            const downloadLink = document.createElement('a');
            downloadLink.href = url;
            downloadLink.download = filename;
            downloadLink.style.display = 'none';

            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);

            // Cleanup
            setTimeout(() => URL.revokeObjectURL(url), 1000);

            console.log(`💾 Downloaded: ${filename}`);
            return true;
        } catch (error) {
            console.error('❌ Download failed:', error);
            return false;
        }
    }

    function convertToText(conversation) {
        return conversation.map(turn =>
            `${turn.speaker}: ${turn.content}`
        ).join('\n\n');
    }

    // Main execution
    try {
        const conversation = extractToJSON();

        if (conversation.length === 0) {
            throw new Error('No conversation content found');
        }

        // Show statistics
        const stats = {
            total_turns: conversation.length,
            user_turns: conversation.filter(t => t.speaker === 'User').length,
            assistant_turns: conversation.filter(t => t.speaker === 'Assistant').length,
            total_words: conversation.reduce((sum, t) => sum + t.word_count, 0),
            total_characters: conversation.reduce((sum, t) => sum + t.char_count, 0),
            total_tokens: conversation.reduce((sum, t) => sum + t.token_count, 0),
            average_tokens_per_turn: Math.round(conversation.reduce((sum, t) => sum + t.token_count, 0) / conversation.length),
            estimated_context_usage: conversation.reduce((sum, t) => sum + t.token_count, 0) + 2000,
            context_utilization_percent: (((conversation.reduce((sum, t) => sum + t.token_count, 0) + 2000) / 200000) * 100).toFixed(1)
        };

        console.log('\n📊 EXTRACTION RESULTS:');
        console.log('========================');
        Object.entries(stats).forEach(([key, value]) => {
            console.log(`${key.toUpperCase()}: ${value}`);
        });

        // Store globally for manual access
        window.conversationData = conversation;

        // Auto-download JSON
        const downloaded = downloadJSON(conversation);

        if (downloaded) {
            console.log('\n🎉 SUCCESS! JSON file downloaded automatically');
        } else {
            console.log('\n⚠️ Auto-download failed, trying manual methods...');
        }

        // Provide manual options
        console.log('\n🔧 MANUAL OPTIONS:');
        console.log('• Download JSON: downloadConversationJSON()');
        console.log('• Download Text: downloadConversationText()');
        console.log('• View JSON: console.log(window.conversationData)');
        console.log('• Copy JSON: copyConversationJSON()');
        console.log('• Analyze tokens: analyzeTokens()');

        // Set up global functions
        window.downloadConversationJSON = () => downloadJSON(conversation);
        window.downloadConversationText = () => {
            const textContent = convertToText(conversation);
            const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'claude_conversation.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            console.log('💾 Text file downloaded');
        };

        window.copyConversationJSON = () => {
            const jsonString = JSON.stringify(conversation, null, 2);
            navigator.clipboard.writeText(jsonString).then(() => {
                console.log('📋 JSON copied to clipboard');
            }).catch(() => {
                console.log('❌ Clipboard failed - JSON is in window.conversationData');
            });
        };

        window.analyzeTokens = () => {
            console.log('\n🔢 TOKEN ANALYSIS:');
            console.log('========================');
            console.log(`Total estimated tokens: ${stats.total_tokens.toLocaleString()}`);
            console.log(`Average per turn: ${stats.average_tokens_per_turn}`);
            console.log(`Estimated context usage: ${stats.estimated_context_usage.toLocaleString()}`);
            console.log(`Context utilization: ${stats.context_utilization_percent}%`);

            // Show top token-consuming turns
            const sortedTurns = [...conversation].sort((a, b) => b.token_count - a.token_count);

            console.log('\n🔝 HIGHEST TOKEN TURNS:');
            sortedTurns.slice(0, 3).forEach((turn, i) => {
                const preview = turn.content.substring(0, 80) + '...';
                console.log(`${i + 1}. ${turn.speaker} (${turn.token_count} tokens): ${preview}`);
            });

            return {
                total_tokens: stats.total_tokens,
                context_usage: stats.estimated_context_usage,
                utilization_percent: stats.context_utilization_percent,
                turns_by_tokens: sortedTurns.slice(0, 5).map(t => ({
                    speaker: t.speaker,
                    tokens: t.token_count,
                    preview: t.content.substring(0, 100)
                }))
            };
        };

        // Show sample of first few turns
        console.log('\n📖 SAMPLE CONVERSATION:');
        console.log('========================');
        conversation.slice(0, 3).forEach(turn => {
            const preview = turn.content.length > 100
                ? turn.content.substring(0, 100) + '...'
                : turn.content;
            console.log(`${turn.speaker}: ${preview}\n`);
        });

        console.log('✅ Extraction complete! Check your Downloads folder for the JSON file.');

    } catch (error) {
        console.error('❌ Extraction failed:', error.message);
        console.log('\n🔧 TROUBLESHOOTING:');
        console.log('1. Make sure the conversation has fully loaded');
        console.log('2. Try scrolling through the entire conversation first');
        console.log('3. Check if you\'re on the correct Claude.ai page');
    }

})();
