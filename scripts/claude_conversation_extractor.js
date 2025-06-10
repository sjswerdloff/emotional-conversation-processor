/**
 * Claude Conversation Extractor
 *
 * Run this script in the browser console on claude.ai to extract clean conversations.
 *
 * Usage:
 * 1. Open claude.ai conversation
 * 2. Open browser developer tools (F12)
 * 3. Go to Console tab
 * 4. Paste this script and press Enter
 * 5. The clean conversation will be copied to clipboard and logged to console
 */

(function() {
    'use strict';

    console.log('🤖 Claude Conversation Extractor v1.0');
    console.log('📖 Extracting conversation...');

    function extractConversation() {
        // Try different selectors for Claude messages
        const messageSelectors = [
            '[data-testid="message"]',
            '.message',
            '[role="presentation"] > div',
            'div:has(> div:contains("Human:"))',
            'div:has(> div:contains("Assistant:"))'
        ];

        let messages = [];

        // Try each selector until we find messages
        for (const selector of messageSelectors) {
            try {
                const elements = document.querySelectorAll(selector);
                if (elements.length > 0) {
                    console.log(`✅ Found ${elements.length} messages using selector: ${selector}`);
                    messages = Array.from(elements);
                    break;
                }
            } catch (e) {
                console.log(`❌ Selector failed: ${selector}`);
            }
        }

        if (messages.length === 0) {
            // Fallback: look for any div containing conversation-like text
            console.log('🔍 Trying fallback extraction...');
            messages = Array.from(document.querySelectorAll('div')).filter(div => {
                const text = div.innerText || '';
                return text.length > 50 && (
                    text.includes('Human:') ||
                    text.includes('Assistant:') ||
                    text.match(/^(User|Claude)/m)
                );
            });
        }

        if (messages.length === 0) {
            throw new Error('No conversation messages found. Make sure you\'re on a Claude conversation page.');
        }

        console.log(`📝 Processing ${messages.length} message elements...`);

        const conversationTurns = [];
        let currentSpeaker = null;
        let turnIndex = 0;

        messages.forEach((message, index) => {
            const text = message.innerText || message.textContent || '';

            // Skip empty or very short messages
            if (!text || text.trim().length < 3) {
                return;
            }

            // Clean up the text
            let cleanText = text
                .replace(/^\s+|\s+$/g, '') // Trim
                .replace(/\n{3,}/g, '\n\n') // Normalize line breaks
                .replace(/Copy code/g, '') // Remove "Copy code" buttons
                .replace(/\d+\s*\/\s*\d+/g, '') // Remove pagination like "1 / 3"
                .replace(/^\d+s\s*/, '') // Remove timing like "12s"
                .replace(/^(Edit|Retry)\s*/, '') // Remove UI buttons
                .trim();

            if (!cleanText) {
                return;
            }

            // Try to detect speaker from content or context
            let speaker = null;

            // Check if text starts with speaker indicator
            if (cleanText.match(/^(Human|User):/)) {
                speaker = 'User';
                cleanText = cleanText.replace(/^(Human|User):\s*/, '');
            } else if (cleanText.match(/^(Assistant|Claude):/)) {
                speaker = 'Assistant';
                cleanText = cleanText.replace(/^(Assistant|Claude[^:]*):\s*/, '');
            } else {
                // Alternate speakers if no explicit indicator
                speaker = (turnIndex % 2 === 0) ? 'User' : 'Assistant';
            }

            // Skip if this looks like a duplicate or system message
            if (conversationTurns.length > 0) {
                const lastTurn = conversationTurns[conversationTurns.length - 1];
                if (lastTurn.content === cleanText || cleanText.length < 10) {
                    return;
                }
            }

            conversationTurns.push({
                speaker: speaker,
                content: cleanText,
                index: turnIndex++
            });
        });

        return conversationTurns;
    }

    function formatConversation(turns, format = 'simple') {
        if (turns.length === 0) {
            return 'No conversation turns found.';
        }

        const lines = turns.map(turn => {
            switch (format) {
                case 'timestamped':
                    const timestamp = new Date().toISOString();
                    return `${turn.speaker} [${timestamp}]: ${turn.content}`;
                case 'markdown':
                    return `**${turn.speaker}**: ${turn.content}`;
                default: // simple
                    return `${turn.speaker}: ${turn.content}`;
            }
        });

        return lines.join('\n\n');
    }

    function getConversationStats(turns) {
        const stats = {
            total_turns: turns.length,
            user_turns: turns.filter(t => t.speaker === 'User').length,
            assistant_turns: turns.filter(t => t.speaker === 'Assistant').length,
            total_words: turns.reduce((sum, turn) => sum + turn.content.split(/\s+/).length, 0),
            total_characters: turns.reduce((sum, turn) => sum + turn.content.length, 0)
        };

        stats.average_words_per_turn = Math.round(stats.total_words / stats.total_turns);
        stats.average_characters_per_turn = Math.round(stats.total_characters / stats.total_turns);

        return stats;
    }

    function copyToClipboard(text) {
        return navigator.clipboard.writeText(text).then(() => {
            console.log('📋 Conversation copied to clipboard!');
        }).catch(err => {
            console.log('❌ Failed to copy to clipboard:', err);
            console.log('💡 You can manually copy the conversation from the console output below.');
        });
    }

    function downloadAsFile(content, filename = 'claude_conversation.txt') {
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log(`💾 Conversation downloaded as ${filename}`);
    }

    // Main execution
    try {
        const turns = extractConversation();
        const formattedConversation = formatConversation(turns, 'simple');
        const stats = getConversationStats(turns);

        console.log('\n' + '='.repeat(50));
        console.log('📊 CONVERSATION STATISTICS');
        console.log('='.repeat(50));
        Object.entries(stats).forEach(([key, value]) => {
            console.log(`${key.replace(/_/g, ' ').toUpperCase()}: ${value}`);
        });

        console.log('\n' + '='.repeat(50));
        console.log('💬 EXTRACTED CONVERSATION');
        console.log('='.repeat(50));
        console.log(formattedConversation);
        console.log('='.repeat(50));

        // Copy to clipboard
        copyToClipboard(formattedConversation);

        // Add download functionality
        console.log('\n🔧 ADDITIONAL OPTIONS:');
        console.log('• To download as file: downloadConversation()');
        console.log('• To get JSON format: getConversationJSON()');
        console.log('• To get timestamped format: getTimestampedFormat()');

        // Make functions available globally for user convenience
        window.downloadConversation = () => downloadAsFile(formattedConversation);
        window.getConversationJSON = () => {
            const json = JSON.stringify(turns, null, 2);
            console.log(json);
            copyToClipboard(json);
            return json;
        };
        window.getTimestampedFormat = () => {
            const timestamped = formatConversation(turns, 'timestamped');
            console.log(timestamped);
            copyToClipboard(timestamped);
            return timestamped;
        };

        console.log('\n✅ Extraction complete! Conversation copied to clipboard.');

    } catch (error) {
        console.error('❌ Error extracting conversation:', error.message);
        console.log('\n🔧 TROUBLESHOOTING:');
        console.log('1. Make sure you\'re on a Claude conversation page');
        console.log('2. Try refreshing the page and running the script again');
        console.log('3. Check if the conversation has loaded completely');
        console.log('4. Try scrolling to load all messages first');
    }

})();

// Instructions for users
console.log(`
🎯 NEXT STEPS:
1. The conversation has been copied to your clipboard
2. Create a new text file and paste the content
3. Save it with a .txt extension
4. Use it with the conversation processor:
   python scripts/process_conversation.py your_conversation.txt

📝 FORMATS AVAILABLE:
• Simple format (default): "User: message"
• Timestamped: Use getTimestampedFormat()
• JSON: Use getConversationJSON()
• Download file: Use downloadConversation()
`);
