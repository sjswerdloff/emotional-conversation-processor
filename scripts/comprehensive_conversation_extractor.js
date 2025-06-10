/**
 * Comprehensive Claude Conversation State Extractor
 *
 * This advanced extractor captures not just conversation text, but all the contextual
 * state needed to replay a conversation with maximum fidelity in a new instance.
 *
 * Extracts:
 * - Complete conversation flow with metadata
 * - All tool calls and results
 * - Created artifacts and their evolution
 * - Document context and processing history
 * - User preferences and behavioral indicators
 * - System configuration and environment
 * - Conversation branching/retries/edits
 * - Timing and sequencing information
 *
 * Usage: Run in browser console on claude.ai conversation page
 */

(function() {
    'use strict';

    console.log('🧠 Comprehensive Claude Conversation State Extractor v1.0');
    console.log('📊 Extracting complete conversation state for replay...');

    const extractorConfig = {
        includeTimestamps: true,
        includeMetadata: true,
        includeArtifacts: true,
        includeDocuments: true,
        includeToolCalls: true,
        includePreferences: true,
        includeEnvironment: true,
        maxContentLength: 50000, // Prevent huge extractions
        verboseLogging: false
    };

    function log(message, level = 'info') {
        const prefix = {
            'debug': '🔍',
            'info': '📖',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }[level] || '📖';

        if (extractorConfig.verboseLogging || level !== 'debug') {
            console.log(`${prefix} ${message}`);
        }
    }

    function generateStateId() {
        return `state_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    function detectEnvironment() {
        log('Detecting environment and system context...', 'debug');

        return {
            platform: navigator.platform,
            userAgent: navigator.userAgent,
            language: navigator.language,
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            },
            url: window.location.href,
            timestamp: new Date().toISOString(),
            claude_version: detectClaudeVersion(),
            available_tools: detectAvailableTools(),
            session_context: detectSessionContext()
        };
    }

    function detectClaudeVersion() {
        // Try to detect Claude version from page elements
        const versionIndicators = [
            document.querySelector('[data-testid="model-name"]'),
            document.querySelector('.model-name'),
            document.querySelector('[class*="model"]'),
        ].filter(Boolean);

        for (const element of versionIndicators) {
            const text = element.textContent || element.innerText || '';
            if (text.includes('Claude') || text.includes('Sonnet') || text.includes('Opus')) {
                return text.trim();
            }
        }

        // Fallback detection from page title or other indicators
        if (document.title.includes('Claude')) {
            return document.title;
        }

        return 'Claude (version unknown)';
    }

    function detectAvailableTools() {
        const tools = [];

        // Look for tool indicators in the UI
        const toolSelectors = [
            '[data-testid="tool"]',
            '[class*="tool"]',
            '[aria-label*="tool"]',
            '.function-call',
            '[data-function]'
        ];

        toolSelectors.forEach(selector => {
            try {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    const toolName = el.getAttribute('data-tool') ||
                                   el.getAttribute('data-function') ||
                                   el.textContent?.match(/\w+/)?.[0];
                    if (toolName && !tools.includes(toolName)) {
                        tools.push(toolName);
                    }
                });
            } catch (e) {
                log(`Tool detection error for ${selector}: ${e.message}`, 'debug');
            }
        });

        // Common tools based on UI elements
        const commonTools = [
            'web_search', 'artifacts', 'repl', 'computer_use',
            'text_editor', 'bash', 'python'
        ];

        commonTools.forEach(tool => {
            if (document.body.innerHTML.includes(tool) && !tools.includes(tool)) {
                tools.push(tool);
            }
        });

        return tools;
    }

    function detectSessionContext() {
        return {
            has_artifacts: !!document.querySelector('[data-testid="artifact"]'),
            has_documents: !!document.querySelector('[data-testid="document"]'),
            has_tools: !!document.querySelector('[data-testid="tool-call"]'),
            conversation_length: document.querySelectorAll('[data-testid="message"]').length,
            is_shared: window.location.href.includes('/share/') ||
                      !!document.querySelector('[data-testid="shared-chat"]'),
            has_regenerations: !!document.querySelector('[data-testid="regenerate"]'),
            has_branches: !!document.querySelector('[data-testid="branch"]')
        };
    }

    function extractUserPreferences() {
        log('Extracting user preferences and behavioral indicators...', 'debug');

        const preferences = {
            detected_style: 'unknown',
            response_length_preference: 'unknown',
            technical_level: 'unknown',
            formatting_preference: 'unknown',
            language: navigator.language,
            indicators: []
        };

        // Analyze conversation for style indicators
        const messages = document.querySelectorAll('[data-testid="message"]');
        const userMessages = Array.from(messages).filter(msg =>
            msg.textContent?.includes('User:') ||
            msg.querySelector('[data-testid="user-message"]')
        );

        // Detect formality level
        const formalIndicators = ['please', 'thank you', 'could you', 'would you'];
        const casualIndicators = ['hey', 'awesome', 'cool', 'thanks'];

        let formalCount = 0, casualCount = 0;

        userMessages.forEach(msg => {
            const text = (msg.textContent || '').toLowerCase();
            formalIndicators.forEach(indicator => {
                if (text.includes(indicator)) formalCount++;
            });
            casualIndicators.forEach(indicator => {
                if (text.includes(indicator)) casualCount++;
            });
        });

        if (formalCount > casualCount) {
            preferences.detected_style = 'formal';
        } else if (casualCount > formalCount) {
            preferences.detected_style = 'casual';
        }

        // Detect technical level
        const technicalTerms = ['api', 'function', 'algorithm', 'database', 'framework'];
        const techTermCount = userMessages.reduce((count, msg) => {
            const text = (msg.textContent || '').toLowerCase();
            return count + technicalTerms.filter(term => text.includes(term)).length;
        }, 0);

        if (techTermCount > 5) {
            preferences.technical_level = 'high';
        } else if (techTermCount > 2) {
            preferences.technical_level = 'medium';
        } else {
            preferences.technical_level = 'low';
        }

        // Response length preference
        const avgResponseLength = Array.from(messages)
            .filter(msg => !msg.textContent?.includes('User:'))
            .reduce((sum, msg, _, arr) => sum + (msg.textContent?.length || 0) / arr.length, 0);

        if (avgResponseLength > 2000) {
            preferences.response_length_preference = 'detailed';
        } else if (avgResponseLength > 500) {
            preferences.response_length_preference = 'moderate';
        } else {
            preferences.response_length_preference = 'concise';
        }

        return preferences;
    }

    function extractArtifacts() {
        log('Extracting artifacts and their evolution...', 'debug');

        const artifacts = [];

        // Find artifact containers
        const artifactElements = document.querySelectorAll('[data-testid="artifact"], .artifact, [class*="artifact"]');

        artifactElements.forEach((element, index) => {
            try {
                const artifact = {
                    id: element.getAttribute('data-artifact-id') || `artifact_${index}`,
                    type: detectArtifactType(element),
                    title: extractArtifactTitle(element),
                    content: extractArtifactContent(element),
                    language: detectArtifactLanguage(element),
                    created_at: extractArtifactTimestamp(element),
                    metadata: extractArtifactMetadata(element),
                    updates: extractArtifactUpdates(element)
                };

                if (artifact.content) {
                    artifacts.push(artifact);
                }
            } catch (e) {
                log(`Error extracting artifact ${index}: ${e.message}`, 'warning');
            }
        });

        return artifacts;
    }

    function detectArtifactType(element) {
        const content = element.textContent || '';
        const classes = element.className || '';

        if (classes.includes('code') || content.includes('```')) return 'code';
        if (classes.includes('html') || content.includes('<!DOCTYPE')) return 'html';
        if (classes.includes('markdown') || content.includes('# ')) return 'markdown';
        if (classes.includes('text')) return 'text';

        return 'unknown';
    }

    function extractArtifactTitle(element) {
        const titleSelectors = [
            '.artifact-title',
            '[data-testid="artifact-title"]',
            'h1', 'h2', 'h3'
        ];

        for (const selector of titleSelectors) {
            const titleEl = element.querySelector(selector);
            if (titleEl) {
                return titleEl.textContent?.trim();
            }
        }

        return 'Untitled Artifact';
    }

    function extractArtifactContent(element) {
        const contentSelectors = [
            '.artifact-content',
            '[data-testid="artifact-content"]',
            'pre', 'code', '.content'
        ];

        for (const selector of contentSelectors) {
            const contentEl = element.querySelector(selector);
            if (contentEl) {
                return contentEl.textContent || contentEl.innerHTML;
            }
        }

        return element.textContent?.trim();
    }

    function detectArtifactLanguage(element) {
        const content = element.textContent || '';
        const classes = element.className || '';

        if (classes.includes('python') || content.includes('def ') || content.includes('import ')) return 'python';
        if (classes.includes('javascript') || content.includes('function ') || content.includes('const ')) return 'javascript';
        if (classes.includes('html') || content.includes('<!DOCTYPE')) return 'html';
        if (classes.includes('css') || content.includes('{') && content.includes('}')) return 'css';
        if (classes.includes('sql') || content.includes('SELECT ')) return 'sql';
        if (classes.includes('bash') || content.includes('#!/bin/bash')) return 'bash';

        return 'text';
    }

    function extractArtifactTimestamp(element) {
        const timeEl = element.querySelector('time, [data-timestamp], .timestamp');
        if (timeEl) {
            return timeEl.getAttribute('datetime') || timeEl.textContent;
        }
        return new Date().toISOString();
    }

    function extractArtifactMetadata(element) {
        return {
            size: element.textContent?.length || 0,
            dom_id: element.id || null,
            classes: element.className || null,
            data_attributes: Array.from(element.attributes)
                .filter(attr => attr.name.startsWith('data-'))
                .reduce((obj, attr) => ({ ...obj, [attr.name]: attr.value }), {})
        };
    }

    function extractArtifactUpdates(element) {
        // Look for version/update indicators
        const updateElements = element.querySelectorAll('[data-version], .version, .update');
        return Array.from(updateElements).map(el => ({
            version: el.getAttribute('data-version') || el.textContent?.trim(),
            timestamp: el.getAttribute('data-timestamp') || new Date().toISOString()
        }));
    }

    function extractDocuments() {
        log('Extracting document context and processing history...', 'debug');

        const documents = [];

        // Find document references
        const docElements = document.querySelectorAll('[data-testid="document"], .document, [class*="document"]');

        docElements.forEach((element, index) => {
            try {
                const doc = {
                    id: element.getAttribute('data-document-id') || `doc_${index}`,
                    title: extractDocumentTitle(element),
                    type: extractDocumentType(element),
                    content_preview: extractDocumentPreview(element),
                    processing_info: extractDocumentProcessing(element),
                    metadata: extractDocumentMetadata(element)
                };

                documents.push(doc);
            } catch (e) {
                log(`Error extracting document ${index}: ${e.message}`, 'warning');
            }
        });

        return documents;
    }

    function extractDocumentTitle(element) {
        const titleSelectors = ['.document-title', '[data-testid="document-title"]', '.filename'];

        for (const selector of titleSelectors) {
            const titleEl = element.querySelector(selector);
            if (titleEl) return titleEl.textContent?.trim();
        }

        return element.textContent?.substring(0, 50) + '...';
    }

    function extractDocumentType(element) {
        const content = element.textContent || '';
        const classes = element.className || '';

        if (classes.includes('pdf') || content.includes('.pdf')) return 'pdf';
        if (classes.includes('image') || content.includes('.jpg') || content.includes('.png')) return 'image';
        if (classes.includes('text') || content.includes('.txt')) return 'text';
        if (classes.includes('code') || content.includes('.py') || content.includes('.js')) return 'code';

        return 'unknown';
    }

    function extractDocumentPreview(element) {
        const content = element.textContent || '';
        return content.substring(0, 200) + (content.length > 200 ? '...' : '');
    }

    function extractDocumentProcessing(element) {
        return {
            analyzed: element.textContent?.includes('analyzed') || false,
            summarized: element.textContent?.includes('summary') || false,
            extracted: element.textContent?.includes('extracted') || false,
            processed_at: new Date().toISOString()
        };
    }

    function extractDocumentMetadata(element) {
        return {
            size_estimate: element.textContent?.length || 0,
            dom_id: element.id || null,
            classes: element.className || null
        };
    }

    function extractToolCalls() {
        log('Extracting tool calls and results...', 'debug');

        const toolCalls = [];

        // Find tool call elements
        const toolElements = document.querySelectorAll(
            '[data-testid="tool-call"], .tool-call, .function-call, [class*="tool"]'
        );

        toolElements.forEach((element, index) => {
            try {
                const toolCall = {
                    id: element.getAttribute('data-tool-id') || `tool_${index}`,
                    tool_name: extractToolName(element),
                    parameters: extractToolParameters(element),
                    result: extractToolResult(element),
                    timestamp: extractToolTimestamp(element),
                    status: extractToolStatus(element),
                    metadata: extractToolMetadata(element)
                };

                if (toolCall.tool_name) {
                    toolCalls.push(toolCall);
                }
            } catch (e) {
                log(`Error extracting tool call ${index}: ${e.message}`, 'warning');
            }
        });

        return toolCalls;
    }

    function extractToolName(element) {
        const nameAttrs = ['data-tool', 'data-function', 'data-tool-name'];
        for (const attr of nameAttrs) {
            const value = element.getAttribute(attr);
            if (value) return value;
        }

        const content = element.textContent || '';
        const toolNameMatch = content.match(/(\w+)\s*\(/);
        return toolNameMatch ? toolNameMatch[1] : 'unknown_tool';
    }

    function extractToolParameters(element) {
        const paramElements = element.querySelectorAll('.parameter, [data-parameter]');
        const parameters = {};

        paramElements.forEach(paramEl => {
            const name = paramEl.getAttribute('data-parameter') || paramEl.dataset.name;
            const value = paramEl.textContent || paramEl.value;
            if (name) parameters[name] = value;
        });

        // Fallback: try to parse from text content
        const content = element.textContent || '';
        const paramMatch = content.match(/\((.*?)\)/);
        if (paramMatch) {
            try {
                const paramStr = paramMatch[1];
                if (paramStr.includes(':')) {
                    // Try to parse as key:value pairs
                    paramStr.split(',').forEach(pair => {
                        const [key, value] = pair.split(':').map(s => s.trim());
                        if (key && value) parameters[key] = value;
                    });
                }
            } catch (e) {
                // Ignore parsing errors
            }
        }

        return parameters;
    }

    function extractToolResult(element) {
        const resultSelectors = ['.result', '.output', '[data-testid="tool-result"]'];

        for (const selector of resultSelectors) {
            const resultEl = element.querySelector(selector);
            if (resultEl) {
                return {
                    content: resultEl.textContent?.trim(),
                    type: resultEl.className || 'text',
                    success: !resultEl.className?.includes('error')
                };
            }
        }

        return null;
    }

    function extractToolTimestamp(element) {
        const timeEl = element.querySelector('time, .timestamp, [data-timestamp]');
        if (timeEl) {
            return timeEl.getAttribute('datetime') || timeEl.textContent;
        }
        return new Date().toISOString();
    }

    function extractToolStatus(element) {
        const classes = element.className || '';
        if (classes.includes('success')) return 'success';
        if (classes.includes('error')) return 'error';
        if (classes.includes('pending')) return 'pending';
        return 'completed';
    }

    function extractToolMetadata(element) {
        return {
            duration: element.getAttribute('data-duration'),
            retries: element.getAttribute('data-retries'),
            dom_id: element.id || null,
            classes: element.className || null
        };
    }

    function extractConversationFlow() {
        log('Extracting conversation flow and interaction patterns...', 'debug');

        const messages = document.querySelectorAll('[data-testid="message"], .message');
        const conversationFlow = [];

        messages.forEach((message, index) => {
            try {
                const messageData = {
                    index: index,
                    id: message.getAttribute('data-message-id') || `msg_${index}`,
                    speaker: detectMessageSpeaker(message),
                    content: extractMessageContent(message),
                    timestamp: extractMessageTimestamp(message),
                    type: detectMessageType(message),
                    metadata: extractMessageMetadata(message),
                    interactions: extractMessageInteractions(message)
                };

                conversationFlow.push(messageData);
            } catch (e) {
                log(`Error extracting message ${index}: ${e.message}`, 'warning');
            }
        });

        return conversationFlow;
    }

    function detectMessageSpeaker(element) {
        const speakerIndicators = [
            element.getAttribute('data-speaker'),
            element.querySelector('[data-testid="user-message"]') ? 'User' : null,
            element.querySelector('[data-testid="assistant-message"]') ? 'Assistant' : null
        ].filter(Boolean);

        if (speakerIndicators.length > 0) {
            return speakerIndicators[0];
        }

        const content = element.textContent || '';
        if (content.startsWith('User:') || content.startsWith('Human:')) return 'User';
        if (content.startsWith('Assistant:') || content.startsWith('Claude:')) return 'Assistant';

        return 'Unknown';
    }

    function extractMessageContent(element) {
        const contentEl = element.querySelector('.message-content, [data-testid="message-content"]');
        if (contentEl) {
            return contentEl.textContent || contentEl.innerHTML;
        }

        return element.textContent?.trim();
    }

    function extractMessageTimestamp(element) {
        const timeEl = element.querySelector('time, .timestamp, [data-timestamp]');
        if (timeEl) {
            return timeEl.getAttribute('datetime') || timeEl.textContent;
        }
        return new Date().toISOString();
    }

    function detectMessageType(element) {
        const classes = element.className || '';
        const content = element.textContent || '';

        if (classes.includes('system') || content.includes('[System]')) return 'system';
        if (classes.includes('error') || content.includes('Error:')) return 'error';
        if (element.querySelector('[data-testid="tool-call"]')) return 'tool_call';
        if (element.querySelector('[data-testid="artifact"]')) return 'artifact';

        return 'regular';
    }

    function extractMessageMetadata(element) {
        return {
            word_count: (element.textContent || '').split(/\s+/).length,
            char_count: (element.textContent || '').length,
            has_code: (element.textContent || '').includes('```'),
            has_links: !!element.querySelector('a'),
            has_images: !!element.querySelector('img'),
            dom_id: element.id || null,
            classes: element.className || null
        };
    }

    function extractMessageInteractions(element) {
        const interactions = [];

        // Look for interaction buttons/elements
        const buttons = element.querySelectorAll('button, [role="button"]');
        buttons.forEach(button => {
            const text = button.textContent?.trim();
            if (text) {
                interactions.push({
                    type: 'button',
                    text: text,
                    action: button.getAttribute('data-action') || 'unknown'
                });
            }
        });

        // Look for regeneration indicators
        if (element.querySelector('[data-testid="regenerate"]')) {
            interactions.push({ type: 'regenerate', available: true });
        }

        // Look for edit indicators
        if (element.querySelector('[data-testid="edit"]')) {
            interactions.push({ type: 'edit', available: true });
        }

        return interactions;
    }

    function extractSystemConfiguration() {
        log('Extracting system configuration and context...', 'debug');

        return {
            model_info: {
                name: detectClaudeVersion(),
                features: detectAvailableFeatures()
            },
            ui_state: extractUIState(),
            session_info: extractSessionInfo(),
            capabilities: detectCapabilities()
        };
    }

    function detectAvailableFeatures() {
        const features = [];

        // Check for various UI features
        if (document.querySelector('[data-testid="artifact"]')) features.push('artifacts');
        if (document.querySelector('[data-testid="tool-call"]')) features.push('tool_use');
        if (document.querySelector('[data-testid="document"]')) features.push('document_analysis');
        if (document.querySelector('[data-testid="image"]')) features.push('image_analysis');
        if (document.querySelector('.code-block')) features.push('code_execution');

        return features;
    }

    function extractUIState() {
        return {
            theme: document.body.className?.includes('dark') ? 'dark' : 'light',
            sidebar_open: !!document.querySelector('.sidebar:not([style*="display: none"])'),
            has_notifications: !!document.querySelector('.notification'),
            active_tools: Array.from(document.querySelectorAll('[data-tool][data-active="true"]'))
                .map(el => el.getAttribute('data-tool'))
        };
    }

    function extractSessionInfo() {
        return {
            conversation_id: extractConversationId(),
            session_start: detectSessionStart(),
            message_count: document.querySelectorAll('[data-testid="message"]').length,
            has_context_files: !!document.querySelector('[data-testid="document"]'),
            is_premium: detectPremiumFeatures()
        };
    }

    function extractConversationId() {
        const url = window.location.href;
        const idMatch = url.match(/\/chat\/([a-f0-9-]+)/);
        return idMatch ? idMatch[1] : 'unknown';
    }

    function detectSessionStart() {
        const firstMessage = document.querySelector('[data-testid="message"]');
        if (firstMessage) {
            const timeEl = firstMessage.querySelector('time');
            if (timeEl) {
                return timeEl.getAttribute('datetime') || timeEl.textContent;
            }
        }
        return new Date().toISOString();
    }

    function detectPremiumFeatures() {
        return !!(
            document.querySelector('[data-testid="premium"]') ||
            document.body.innerHTML.includes('Claude Pro') ||
            document.body.innerHTML.includes('premium')
        );
    }

    function detectCapabilities() {
        const capabilities = {
            can_browse_web: false,
            can_execute_code: false,
            can_create_artifacts: false,
            can_analyze_images: false,
            can_process_documents: false
        };

        // Detect capabilities based on UI elements and content
        if (document.querySelector('[data-tool="web_search"]')) capabilities.can_browse_web = true;
        if (document.querySelector('[data-tool="repl"]')) capabilities.can_execute_code = true;
        if (document.querySelector('[data-testid="artifact"]')) capabilities.can_create_artifacts = true;
        if (document.querySelector('[data-testid="image"]')) capabilities.can_analyze_images = true;
        if (document.querySelector('[data-testid="document"]')) capabilities.can_process_documents = true;

        return capabilities;
    }

    function createComprehensiveState() {
        const state = {
            meta: {
                extractor_version: '1.0',
                extraction_id: generateStateId(),
                extracted_at: new Date().toISOString(),
                extraction_type: 'comprehensive_conversation_state'
            },
            environment: extractorConfig.includeEnvironment ? detectEnvironment() : null,
            user_preferences: extractorConfig.includePreferences ? extractUserPreferences() : null,
            conversation_flow: extractConversationFlow(),
            artifacts: extractorConfig.includeArtifacts ? extractArtifacts() : [],
            documents: extractorConfig.includeDocuments ? extractDocuments() : [],
            tool_calls: extractorConfig.includeToolCalls ? extractToolCalls() : [],
            system_config: extractSystemConfiguration(),
            replay_instructions: generateReplayInstructions()
        };

        return state;
    }

    function generateReplayInstructions() {
        return {
            setup_requirements: [
                'Ensure same Claude model/version if possible',
                'Upload any referenced documents before replay',
                'Enable same tools that were used in original conversation',
                'Set similar user preferences if specified'
            ],
            replay_process: [
                'Start new conversation with same model',
                'Apply user preferences and context',
                'Upload documents in original order',
                'Replay conversation flow with tools and artifacts',
                'Maintain same interaction patterns'
            ],
            fidelity_notes: [
                'Tool results may vary due to external data changes',
                'Some system context may not be perfectly replicable',
                'Timing and exact responses may differ',
                'User preferences are inferred and may need adjustment'
            ]
        };
    }

    function downloadState(state, filename) {
        try {
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

            log(`Downloaded comprehensive state: ${filename}`, 'success');
            return true;
        } catch (error) {
            log(`Download failed: ${error.message}`, 'error');
            return false;
        }
    }

    function displayStateSummary(state) {
        const summary = {
            conversation_length: state.conversation_flow.length,
            artifacts_count: state.artifacts.length,
            documents_count: state.documents.length,
            tool_calls_count: state.tool_calls.length,
            model_version: state.system_config.model_info.name,
            available_features: state.system_config.model_info.features,
            user_style: state.user_preferences?.detected_style,
            technical_level: state.user_preferences?.technical_level,
            extraction_size: JSON.stringify(state).length
        };

        console.log('\n' + '='.repeat(70));
        console.log('🧠 COMPREHENSIVE CONVERSATION STATE EXTRACTED');
        console.log('='.repeat(70));

        Object.entries(summary).forEach(([key, value]) => {
            console.log(`   ${key.toUpperCase().replace(/_/g, ' ')}: ${Array.isArray(value) ? value.join(', ') : value}`);
        });

        console.log('\n📊 STATE BREAKDOWN:');
        console.log(`   • Conversation Messages: ${state.conversation_flow.length}`);
        console.log(`   • Created Artifacts: ${state.artifacts.length}`);
        console.log(`   • Processed Documents: ${state.documents.length}`);
        console.log(`   • Tool Interactions: ${state.tool_calls.length}`);
        console.log(`   • Total State Size: ${(JSON.stringify(state).length / 1024).toFixed(1)} KB`);

        return summary;
    }

    // Main execution
    try {
        log('Starting comprehensive conversation state extraction...');

        const state = createComprehensiveState();
        const summary = displayStateSummary(state);

        // Auto-download the state
        const filename = `claude_conversation_state_${state.meta.extraction_id}.json`;
        const downloaded = downloadState(state, filename);

        // Make state available globally
        window.conversationState = state;
        window.downloadConversationState = () => downloadState(state, filename);
        window.getStateJSON = () => {
            const json = JSON.stringify(state, null, 2);
            console.log(json);
            return json;
        };

        console.log('\n🔧 AVAILABLE COMMANDS:');
        console.log('• downloadConversationState() - Re-download state file');
        console.log('• getStateJSON() - Display full JSON in console');
        console.log('• window.conversationState - Access extracted state object');

        console.log('\n🎯 REPLAY READY:');
        console.log('This state package contains everything needed for high-fidelity');
        console.log('conversation replay in a new Claude instance, including:');
        console.log('• Complete conversation flow with metadata');
        console.log('• All tool interactions and results');
        console.log('• Created artifacts and document context');
        console.log('• User preferences and system configuration');
        console.log('• Replay instructions and setup requirements');

        if (downloaded) {
            console.log(`\n✅ Comprehensive state exported successfully!`);
            console.log(`📁 File saved: ${filename}`);
        } else {
            console.log('\n⚠️ Auto-download failed - use downloadConversationState()');
        }

    } catch (error) {
        log(`Extraction failed: ${error.message}`, 'error');
        console.log('\n🔧 TROUBLESHOOTING:');
        console.log('1. Ensure conversation has fully loaded');
        console.log('2. Try on a different conversation page');
        console.log('3. Check browser console for errors');
        console.log('4. Some features may not be available on shared conversations');
    }

})();
