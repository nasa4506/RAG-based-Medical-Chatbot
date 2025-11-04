const { useState, useEffect, useRef } = React;

function MedicalChatbot() {
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const formatTime = () => {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    const handleSend = async () => {
        if (!inputValue.trim() || isLoading) return;

        const userMessage = {
            text: inputValue.trim(),
            sender: 'user',
            time: formatTime()
        };

        setMessages(prev => [...prev, userMessage]);
        setInputValue('');
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage.text })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            const assistantMessage = {
                text: data.answer,
                sender: 'assistant',
                time: formatTime()
            };

            setMessages(prev => [...prev, assistantMessage]);
        } catch (err) {
            console.error('Error:', err);
            setError('Sorry, I encountered an error. Please try again.');
            const errorMessage = {
                text: 'Sorry, I encountered an error processing your request. Please try again.',
                sender: 'assistant',
                time: formatTime(),
                isError: true
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
            inputRef.current?.focus();
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const sampleQuestions = [
        "What is acne?",
        "How to treat a fever?",
        "What are the symptoms of diabetes?",
        "Explain hypertension"
    ];

    const handleSampleQuestion = (question) => {
        setInputValue(question);
    };

    // Function to clean and render text safely - removes all markdown symbols
    const renderText = (text) => {
        if (!text) return { __html: '' };
        
        let html = text;
        
        // Step 1: Convert markdown bold (**text** or __text__) to HTML bold first
        // Process double asterisks/underscores before single ones
        html = html.replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/__([^_\n]+?)__/g, '<strong>$1</strong>');
        
        // Step 2: Convert remaining single asterisks/underscores to italic (only if not part of bold)
        html = html.replace(/([^*]|^)\*([^*\n]+?)\*([^*]|$)/g, '$1<em>$2</em>$3');
        
        // Step 3: Remove all markdown heading symbols (#, ##, ###, etc.)
        html = html.replace(/^#{1,6}\s*/gm, '');
        html = html.replace(/###+/g, ''); // Remove any standalone ###
        html = html.replace(/##+/g, '');  // Remove any standalone ##
        html = html.replace(/^#+\s*/gm, ''); // Clean up any remaining # at line start
        
        // Step 4: Remove any leftover markdown symbols (clean up any remaining marks)
        html = html.replace(/\*\*/g, ''); // Remove leftover **
        // Remove standalone single asterisks (not inside words or between spaces)
        html = html.replace(/\s+\*\s+/g, ' '); // Remove * surrounded by spaces
        html = html.replace(/\*\s+/g, ''); // Remove * at start of text with space after
        
        // Step 5: Process lists - convert markdown lists to HTML
        const lines = html.split('\n');
        let processedLines = [];
        let inList = false;
        
        lines.forEach((line) => {
            const listMatch = line.match(/^\s*[-*+]\s+(.+)$/);
            if (listMatch) {
                if (!inList) {
                    processedLines.push('<ul>');
                    inList = true;
                }
                processedLines.push('<li>' + listMatch[1] + '</li>');
            } else {
                if (inList) {
                    processedLines.push('</ul>');
                    inList = false;
                }
                if (line.trim()) {
                    processedLines.push(line);
                }
            }
        });
        if (inList) {
            processedLines.push('</ul>');
        }
        html = processedLines.join('\n');
        
        // Step 6: Escape HTML to prevent XSS
        html = html
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        
        // Step 7: Restore our HTML tags (they were escaped)
        html = html
            .replace(/&lt;strong&gt;/g, '<strong>')
            .replace(/&lt;\/strong&gt;/g, '</strong>')
            .replace(/&lt;em&gt;/g, '<em>')
            .replace(/&lt;\/em&gt;/g, '</em>')
            .replace(/&lt;ul&gt;/g, '<ul>')
            .replace(/&lt;\/ul&gt;/g, '</ul>')
            .replace(/&lt;li&gt;/g, '<li>')
            .replace(/&lt;\/li&gt;/g, '</li>');
        
        // Step 8: Handle line breaks
        html = html.replace(/\n\n+/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');
        
        // Step 9: Wrap content in paragraphs
        if (html.trim() && !html.trim().startsWith('<ul>') && !html.trim().startsWith('<p>')) {
            html = '<p>' + html + '</p>';
        }
        
        return { __html: html };
    };

    return (
        <div className="app-container">
            <header className="header">
                <div className="header-content">
                    <div className="logo-section">
                        <div className="logo-icon">🏥</div>
                        <div className="logo-text">
                            <div className="logo-title">Medical Chatbot</div>
                            <div className="logo-subtitle">AI-Powered Healthcare Assistant</div>
                        </div>
                    </div>
                    <div className="status-indicator">
                        <div className="status-dot"></div>
                        <span>Online</span>
                    </div>
                </div>
            </header>

            <main className="main-content">
                <div className="chat-container">
                    <div className="chat-header">
                        <h2>Ask me anything about medical topics</h2>
                        <p>I'm here to help answer your medical questions based on reliable sources</p>
                    </div>

                    <div className="messages-container">
                        {messages.length === 0 && !isLoading && (
                            <div className="welcome-message">
                                <div className="welcome-icon">👋</div>
                                <div className="welcome-title">Welcome to Medical Chatbot</div>
                                <div className="welcome-description">
                                    I'm your AI-powered medical assistant. Ask me questions about medical conditions, 
                                    treatments, symptoms, and more. I'll provide accurate information based on medical literature.
                                </div>
                                <div style={{ marginTop: '2rem', display: 'flex', flexWrap: 'wrap', gap: '0.75rem', justifyContent: 'center' }}>
                                    {sampleQuestions.map((question, idx) => (
                                        <button
                                            key={idx}
                                            onClick={() => handleSampleQuestion(question)}
                                            style={{
                                                padding: '0.5rem 1rem',
                                                background: 'rgba(255, 255, 255, 0.2)',
                                                border: '1px solid rgba(255, 255, 255, 0.3)',
                                                borderRadius: '20px',
                                                color: 'white',
                                                fontSize: '0.875rem',
                                                cursor: 'pointer',
                                                transition: 'all 0.3s ease'
                                            }}
                                            onMouseOver={(e) => {
                                                e.target.style.background = 'rgba(255, 255, 255, 0.3)';
                                            }}
                                            onMouseOut={(e) => {
                                                e.target.style.background = 'rgba(255, 255, 255, 0.2)';
                                            }}
                                        >
                                            {question}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {messages.map((message, index) => (
                            <div key={index} className={`message ${message.sender}`}>
                                <div className="message-avatar">
                                    {message.sender === 'user' ? '👤' : '🤖'}
                                </div>
                                <div className="message-content">
                                    <div 
                                        dangerouslySetInnerHTML={renderText(message.text)}
                                        style={{ lineHeight: '1.6' }}
                                    ></div>
                                    <div className="message-time">{message.time}</div>
                                </div>
                            </div>
                        ))}

                        {isLoading && (
                            <div className="message assistant">
                                <div className="message-avatar">🤖</div>
                                <div className="message-content">
                                    <div className="loading">
                                        <div className="loading-dot"></div>
                                        <div className="loading-dot"></div>
                                        <div className="loading-dot"></div>
                                    </div>
                                </div>
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>

                    <div className="input-container">
                        {error && (
                            <div className="error-message">
                                ⚠️ {error}
                            </div>
                        )}
                        <div className="input-wrapper">
                            <textarea
                                ref={inputRef}
                                className="input-field"
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyPress={handleKeyPress}
                                placeholder="Type your medical question here..."
                                rows="1"
                                disabled={isLoading}
                            />
                            <button
                                className="send-button"
                                onClick={handleSend}
                                disabled={isLoading || !inputValue.trim()}
                            >
                                {isLoading ? (
                                    <>
                                        <svg className="animate-spin" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Processing...
                                    </>
                                ) : (
                                    <>
                                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" width="20" height="20">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                                        </svg>
                                        Send
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}

ReactDOM.render(<MedicalChatbot />, document.getElementById('root'));

