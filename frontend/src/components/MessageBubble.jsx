import ReactMarkdown from 'react-markdown'

export default function MessageBubble({ message }) {
  if (!message || !message.role) return null
  
  const isUser = message.role === 'user'

  return (
    <div style={{
      ...styles.row,
      justifyContent: isUser ? 'flex-end' : 'flex-start',
    }}>
      {!isUser && (
        <div style={styles.avatar}>⚖️</div>
      )}

      <div style={{
        ...styles.bubble,
        ...(isUser ? styles.userBubble : styles.botBubble),
        ...(message.isError ? styles.errorBubble : {}),
      }}>
        {isUser ? (
          <p style={styles.userText}>{message.text}</p>
        ) : (
          <div style={styles.botContent}>
            <ReactMarkdown
              components={{
                p: ({ children }) => <p style={styles.p}>{children}</p>,
                h1: ({ children }) => <h1 style={styles.h}>{children}</h1>,
                h2: ({ children }) => <h2 style={styles.h}>{children}</h2>,
                h3: ({ children }) => <h3 style={styles.h3}>{children}</h3>,
                ul: ({ children }) => <ul style={styles.ul}>{children}</ul>,
                ol: ({ children }) => <ol style={styles.ol}>{children}</ol>,
                li: ({ children }) => <li style={styles.li}>{children}</li>,
                strong: ({ children }) => <strong style={styles.strong}>{children}</strong>,
                code: ({ children }) => <code style={styles.code}>{children}</code>,
                hr: () => <hr style={styles.hr} />,
              }}
            >
              {message.text}
            </ReactMarkdown>
          </div>
        )}

        {!isUser && message.lang && (
          <span style={styles.langBadge}>
            {message.lang === 'bn' ? '🇧🇩 বাংলা' : '🇬🇧 English'}
          </span>
        )}
      </div>
    </div>
  )
}

const styles = {
  row: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '10px',
  },
  avatar: {
    fontSize: '22px',
    marginTop: '4px',
    flexShrink: 0,
  },
  bubble: {
    maxWidth: '78%',
    borderRadius: '16px',
    padding: '12px 16px',
    wordBreak: 'break-word',
  },
  userBubble: {
    background: '#1d4ed8',
    borderBottomRightRadius: '4px',
  },
  botBubble: {
    background: '#1e293b',
    borderBottomLeftRadius: '4px',
    border: '1px solid #334155',
  },
  errorBubble: {
    background: '#450a0a',
    border: '1px solid #991b1b',
  },
  userText: {
    color: '#f1f5f9',
    fontSize: '15px',
    lineHeight: '1.6',
  },
  botContent: {
    color: '#e2e8f0',
  },
  p: {
    fontSize: '15px',
    lineHeight: '1.7',
    marginBottom: '10px',
    color: '#e2e8f0',
  },
  h: {
    fontSize: '17px',
    fontWeight: '700',
    color: '#f1f5f9',
    marginBottom: '10px',
    marginTop: '14px',
    borderBottom: '1px solid #334155',
    paddingBottom: '6px',
  },
  h3: {
    fontSize: '15px',
    fontWeight: '600',
    color: '#cbd5e1',
    marginBottom: '8px',
    marginTop: '12px',
  },
  ul: {
    paddingLeft: '20px',
    marginBottom: '10px',
  },
  ol: {
    paddingLeft: '20px',
    marginBottom: '10px',
  },
  li: {
    fontSize: '14px',
    lineHeight: '1.7',
    marginBottom: '4px',
    color: '#e2e8f0',
  },
  strong: {
    color: '#f8fafc',
    fontWeight: '600',
  },
  code: {
    background: '#0f172a',
    padding: '2px 6px',
    borderRadius: '4px',
    fontSize: '13px',
    color: '#7dd3fc',
    fontFamily: 'monospace',
  },
  hr: {
    border: 'none',
    borderTop: '1px solid #334155',
    margin: '12px 0',
  },
  langBadge: {
    display: 'inline-block',
    marginTop: '10px',
    fontSize: '11px',
    color: '#64748b',
    background: '#0f172a',
    padding: '2px 8px',
    borderRadius: '10px',
  },
}