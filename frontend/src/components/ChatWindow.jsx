import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

export default function ChatWindow({ messages, loading, indexReady }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const safeMessages = Array.isArray(messages) ? messages.filter(m => m && m.role) : []

  return (
    <div style={styles.window}>
      {safeMessages.length === 0 && (
        <div style={styles.welcome}>
          {indexReady ? (
            <>
              <div style={styles.welcomeIcon}>⚖️</div>
              <h2 style={styles.welcomeTitle}>Bangladesh Legal Advisor</h2>
              <p style={styles.welcomeText}>
                আপনার আইনি প্রশ্ন করুন — বাংলায় বা ইংরেজিতে।
              </p>
              <p style={styles.welcomeText}>
                Ask your legal question in Bangla or English.
              </p>
              <div style={styles.examples}>
                <p style={styles.exampleLabel}>উদাহরণ / Examples:</p>
                <p style={styles.exampleItem}>• মাতৃত্বকালীন ছুটি কতদিন?</p>
                <p style={styles.exampleItem}>• Is the Digital Security Act still in force?</p>
                <p style={styles.exampleItem}>• What is the punishment for rape in Bangladesh?</p>
              </div>
            </>
          ) : (
            <>
              <div style={styles.welcomeIcon}>⏳</div>
              <h2 style={styles.welcomeTitle}>Loading Index...</h2>
              <p style={styles.welcomeText}>
                RAG index building হচ্ছে। একটু অপেক্ষা করুন।
              </p>
              <div style={styles.spinner} />
            </>
          )}
        </div>
      )}

      {safeMessages.map((msg, i) => (
        <MessageBubble key={i} message={msg} />
      ))}

      {loading && (
        <div style={styles.loadingRow}>
          <div style={styles.loadingBubble}>
            <span className="dot" style={styles.dot} />
            <span className="dot" style={{...styles.dot, animationDelay: '0.2s'}} />
            <span className="dot" style={{...styles.dot, animationDelay: '0.4s'}} />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}

const styles = {
  window: {
    flex: 1,
    overflowY: 'auto',
    padding: '24px 20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  welcome: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1,
    textAlign: 'center',
    padding: '40px 20px',
  },
  welcomeIcon: {
    fontSize: '56px',
    marginBottom: '16px',
  },
  welcomeTitle: {
    fontSize: '22px',
    fontWeight: '700',
    color: '#f1f5f9',
    marginBottom: '12px',
  },
  welcomeText: {
    fontSize: '15px',
    color: '#94a3b8',
    marginBottom: '6px',
  },
  examples: {
    marginTop: '24px',
    background: '#1e293b',
    borderRadius: '12px',
    padding: '16px 24px',
    textAlign: 'left',
    maxWidth: '480px',
    width: '100%',
  },
  exampleLabel: {
    color: '#64748b',
    fontSize: '12px',
    marginBottom: '8px',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  exampleItem: {
    color: '#94a3b8',
    fontSize: '14px',
    marginBottom: '6px',
    lineHeight: '1.5',
  },
  spinner: {
    marginTop: '24px',
    width: '32px',
    height: '32px',
    border: '3px solid #334155',
    borderTop: '3px solid #3b82f6',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  loadingRow: {
    display: 'flex',
    justifyContent: 'flex-start',
    paddingLeft: '8px',
  },
  loadingBubble: {
    background: '#1e293b',
    borderRadius: '16px',
    padding: '12px 18px',
    display: 'flex',
    gap: '6px',
    alignItems: 'center',
  },
  dot: {
    width: '8px',
    height: '8px',
    background: '#64748b',
    borderRadius: '50%',
    display: 'inline-block',
    animation: 'bounce 1.2s infinite',
  },
}