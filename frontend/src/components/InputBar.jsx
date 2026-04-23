import { useState, useRef } from 'react'

export default function InputBar({ onSend, loading, disabled }) {
  const [text, setText] = useState('')
  const textareaRef = useRef(null)

  const handleSend = () => {
    if (!text.trim() || loading || disabled) return
    onSend(text.trim())
    setText('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInput = (e) => {
    setText(e.target.value)
    const ta = textareaRef.current
    if (ta) {
      ta.style.height = 'auto'
      ta.style.height = Math.min(ta.scrollHeight, 140) + 'px'
    }
  }

  const placeholder = disabled
    ? 'Index loading... please wait'
    : 'আপনার প্রশ্ন লিখুন... / Type your question... (Enter to send)'

  return (
    <div style={styles.container}>
      <div style={styles.inputRow}>
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled || loading}
          rows={1}
          style={{
            ...styles.textarea,
            opacity: (disabled || loading) ? 0.5 : 1,
            cursor: (disabled || loading) ? 'not-allowed' : 'text',
          }}
        />
        <button
          onClick={handleSend}
          disabled={!text.trim() || loading || disabled}
          style={{
            ...styles.sendBtn,
            opacity: (!text.trim() || loading || disabled) ? 0.4 : 1,
            cursor: (!text.trim() || loading || disabled) ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? '⏳' : '➤'}
        </button>
      </div>
      <p style={styles.hint}>
        Enter = Send • Shift+Enter = New line
      </p>
    </div>
  )
}

const styles = {
  container: {
    padding: '12px 20px 16px',
    background: '#1e293b',
    borderTop: '1px solid #334155',
    flexShrink: 0,
  },
  inputRow: {
    display: 'flex',
    gap: '10px',
    alignItems: 'flex-end',
  },
  textarea: {
    flex: 1,
    background: '#0f172a',
    border: '1px solid #334155',
    borderRadius: '12px',
    padding: '12px 16px',
    color: '#f1f5f9',
    fontSize: '15px',
    lineHeight: '1.5',
    resize: 'none',
    outline: 'none',
    fontFamily: 'inherit',
    minHeight: '48px',
    maxHeight: '140px',
  },
  sendBtn: {
    width: '48px',
    height: '48px',
    borderRadius: '12px',
    background: '#1d4ed8',
    border: 'none',
    color: 'white',
    fontSize: '20px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  hint: {
    fontSize: '11px',
    color: '#475569',
    marginTop: '6px',
    textAlign: 'center',
  },
}