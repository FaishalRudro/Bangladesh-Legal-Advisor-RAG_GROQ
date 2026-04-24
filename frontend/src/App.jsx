import { useState, useEffect } from 'react'
import ChatWindow from './components/ChatWindow'
import InputBar from './components/InputBar'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || ''

export default function App() {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [indexReady, setIndexReady] = useState(false)
  const [statusText, setStatusText] = useState('Index loading, please wait...')

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_URL}/status`)
        if (res.data.ready) {
          setIndexReady(true)
          setStatusText('Ready')
          clearInterval(interval)
        } else if (res.data.error) {
          setStatusText(`Error: ${res.data.error}`)
          clearInterval(interval)
        }
      } catch {
        setStatusText('Cannot connect to backend...')
      }
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  const sendMessage = async (query) => {
    if (!query.trim() || loading || !indexReady) return
    const userMsg = { role: 'user', text: query }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)
    try {
      const res = await axios.post(`${API_URL}/chat`, { query })
      const botMsg = { role: 'bot', text: res.data.answer, lang: res.data.lang }
      setMessages(prev => [...prev, botMsg])
    } catch (err) {
      const errText = err.response?.data?.detail || 'Something went wrong. Try again.'
      setMessages(prev => [...prev, { role: 'bot', text: `❌ ${errText}`, isError: true }])
    } finally {
      setLoading(false)
    }
  }

  const clearHistory = async () => {
    try {
      await axios.post(`${API_URL}/clear-history`)
      setMessages([])
    } catch {
      setMessages([])
    }
  }

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <span style={styles.logo}>⚖️</span>
          <div>
            <h1 style={styles.title}>Bangladesh Legal Advisor</h1>
            <p style={styles.subtitle}>বাংলাদেশ আইনি উপদেষ্টা AI</p>
          </div>
        </div>
        <div style={styles.headerRight}>
          <span style={{
            ...styles.statusDot,
            background: indexReady ? '#22c55e' : '#f59e0b'
          }} />
          <span style={styles.statusText}>
            {indexReady ? 'Ready' : statusText}
          </span>
          {indexReady && messages.length > 0 && (
            <button onClick={clearHistory} style={styles.clearBtn}>
              🗑️ Clear
            </button>
          )}
        </div>
      </header>
      <ChatWindow messages={messages} loading={loading} indexReady={indexReady} />
      <InputBar onSend={sendMessage} loading={loading} disabled={!indexReady} />
    </div>
  )
}

const styles = {
  app: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    maxWidth: '900px',
    margin: '0 auto',
    background: '#0f172a',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '16px 24px',
    background: '#1e293b',
    borderBottom: '1px solid #334155',
    flexShrink: 0,
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  logo: { fontSize: '32px' },
  title: {
    fontSize: '18px',
    fontWeight: '700',
    color: '#f1f5f9',
  },
  subtitle: {
    fontSize: '12px',
    color: '#94a3b8',
    marginTop: '2px',
  },
  headerRight: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  statusDot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    display: 'inline-block',
  },
  statusText: {
    fontSize: '13px',
    color: '#94a3b8',
  },
  clearBtn: {
    marginLeft: '12px',
    padding: '6px 12px',
    background: '#334155',
    border: 'none',
    borderRadius: '6px',
    color: '#e2e8f0',
    cursor: 'pointer',
    fontSize: '13px',
  },
}