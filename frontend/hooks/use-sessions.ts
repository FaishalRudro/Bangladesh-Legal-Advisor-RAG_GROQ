import { useState, useCallback, useEffect } from "react"
import { sessionsApi } from "@/lib/api/sessions"
import { getApiErrorMessage } from "@/lib/http"
import type { SessionSummary, SessionDetailResponse } from "@/lib/types"

export function useSessions(enabled: boolean) {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const [hasMore, setHasMore] = useState(false)
  const [page, setPage] = useState(1)
  const [isLoadingMore, setIsLoadingMore] = useState(false)

  const loadSessions = useCallback(async (reset = true) => {
    if (!enabled) return
    if (reset) {
      setIsLoading(true)
      setPage(1)
    } else {
      setIsLoadingMore(true)
    }
    setError("")
    
    try {
      const currentPage = reset ? 1 : page + 1
      const response = await sessionsApi.getSessions(currentPage, 30) // Use size 30
      
      setSessions((prev) => reset ? response.sessions : [...prev, ...response.sessions])
      setHasMore(response.page * response.size < response.total)
      if (!reset) setPage(currentPage)
    } catch (err) {
      setError(getApiErrorMessage(err))
    } finally {
      if (reset) setIsLoading(false)
      else setIsLoadingMore(false)
    }
  }, [enabled, page])

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  const deleteSession = useCallback(async (id: number) => {
    try {
      await sessionsApi.deleteSession(id)
      setSessions((prev) => prev.filter((s) => s.id !== id))
    } catch (err) {
      setError(getApiErrorMessage(err))
      throw err
    }
  }, [])

  const pinSession = useCallback(async (id: number, isPinned: boolean) => {
    try {
      await sessionsApi.pinSession(id, isPinned)
      setSessions((prev) =>
        prev.map((s) => (s.id === id ? { ...s, is_pinned: isPinned } : s))
      )
      loadSessions() // To keep the list sorted by pinned
    } catch (err) {
      setError(getApiErrorMessage(err))
      throw err
    }
  }, [loadSessions])

  return {
    sessions,
    isLoading,
    isLoadingMore,
    hasMore,
    error,
    loadSessions,
    loadMoreSessions: () => loadSessions(false),
    deleteSession,
    pinSession,
  }
}
