import { useCallback, useEffect, useState } from "react"
import { authApi } from "@/lib/api/auth"
import { usersApi } from "@/lib/api/users"
import { getApiErrorMessage, setAuthToken } from "@/lib/http"
import type { AuthSession, Token, UserCreate, UserLogin, VerifyOTP } from "@/lib/types"

const STORAGE_KEY = "Bangladesh-Legal-Advisor-auth-session"

function tokenToSession(token: Token): AuthSession {
  return {
    accessToken: token.access_token,
    tokenType: token.token_type,
    user: token.user,
  }
}

function saveSession(session: AuthSession | null) {
  if (typeof window === "undefined") return

  if (session) {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session))
  } else {
    window.localStorage.removeItem(STORAGE_KEY)
  }
}

export function useAuth() {
  const [session, setSession] = useState<AuthSession | null>(null)
  const [isRestoring, setIsRestoring] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isVerifying, setIsVerifying] = useState(false)
  const [isResending, setIsResending] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    async function restoreSession() {
      try {
        const stored = window.localStorage.getItem(STORAGE_KEY)

        if (stored) {
          const parsed = JSON.parse(stored) as AuthSession

          if (parsed?.accessToken && parsed?.user?.email) {
            setAuthToken(parsed.accessToken)
            try {
              const freshUser = await usersApi.getMe()
              const freshSession = { ...parsed, user: freshUser }
              setSession(freshSession)
              saveSession(freshSession)
            } catch (err) {
              setSession(null)
              saveSession(null)
              setAuthToken(null)
            }
          } else {
            setSession(null)
            saveSession(null)
            setAuthToken(null)
          }
        }
      } catch {
        setSession(null)
        saveSession(null)
        setAuthToken(null)
      } finally {
        setIsRestoring(false)
      }
    }

    restoreSession()
  }, [])

  const login = useCallback(async (payload: UserLogin) => {
    setError("")
    setIsSubmitting(true)

    try {
      const token = await authApi.login(payload)
      setAuthToken(token.access_token)
      
      const freshUser = await usersApi.getMe()
      const nextSession: AuthSession = {
        accessToken: token.access_token,
        tokenType: token.token_type,
        user: freshUser,
      }

      setSession(nextSession)
      saveSession(nextSession)

      return nextSession
    } catch (error) {
      const message = getApiErrorMessage(error)
      setError(message)
      throw error
    } finally {
      setIsSubmitting(false)
    }
  }, [])

  const signup = useCallback(async (payload: UserCreate) => {
    setError("")
    setIsSubmitting(true)

    try {
      const response = await authApi.signup(payload)
      return response
    } catch (error) {
      const message = getApiErrorMessage(error)
      setError(message)
      throw error
    } finally {
      setIsSubmitting(false)
    }
  }, [])

  const verifyOtp = useCallback(async (payload: VerifyOTP) => {
    setError("")
    setIsVerifying(true)

    try {
      return await authApi.verifyOtp(payload)
    } catch (error) {
      const message = getApiErrorMessage(error)
      setError(message)
      throw error
    } finally {
      setIsVerifying(false)
    }
  }, [])

  const resendOtp = useCallback(async (email: string) => {
    setError("")
    setIsResending(true)

    try {
      const response = await authApi.resendOtp({ email })
      return response
    } catch (error) {
      const message = getApiErrorMessage(error)
      setError(message)
      throw error
    } finally {
      setIsResending(false)
    }
  }, [])

  const logout = useCallback(() => {
    setSession(null)
    setError("")
    setAuthToken(null)
    saveSession(null)
  }, [])

  const forgotPassword = useCallback(async (email: string) => {
    setError("")
    setIsSubmitting(true)

    try {
      const response = await usersApi.forgotPassword({ email })
      return response
    } catch (error) {
      const message = getApiErrorMessage(error)
      setError(message)
      throw error
    } finally {
      setIsSubmitting(false)
    }
  }, [])

  const fetchSession = useCallback(async () => {
    try {
      const freshUser = await usersApi.getMe()
      setSession((prev) => {
        if (!prev) return prev
        const nextSession = { ...prev, user: freshUser }
        saveSession(nextSession)
        return nextSession
      })
    } catch (err) {
      // Ignored
    }
  }, [])

  return {
    session,
    isRestoring,
    isSubmitting,
    isVerifying,
    isResending,
    error,
    clearError: () => setError(""),
    login,
    signup,
    verifyOtp,
    resendOtp,
    logout,
    forgotPassword,
    fetchSession,
  }
}
